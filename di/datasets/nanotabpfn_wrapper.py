"""
Wrapper classes for NanoTabPFN models to make them compatible with DI framework.

NanoTabPFN is an open-source transformer-based foundation model for tabular data.
This wrapper provides a TabDPT-compatible interface for use in the DI framework.
"""

import json
from typing import Optional
import numpy as np
import torch
from pathlib import Path

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor, get_feature_preprocessor
from tfmplayground.model import NanoTabPFNModel
from pfns.bar_distribution import FullSupportBarDistribution


_DEFAULT_CLASSIFIER_ARCH = {
    "num_attention_heads": 6,
    "embedding_size": 192,
    "mlp_hidden_size": 768,
    "num_layers": 6,
    "num_outputs": 10,
}


def _load_arch_sidecar(weight_path: Path) -> dict:
    """Look for `<stem>.arch.json` next to the weight file; fall back to defaults."""
    sidecar = weight_path.with_suffix(".arch.json")
    if not sidecar.exists():
        # Tolerate stems like 'foo.pt' -> 'foo.arch.json' vs 'foo_latest.arch.json'.
        alt = weight_path.parent / f"{weight_path.stem}.arch.json"
        if alt.exists():
            sidecar = alt
    if sidecar.exists():
        with sidecar.open() as f:
            meta = json.load(f)
        return {k: meta[k] for k in _DEFAULT_CLASSIFIER_ARCH if k in meta}
    return dict(_DEFAULT_CLASSIFIER_ARCH)


class NanoTabPFNClassifierWrapper:
    """
    Wrapper around NanoTabPFN Classifier to match TabDPT interface.
    
    The wrapper loads a pretrained NanoTabPFN model and provides the same
    interface as TabDPT for use in the DatasetInferenceAttack framework.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = False,
        model_weight_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the classifier wrapper.

        Parameters:
        -----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        verbose : bool, default=False
            Whether to print debug information.
        model_weight_path : str, optional
            Path to pretrained model weights. If None, uses default location.
        **kwargs :
            Additional kwargs from the TabDPT-compatible interface (e.g.
            ``normalizer``) that NanoTabPFN does not use. Ignored so the grid
            pipeline can pass them uniformly across backends.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model_weight_path = model_weight_path or "models/nanotabpfn/classifier.pt"

        # Model will be loaded lazily on first fit()
        self._model = None
        self._fitted = False

    def _load_model(self):
        """Load the pretrained model from disk."""
        weight_path = Path(self.model_weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. "
                f"Please ensure the model has been trained and saved."
            )
        
        if self.verbose:
            print(f"Loading classifier from {weight_path}")
        
        # Create model with known architecture, load weights, wrap in classifier interface
        state_dict = torch.load(weight_path, map_location=self.device)
        arch = _load_arch_sidecar(weight_path)
        if self.verbose:
            print(f"NanoTabPFN arch: {arch}")
        nano_model = NanoTabPFNModel(**arch)
        nano_model.load_state_dict(state_dict)
        self._model = NanoTabPFNClassifier(model=nano_model, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NanoTabPFNClassifierWrapper":
        """
        Fit the classifier on training data.

        For NanoTabPFN, this loads the pretrained weights and stores the training data 
        to use as context during inference (no actual retraining occurs).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns:
        --------
        self
        """
        if self._model is None:
            self._load_model()
        
        # Store training data
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        
        # Fit the underlying model (this sets up X_train, y_train, feature_preprocessor, etc.)
        self._model.fit(self.X_train_, self.y_train_)
        self._fitted = True
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        """Extract raw logits for X using the current internal model context."""
        x = np.concatenate((self._model.X_train, self._model.feature_preprocessor.transform(X)))
        y = self._model.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self._model.model((x, y), single_eval_pos=len(self._model.X_train),
                                   num_mem_chunks=self._model.num_mem_chunks).squeeze(0)
            out = out[:, :self._model.num_classes]
            return out.cpu().numpy()

    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 1.0,
        context_size: Optional[int] = None,
        seed: Optional[int] = None,
        return_logits: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Predict class probabilities or logits.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        temperature : float, default=1.0
            Temperature for softmax scaling.
        context_size : int, optional
            Number of training examples to use as context. If None,
            uses all available training data.
        seed : int, optional
            Random seed for feature column permutation.
        return_logits : bool, default=False
            If True, return raw logits instead of probabilities.

        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities or logits.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before making predictions")

        X = np.asarray(X, dtype=np.float32)

        # Determine effective context window
        n_context = len(self.X_train_)
        if context_size is not None:
            n_context = min(context_size, n_context)
        ctx_X = self.X_train_[:n_context]
        ctx_y = self.y_train_[:n_context]

        # Apply feature column permutation if seed is specified
        if seed is not None:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(X.shape[1])
            ctx_X = ctx_X[:, perm]
            X = X[:, perm]

        # Temporarily override internal model state if context differs from fitted state
        need_temp_ctx = n_context < len(self.X_train_) or seed is not None
        if need_temp_ctx:
            orig_X_train = self._model.X_train
            orig_y_train = self._model.y_train
            orig_num_classes = self._model.num_classes
            orig_preprocessor = self._model.feature_preprocessor

            new_preprocessor = get_feature_preprocessor(ctx_X)
            self._model.feature_preprocessor = new_preprocessor
            self._model.X_train = new_preprocessor.fit_transform(ctx_X)
            self._model.y_train = ctx_y
            self._model.num_classes = int(max(ctx_y)) + 1

        try:
            logits = self._get_logits(X)
        finally:
            if need_temp_ctx:
                self._model.X_train = orig_X_train
                self._model.y_train = orig_y_train
                self._model.num_classes = orig_num_classes
                self._model.feature_preprocessor = orig_preprocessor

        if return_logits:
            return logits

        # Apply temperature to logits and compute softmax
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    def ensemble_predict_proba(
        self,
        X: np.ndarray,
        n_ensembles: int = 1,
        temperature: float = 1.0,
        context_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Predict class probabilities using multiple forward passes.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        n_ensembles : int, default=1
            Number of ensemble iterations.
        temperature : float, default=1.0
            Temperature for softmax scaling.
        context_size : int, optional
            Size of context to use.
        seed : int, optional
            Random seed for reproducibility.

        Returns:
        --------
        np.ndarray of shape (n_samples, n_classes)
            Averaged ensemble probabilities.
        """
        probabilities_list = []
        
        for i in range(n_ensembles):
            current_seed = seed + i if seed is not None else None
            probs = self.predict_proba(
                X,
                temperature=temperature,
                context_size=context_size,
                seed=current_seed,
            )
            probabilities_list.append(probs)
        
        # Average across ensemble
        avg_probabilities = np.mean(probabilities_list, axis=0)
        return avg_probabilities

    @staticmethod
    def download_weights():
        """
        Download model weights if needed.

        NanoTabPFN weights need to be trained separately. This is a no-op.
        """
        return None


class NanoTabPFNRegressorWrapper:
    """
    Wrapper around NanoTabPFN Regressor to match TabDPT interface.
    
    The wrapper loads a pretrained NanoTabPFN regression model and provides
    the same interface as TabDPT for use in the DI framework.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        verbose: bool = False,
        model_weight_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the regressor wrapper.

        Parameters:
        -----------
        device : str, optional
            Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        verbose : bool, default=False
            Whether to print debug information.
        model_weight_path : str, optional
            Path to pretrained model weights. If None, uses default location.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model_weight_path = model_weight_path or "models/nanotabpfn/regressor.pt"
        
        # Derive bucket path from model path
        weight_stem = Path(model_weight_path).stem if model_weight_path else "regressor"
        weight_dir = Path(model_weight_path).parent if model_weight_path else Path("models/nanotabpfn")
        self.buckets_path = weight_dir / f"{weight_stem}_buckets.pt"
        
        # Models will be loaded lazily on first fit()
        self._model = None
        self._buckets = None
        self._fitted = False

    def _load_model(self):
        """Load the pretrained model and bucket edges from disk."""
        weight_path = Path(self.model_weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. "
                f"Please ensure the model has been trained and saved."
            )
        
        buckets_path = Path(self.buckets_path)
        if not buckets_path.exists():
            raise FileNotFoundError(
                f"Bucket edges not found at {buckets_path}. "
                f"Please ensure the model was trained with bucket computation."
            )
        
        if self.verbose:
            print(f"Loading regressor from {weight_path}")
            print(f"Loading buckets from {buckets_path}")
        
        # Load bucket edges
        self._buckets = torch.load(buckets_path, map_location=self.device)
        bucket_dist = FullSupportBarDistribution(self._buckets)
        
        # Create model and load weights
        self._model = NanoTabPFNRegressor(bucket_distribution=bucket_dist)
        state_dict = torch.load(weight_path, map_location=self.device)
        self._model.model.load_state_dict(state_dict)
        self._model.model.to(self.device)
        self._model.model.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NanoTabPFNRegressorWrapper":
        """
        Fit the regressor on training data.

        For NanoTabPFN, this just stores the training data to use as context
        during inference (no actual retraining occurs).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns:
        --------
        self
        """
        if not self._fitted:
            self._load_model()
        
        # Store training data as context
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = np.asarray(y, dtype=np.float32)
        self._fitted = True
        
        return self

    def predict(
        self,
        X: np.ndarray,
        context_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Predict regression targets.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.
        context_size : int, optional
            Number of training examples to use as context.
        seed : int, optional
            Random seed for feature permutation.

        Returns:
        --------
        np.ndarray of shape (n_samples,)
            Predicted regression targets.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        
        # The NanoTabPFNRegressor internally stores X_train and uses it as context
        if context_size is not None and context_size < len(self.X_train_):
            # Save original training data
            orig_X_train = self._model.X_train.copy()
            orig_y_train = self._model.y_train.copy()
            
            # Use only first context_size examples
            limited_X = self.X_train_[:context_size]
            limited_y = self.y_train_[:context_size]
            
            # Refit the model with limited data
            self._model.fit(limited_X, limited_y)
            predictions = self._model.predict(X)
            
            # Restore original training data
            self._model.fit(orig_X_train, orig_y_train)
        else:
            # Use all training data (default)
            predictions = self._model.predict(X)
        
        return predictions

    @staticmethod
    def download_weights():
        """
        Download model weights if needed.

        NanoTabPFN weights need to be trained separately. This is a no-op.
        """
        return None
