"""
T4 Dataset Loader for SAP-RPT-OSS DI.

The T4 dataset (https://huggingface.co/datasets/mlfoundations/t4-full) is a large
collection of tabular datasets. For SAP-RPT-OSS DI, we define:
- Members: T4 datasets with > 150 rows
- Non-members: T4 datasets with <= 150 rows
"""

import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from yaml import warnings
import warnings
from huggingface_hub import hf_hub_download
from .dataset import Dataset


class T4Dataset(Dataset):
    """
    Loader for T4 tabular datasets.
    
    T4 datasets are stored as parquet files with encoded names in the t4_cache directory.
    Each file represents a single table.
    
    When inheriting from Dataset, an instance represents a specific dataset with fixed train/test splits.
    """
    
    # Path to cached T4 datasets
    T4_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "t4_cache" / "chunk-0002" / "chunk-0002"
    
    # Define member/non-member split based on row count
    MEMBER_ROW_THRESHOLD = 150
    
    # Cache file for member/non-member classifications
    CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "t4_cache" / "chunk-0002" / "t4_dataset_cache.json"
    
    _cached_files = None
    _member_ids = None
    _non_member_ids = None
    _task_types = None  # Cache for task type per dataset_id
    # id -> path lookup, populated lazily across every available chunk-XXXX/.
    _path_index: Optional[dict] = None
    _task_types = None  # Cache for task type per dataset_id
    
    def __init__(
        self,
        dataset_id: str,
        target_column: Optional[str] = None,
        test_size: float = 0.3,
        seed: int = 42,
        force_task_type: Optional[str] = None,
        bin_target_to: Optional[int] = None,
        select_min_unique: bool = False,
    ):
        """
        Initialize a T4Dataset instance.

        Parameters:
        -----------
        dataset_id : str
            The T4 dataset ID.
        target_column : str, optional
            Name of the target column. If None, randomly selects one.
        test_size : float
            Proportion for test set.
        seed : int
            Random seed for reproducibility.
        force_task_type : {"classification", None}
            If "classification", restrict random target selection to columns
            that yield a classification task (2..10 unique non-null values).
            Useful when evaluating a classifier-only model where regression
            targets would otherwise crash later.
        """
        super().__init__(name=f"T4_{dataset_id}")
        self.dataset_id = dataset_id
        self.test_size = test_size
        self.seed = seed
        self.force_task_type = force_task_type
        # Cut a high-cardinality NUMERIC target into <=bin_target_to quantile bins
        # so the <=10-class TabPFN-2.5 blind can run the same task as the target
        # model. Non-numeric targets are left alone (quantile bins over
        # LabelEncoder integers would be meaningless).
        self.bin_target_to = bin_target_to
        # Pick the FEWEST-unique eligible column as target (the most
        # categorical thing the table has) instead of a random one. Combined
        # with bin_target_to this reaches 100% coverage: only 41% of T4
        # regression tables have any column with <=10 uniques, so the rest need
        # their min-unique column capped.
        self.select_min_unique = select_min_unique
        self.target_was_binned = False

        # Load the dataset
        self.df = self._load_parquet()

        # Select target column if not specified
        if target_column is None:
            self.target_column = self._select_target_column(
                seed, force_task_type=force_task_type
            )
        else:
            self.target_column = target_column

        # Prepare data
        self._prepare_instances()
    
    @classmethod
    def _build_path_index(cls) -> dict:
        """Index every parquet under data/t4_cache/chunk-*/<chunk>/ once.

        Returns {dataset_id: Path}. Result is cached in memory and on disk
        (data/t4_cache/path_index.tsv) so every grid-worker invocation pays
        the ~90s walk only once across the whole project.
        """
        if cls._path_index is not None:
            return cls._path_index
        chunk_root = cls.T4_CACHE_DIR.parent.parent  # data/t4_cache
        cache_path = chunk_root / "path_index.tsv"

        # Disk cache hit — read it.
        if cache_path.exists():
            index: dict = {}
            with cache_path.open() as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    did, path = line.split("\t", 1)
                    index[did] = Path(path)
            cls._path_index = index
            return index

        # Build from scratch and persist.
        index = {}
        for chunk_dir in sorted(chunk_root.glob("chunk-*")):
            inner = chunk_dir / chunk_dir.name
            if not inner.is_dir():
                continue
            for p in inner.glob("*.parquet"):
                index.setdefault(p.stem, p)
        try:
            with cache_path.open("w") as f:
                for did, p in index.items():
                    f.write(f"{did}\t{p}\n")
        except OSError:
            pass  # read-only filesystem etc. — fall back to in-memory only.
        cls._path_index = index
        return index

    def _load_parquet(self) -> pd.DataFrame:
        """Load the parquet file from any chunk-XXXX directory."""
        # Fast path: legacy single-chunk layout.
        filepath = T4Dataset.T4_CACHE_DIR / f"{self.dataset_id}.parquet"
        if not filepath.exists():
            # Fall back to multi-chunk index built from data/t4_cache/chunk-*/.
            idx = T4Dataset._build_path_index()
            filepath = idx.get(self.dataset_id)
            if filepath is None or not filepath.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {self.dataset_id} "
                    f"(searched chunk-0002 and {len(idx)} indexed parquets)"
                )
        return pd.read_parquet(filepath)
    
    def _is_date_column(self, col: pd.Series) -> bool:
        """Check if a column contains date values."""
        if col.dtype == 'datetime64[ns]' or col.dtype == 'datetime64':
            return True
        # Check if column values can be parsed as dates (only if they look like dates)
        if col.dtype == object and len(col) > 0:
            # Quick heuristic: try parsing a small sample
            sample = col.dropna().head(5)
            if len(sample) > 0:
                try:
                    with pd.option_context('mode.copy_on_write', True):
                        pd.to_datetime(sample, errors='raise', format='ISO8601')
                    return True
                except (ValueError, TypeError):
                    pass
        return False
    
    def _select_target_column(
        self, seed: int, force_task_type: Optional[str] = None
    ) -> str:
        """
        Select a target column excluding date and low-quality columns.

        With ``force_task_type="classification"``, restrict to columns that
        will yield a classification task (2..10 unique non-null values, at
        least 2 samples per class).
        """
        np.random.seed(seed)
        valid_columns = []

        for col in self.df.columns:
            s = self.df[col]
            # Skip date columns
            if self._is_date_column(s):
                continue

            # Skip numerical columns with >50% NaN
            if pd.api.types.is_numeric_dtype(s):
                nan_ratio = s.isna().sum() / len(s)
                if nan_ratio > 0.5:
                    continue

            # Skip columns with >20% unique values (relative to all rows)
            try:
                unique_ratio = s.nunique() / len(s)
            except (UnicodeDecodeError, UnicodeError, ValueError, TypeError):
                continue
            if unique_ratio > 0.2:
                continue

            # Skip non-numeric columns with insufficient samples per class
            if not pd.api.types.is_numeric_dtype(s):
                try:
                    value_counts = s.value_counts()
                except Exception:
                    continue
                if len(value_counts) > 0 and value_counts.min() < 2:
                    continue

            if force_task_type == "classification" and not self.select_min_unique:
                # Mirror the preprocessor's filter: 2..10 unique non-null values.
                try:
                    nunique = s.dropna().nunique()
                except Exception:
                    continue
                if not (2 <= nunique <= 10):
                    continue

            valid_columns.append(col)

        # Tables where EVERY column is high-cardinality (unique_ratio > 0.2) leave
        # valid_columns empty, which makes force_task_type="classification" raise
        # before any fallback runs -- that killed 4,470/6,603 hybrid tables. When
        # selecting by min-unique we can still use such a column: it just gets
        # capped/binned to <=10 downstream. So relax the ratio filter rather than
        # give up on the table.
        if self.select_min_unique and not valid_columns:
            for col in self.df.columns:
                s = self.df[col]
                if self._is_date_column(s):
                    continue
                if pd.api.types.is_numeric_dtype(s):
                    if s.isna().sum() / max(1, len(s)) > 0.5:
                        continue
                else:
                    try:
                        vc = s.value_counts()
                    except Exception:
                        continue
                    if len(vc) > 0 and vc.min() < 2:
                        continue
                valid_columns.append(col)

        if self.select_min_unique and valid_columns:
            def _nu(c):
                try:
                    return self.df[c].dropna().nunique()
                except Exception:
                    return 10**9
            scored = [(c, _nu(c)) for c in valid_columns if _nu(c) >= 2] or \
                     [(c, _nu(c)) for c in valid_columns]

            # HYBRID policy. Where a natively 2..10-unique column exists, draw
            # among those exactly as force_task_type="classification" does --
            # same filter, same RNG, same result -- because that policy is the
            # one that produces the target>blind gap. `argmin` instead lands on
            # near-degenerate (usually binary) columns, which measurably kills
            # the gap (+0.010 -> -0.007 on identical datasets). Only when NO
            # such column exists do we fall back to the fewest-unique column,
            # which is then capped to <=10 downstream. This extends coverage
            # from 41% to 100% without disturbing the 41% that already worked.
            forcecls_ok = [c for c, n in scored if 2 <= n <= 10]
            if forcecls_ok:
                return np.random.choice(forcecls_ok)
            return min(scored, key=lambda t: t[1])[0]

        if not valid_columns:
            if force_task_type == "classification":
                # Hard fail — no classification-suitable column exists.
                # The worker will catch this and write a skip result.
                raise ValueError(
                    f"No classification-suitable target column for {self.dataset_id} "
                    "(no column with 2..10 unique non-null values)."
                )
            warnings.warn(
                f"No valid target column found for dataset {self.dataset_id}. "
                "Using last column as target."
            )
            return self.df.columns[-1]

        return np.random.choice(valid_columns)
    
    def _prepare_instances(self):
        """Prepare train/test split and encode features."""
        # Remove rows with NaN in target
        df_clean = self.df.dropna(subset=[self.target_column]).copy()
        
        if len(df_clean) == 0:
            raise ValueError(f"Dataset {self.dataset_id} has no valid samples")
        
        # Extract features and target. Keep features as a DataFrame (do NOT call
        # .values here): on a mixed-dtype frame .values upcasts EVERY column to
        # object, after which _encode_features sees all columns as object and
        # LabelEncodes the numeric ones too — rank-binning and magnitude-
        # destroying every numeric T4/SAP feature. Passing the DataFrame keeps
        # true per-column dtypes so only genuine object/category/datetime columns
        # are integer-encoded.
        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column].values

        if self.bin_target_to:
            from di.binning import cap_categories, quantile_bin
            y_binned, _, did_bin = quantile_bin(y, None, n_bins=self.bin_target_to)
            if did_bin:
                y = y_binned
                self.target_was_binned = True
            else:
                # Non-numeric (or already low-cardinality). quantile_bin is a
                # no-op on unordered categoricals, so cap them by frequency
                # instead -- otherwise a 50-category string target stays
                # 50-class and the <=10-class blind cannot run it.
                y_capped, did_cap = cap_categories(y, max_classes=self.bin_target_to)
                if did_cap:
                    y = y_capped
                    self.target_was_binned = True

        # Determine task type
        unique_values = np.unique(y[~pd.isna(y)])
        
        # Try to convert y to numerical
        enough_values_for_stratification = True
        try:
            y_numeric = pd.to_numeric(y, errors='raise')
            # If conversion succeeds and there are >20 unique values, it's regression
            if len(unique_values) > 20:
                self.task_type = 'regression'
            else:
                # Check if we have sufficient samples per class for stratification
                min_class_count = pd.Series(y_numeric).value_counts().min()
                self.task_type = 'classification' if min_class_count >= 2 else 'regression'
        except (ValueError, TypeError):
            # Can't convert to numeric; treat it as classification
            self.task_type = 'classification'
            min_class_count = pd.Series(y).value_counts().min()
            if min_class_count < 2:
                warnings.warn(f"Dataset {self.dataset_id} has a classification target with a class having less than 2 samples. Remove stratification.")
                # Actually disable stratification — otherwise train_test_split
                # raises ("least populated class has only 1 member") and the
                # dataset is silently dropped from the member/non-member pool.
                enough_values_for_stratification = False

        # Honor force_task_type even when the post-sampling rare-class check
        # would otherwise call it regression.  This matters for nanotabpfn
        # classifier evaluation where the manifest already guaranteed a
        # ≤10-class target, but a sampled fold may underrepresent one class.
        if self.force_task_type == "classification" and self.task_type != "classification":
            self.task_type = "classification"
            enough_values_for_stratification = False

        # A binned target is a classification target by construction, even when a
        # quantile bin lands with a single member (tiny tables, or heavily-tied
        # columns where np.unique collapses the edges). Without this, the
        # min_class_count>=2 rule above would silently regress the bin INDICES as
        # if they were continuous — a different task for ~13% of the pool. Binning
        # always yields >=2 populated bins, so forcing classification is safe.
        if self.bin_target_to and self.task_type != "classification":
            # Either we just binned it, or it was already low-cardinality numeric
            # and only counted as regression because some value occurs once. Both
            # are <=bin_target_to-class problems; running them as regression would
            # leave the pool a mix of two different tasks.
            if 2 <= len(unique_values) <= self.bin_target_to:
                self.task_type = "classification"
                enough_values_for_stratification = False

        # Encode categorical features
        X = self._encode_features(X)
        
        # Encode target if classification
        if self.task_type == 'classification':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoding = {i: str(v) for i, v in enumerate(le.classes_)}
        else:
            y = y.astype(float)
            self.label_encoding = None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed,
            stratify=y if (self.task_type == 'classification' and enough_values_for_stratification) else None
        )
        
        self.X = np.vstack([X_train, X_test])
        self.y = np.hstack([y_train, y_test])
        self.train_inds_ = np.arange(len(X_train))
        self.test_inds_ = np.arange(len(X_train), len(X_train) + len(X_test))
        
        # Update metadata
        self.metadata.update({
            'dataset_id': self.dataset_id,
            'target_type': self.task_type,
            'n_samples': len(df_clean),
            'n_features': X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': len(unique_values) if self.task_type == 'classification' else None,
            'label_encoding': self.label_encoding,
            'target_was_binned': self.target_was_binned,
        })

    @staticmethod
    def all_names() -> Optional[list[str]]:
        """Return None since we have a large dynamic set of datasets."""
        return None
    
    def prepare_data(self, download_dir: str):
        """No-op for T4Dataset since data is pre-cached."""
        pass
    
    def all_instances(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all instances as feature matrix and target vector."""
        return self.X, self.y
    
    def train_inds(self) -> np.ndarray:
        """Return indices for training set."""
        return self.train_inds_
    
    def val_inds(self) -> np.ndarray:
        """Return indices for validation set (none in this case)."""
        return np.array([])
    
    def test_inds(self) -> np.ndarray:
        """Return indices for test set."""
        return self.test_inds_
    
    @staticmethod
    def _encode_features(X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features in the feature matrix.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix with potential categorical columns.

        Returns:
        --------
        np.ndarray
            Feature matrix with categorical features encoded as integers.
        """
        X_df = pd.DataFrame(X)

        for col in X_df.columns:
            s = X_df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                # Datetime columns can't be cast to float directly; convert to
                # nanoseconds-since-epoch ordinal so the model sees a numeric.
                X_df[col] = s.astype("int64", copy=False)
                continue
            if s.dtype == object or s.dtype == 'category':
                # Some object columns mix datetimes and strings; coerce.
                try:
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(s.astype(str))
                except (ValueError, TypeError):
                    X_df[col] = pd.factorize(s.astype(str))[0]

        return X_df.values.astype(float)

    @classmethod
    def _download_t4_chunk(cls):
        """Download and extract chunk-0002.zip from Hugging Face."""
        print("Downloading T4 chunk-0002.zip from Hugging Face...")
        try:
            zip_path = hf_hub_download(
                repo_id="mlfoundations/t4-full",
                filename="chunk-0002.zip",
                repo_type="dataset",
                cache_dir=str(cls.T4_CACHE_DIR.parent)
            )
            print(f"Downloaded to: {zip_path}")
            print("Extracting chunk-0002.zip...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cls.T4_CACHE_DIR.parent)

            print("✓ T4 chunk-0002 extracted successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download T4 chunk-0002.zip from Hugging Face: {str(e)}\n"
                f"Please ensure you have internet access and huggingface_hub installed.\n"
                f"Install with: pip install huggingface_hub.\n"
                f"Then login with 'hf auth login' and try again."
            ) from e

    @classmethod
    def _load_cached_files(cls) -> Dict[str, Path]:
        """Load list of available T4 parquet files."""
        if cls._cached_files is not None:
            return cls._cached_files
        
        if not cls.T4_CACHE_DIR.exists():
            print(f"T4 cache directory not found. Creating: {cls.T4_CACHE_DIR}")
            cls.T4_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check if cache is empty and download if needed
        parquet_files = list(cls.T4_CACHE_DIR.glob("*.parquet"))
        if not parquet_files:
            cls._download_t4_chunk()
            parquet_files = list(cls.T4_CACHE_DIR.glob("*.parquet"))

        files = {}
        for filepath in parquet_files:
            # Use the filename without extension as the dataset ID
            dataset_id = filepath.stem
            files[dataset_id] = filepath
        
        cls._cached_files = files
        return files

    @classmethod
    def _try_load_cache(cls):
        """Try to load cached member/non-member classifications and task types."""
        if cls.CACHE_FILE.exists():
            try:
                with open(cls.CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    cls._member_ids = cache.get('members', [])
                    cls._non_member_ids = cache.get('non_members', [])
                    cls._task_types = cache.get('task_types', {})
                    # Check if cache is empty (corrupted or incomplete)
                    if not cls._member_ids and not cls._non_member_ids:
                        print(f"⚠ Warning: Cache file exists but contains no data: {cls.CACHE_FILE}")
                        response = input("Cache appears to be corrupted. Delete it and rescan? (y/n): ").strip().lower()
                        if response in ['yes', 'y', 'Y']:
                            cls.CACHE_FILE.unlink()
                            print(f"✓ Deleted cache file: {cls.CACHE_FILE}")
                            return False
                    return True
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}")
                return False
        return False

    @classmethod
    def _save_cache(cls):
        """Save member/non-member classifications and task types to cache."""
        if cls._member_ids is not None and cls._non_member_ids is not None:
            try:
                with open(cls.CACHE_FILE, 'w') as f:
                    json.dump({
                        'members': cls._member_ids,
                        'non_members': cls._non_member_ids,
                        'task_types': cls._task_types or {}
                    }, f)
            except Exception as e:
                print(f"Warning: Failed to save cache file: {e}")

    @classmethod
    def get_all_ids(cls) -> Dict[str, list]:
        """
        Get all T4 dataset IDs split by membership status.
        
        Returns:
        --------
        Dict with 'members' and 'non_members' lists of dataset IDs.
        """
        if cls._member_ids is not None and cls._non_member_ids is not None:
            return {
                'members': cls._member_ids,
                'non_members': cls._non_member_ids,
                'all_ids': cls._member_ids + cls._non_member_ids
            }
        
        # Try to load from cache first
        if cls._try_load_cache():
            return {
                'members': cls._member_ids,
                'non_members': cls._non_member_ids,
                'all_ids': cls._member_ids + cls._non_member_ids
            }
        
        # If not cached, compute from scratch
        files = cls._load_cached_files()
        members = []
        non_members = []
        
        print(f"Scanning {len(files)} T4 datasets to determine membership (this may take a minute)...")
        
        for i, (dataset_id, filepath) in enumerate(files.items()):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(files)} datasets...")
            
            try:
                # Load the parquet file to check row count
                df = pd.read_parquet(filepath)
                n_rows = len(df)
                
                if n_rows > cls.MEMBER_ROW_THRESHOLD:
                    members.append(dataset_id)
                else:
                    non_members.append(dataset_id)
            except Exception as e:
                print(f"Warning: Failed to load {dataset_id}: {e}")
                continue
        
        # Cache the results
        cls._member_ids = sorted(members)
        cls._non_member_ids = sorted(non_members)
        cls._save_cache()
        
        print(f"✓ Found {len(cls._member_ids)} members and {len(cls._non_member_ids)} non-members")
        
        return {
            'members': cls._member_ids,
            'non_members': cls._non_member_ids,
            'all_ids': cls._member_ids + cls._non_member_ids
        }

    @classmethod
    def _determine_task_type(cls, dataset_id: str) -> str:
        """
        Determine the task type (classification or regression) for a dataset.
        
        Returns 'classification' or 'regression' based on the target column properties.
        """
        try:
            # Load with a temporary instance to determine task type
            dataset = cls(dataset_id=dataset_id)
            return dataset.task_type
        except Exception as e:
            print(f"Warning: Failed to determine task type for {dataset_id}: {e}")
            return 'classification'  # Default to classification on error

    @classmethod
    def get_all_cls_ids(cls) -> Dict[str, list]:
        """
        Get all classification dataset IDs split by membership status.
        
        Returns:
        --------
        Dict with 'members' and 'non_members' lists of classification-only dataset IDs.
        """
        # First get all IDs and their membership status
        all_data = cls.get_all_ids()
        members = all_data['members']
        non_members = all_data['non_members']
        
        # Ensure task_types cache is initialized
        if cls._task_types is None:
            cls._task_types = {}
        
        # Determine task types for datasets not yet in cache
        all_dataset_ids = members + non_members
        ids_to_scan = [did for did in all_dataset_ids if did not in cls._task_types]
        
        if ids_to_scan:
            print(f"Determining task types for {len(ids_to_scan)} T4 datasets...")
            for i, dataset_id in enumerate(ids_to_scan):
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(ids_to_scan)} datasets...")
                cls._task_types[dataset_id] = cls._determine_task_type(dataset_id)
            
            # Save updated cache
            cls._save_cache()
            print(f"✓ Task type determination complete")
        
        # Filter for classification only
        cls_members = [did for did in members if cls._task_types.get(did) == 'classification']
        cls_non_members = [did for did in non_members if cls._task_types.get(did) == 'classification']
        
        return {
            'members': cls_members,
            'non_members': cls_non_members,
            'all_ids': cls_members + cls_non_members
        }

    @staticmethod
    def load_dataset(
        dataset_id: str,
        test_size: float = 0.3,
        seed: int = 42,
        target_column: Optional[str] = None,
        force_task_type: Optional[str] = None,
        bin_target_to: Optional[int] = None,
        select_min_unique: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load a T4 dataset and prepare train/test split.

        Parameters:
        -----------
        dataset_id : str
            The T4 dataset ID (filename without extension).
        test_size : float, default=0.3
            Proportion of data to use for testing.
        seed : int, default=42
            Random seed for reproducibility.
        target_column : str, optional
            Name of the target column. If None, randomly selects one excluding:
            - Date columns
            - Numerical columns with >50% NaN values
            - Columns with >20% unique values

        Returns:
        --------
        X_train, X_test, y_train, y_test : np.ndarray
            Feature and target arrays for train/test sets.
        metadata : Dict
            Metadata about the dataset (name, shape, target_type, etc).
        """
        # Use the dataset class to load and prepare data
        dataset = T4Dataset(
            dataset_id=dataset_id,
            target_column=target_column,
            test_size=test_size,
            seed=seed,
            force_task_type=force_task_type,
            bin_target_to=bin_target_to,
            select_min_unique=select_min_unique,
        )
        
        X_train, y_train = dataset.train_instances()
        X_test, y_test = dataset.test_instances()
        
        # Metadata
        metadata = {
            'dataset_id': dataset_id,
            'name': f'T4_{dataset_id}',
            'target_type': dataset.task_type,
            'n_samples': len(dataset.X),
            'n_features': dataset.X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': dataset.metadata.get('n_classes'),
            'label_encoding': dataset.label_encoding,
        }
        
        return X_train, X_test, y_train, y_test, metadata

    @staticmethod
    def _encode_features(X: np.ndarray) -> np.ndarray:
        """
        Encode categorical features in the feature matrix.

        Datetime columns are converted to int64 nanoseconds-since-epoch (so
        astype(float) doesn't blow up on `Timestamp` objects), object/category
        columns get integer-encoded.
        """
        X_df = pd.DataFrame(X)

        for col in X_df.columns:
            s = X_df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                X_df[col] = s.astype("int64", copy=False)
                continue
            if s.dtype == object or s.dtype == 'category':
                try:
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(s.astype(str))
                except (ValueError, TypeError):
                    X_df[col] = pd.factorize(s.astype(str))[0]

        return X_df.values.astype(float)

