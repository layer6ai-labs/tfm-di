"""Paper names of the signals: ``<Manipulation>_<Signal>_<Reduction>`` with the M/S/R codes
of the paper's manipulation, signal and reduction tables."""
from __future__ import annotations

from di.audit.pools import fam_of

MANIPULATIONS = {
    "base": ("Default", "M1"), "splitsize": ("SplitFraction", "M1"),
    "ctxeqq": ("QueryEqualsContext", "M2"), "ctx": ("RowSampling", "M4"),
    "rowshuf": ("RowShuffle", "M5"), "col": ("FeatureSample", "M6"),
    "noise": ("FeatureNoise", "M8"), "ftx": ("FeatureTransform", "M9"),
    "tshuffle": ("TaskSwap", "M10"), "lperm": ("ClassPermutation", "M11"),
    "mislbl": ("LabelCorruption", "M12"), "rowdup": ("RowDuplication", "M14"),
    "tserum": ("TruthSerum", "M15"), "qleak": ("QueryLeak", "M16"),
    "bwash": ("Brainwash", "M17"), "nnsplit": ("NN", "M18"), "tabdpt_sim": ("Retrieval", "M19"),
    "temp": ("Temperature", "M20"),
}
SIGNALS = {
    "correct_confidence": ("Confidence", "S1"), "max_confidence": ("MaxConfidence", "S2"),
    "prediction_margin": ("Margin", "S3"), "accuracy": ("Accuracy", "S4"), "loss": ("CELoss", "S5"),
    "chi_squared": ("ChiSquared", "S10"), "entropy": ("Entropy", "S11"),
    "kl_divergence": ("KLD", "S12"), "kl2clean": ("KLDClean", "S12"),
    "rkl2clean": ("ReverseKLD", "S13"), "wasserstein": ("Wasserstein", "S14"),
}
REDUCTIONS = {
    "slope": ("Slope", "R2"), "auc": ("AUC", "R3"), "gain": ("Gain", "R4"), "early": ("Early", "R5"),
    "late_gain": ("LateGain", "R6"), "mean": ("Mean", "R7"), "std": ("Std", "R8"), "range": ("Range", "R9"),
}
DEFAULT_REDUCTION = ("Default", "R1")


def describe(signal: str) -> dict[str, str]:
    """Paper label and codes of one signal column; raises ``KeyError`` on an unknown token."""
    fam = fam_of(signal)
    manip, mcode = MANIPULATIONS[fam]
    rest = signal[len(fam) + 1:] if signal != fam else ""
    if fam == "mislbl" and rest.startswith("random_"):
        rest = rest[len("random_"):]
    red, rcode = DEFAULT_REDUCTION
    for suffix in sorted(REDUCTIONS, key=len, reverse=True):
        if rest.endswith("_" + suffix):
            red, rcode = REDUCTIONS[suffix]
            rest = rest[:-len(suffix) - 1]
            break
    sig, scode = SIGNALS[rest]
    return {"feature": f"{manip}_{sig}_{red}", "manipulation": mcode, "signal": scode,
            "reduction": rcode}
