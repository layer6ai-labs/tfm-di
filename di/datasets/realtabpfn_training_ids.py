"""
RealTabPFN training dataset IDs from OpenML only. Kaggle training datasets not included.

Paper: https://arxiv.org/abs/2511.08667 see Appendix A
"""

# 37 OpenML datasets used to train RealTabPFN. 71 used in total, others are from Kaggle.
# All classification. RealTabPFN does not support regression.
MEMBER_IDS_REALTABPFN = [
    4135, # start openml
    1459,
    44234,
    251,
    137,
    41434,
    4154,
    40668,
    1471,
    151,
    846,
    43551,
    1044,
    41787,
    1477,
    1476,
    23512,
    821,
    843,
    44201,
    43904,
    40679,
    1120,
    43617,
    41671,
    901,
    1046,
    43923,
    44226,
    1568,
    45067,
    32,
    4534,
    44122,
    46676,
    1461,
    43039 # end openml
]

# Eval datasets, which RealTabPFN did not use for training.
# All classification. RealTabPFN does not support regression.
EVAL_IDS_REALTABPFN = [
    41156,
    40981,
    1464,
    40975,
    40701,
    23,
    31,
    40670,
    188,
    1475,
    4538,
    41143,
    1067,
    3,
    41144,
    12,
    1487,
    1049,
    41145,
    1489,
    1494,
    40900,
    40984,
    40982,
    41146,
    54,
    40983,
    40498,
    181,
]


# Combined non-member IDs (backwards-compatible alias)
NON_MEMBER_IDS_REALTABPFN = EVAL_IDS_REALTABPFN


def get_label(dataset_id: int) -> int:
    """Get membership label (1 for member, 0 for non-member)."""
    return dataset_id in MEMBER_IDS_REALTABPFN


def get_all_ids():
    """
    Get all labeled dataset IDs (members + non-members) as tuples with family.
    
    Returns a dict with 'members' and 'non_members' as lists of (family, dataset_id) tuples.
    """
    return {
        'members': [('openml', id) for id in MEMBER_IDS_REALTABPFN],
        'non_members': [('openml', id) for id in NON_MEMBER_IDS_REALTABPFN],
        'all_ids': [('openml', id) for id in MEMBER_IDS_REALTABPFN + NON_MEMBER_IDS_REALTABPFN]
    }


if __name__ == "__main__":
    all_ids = get_all_ids()
    print(f"RealTabPFN Training Datasets: {len(all_ids['members'])} datasets")
    print(f"Non-member Datasets: {len(all_ids['non_members'])} datasets")
    print(f"Total Labeled: {len(all_ids['all_ids'])} datasets")

    print(f"\nSample member IDs: {all_ids['members'][:10]}")
    print(f"Sample non-member IDs: {all_ids['non_members'][:10]}")
