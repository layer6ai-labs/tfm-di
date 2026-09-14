"""
RealTabPFN-2.5 training dataset IDs from OpenML only. Kaggle training datasets not included.

Paper: https://arxiv.org/abs/2511.08667 see Appendix C
"""

# 20 OpenML datasets used to train RealTabPFN-2.5. 43 used in total, others are from Kaggle.
MEMBER_IDS_REALTABPFN25 = [
    1459, # start openml
    251,
    137,
    40668,
    1471,
    43551,
    1044,
    41787,
    1476,
    23512,
    44201,
    43904,
    43617,
    41671,
    901,
    43923,
    44226,
    1568,
    46676,
    43039 # end openml
]

# 51 TabArena datasets, which RealTabPFN-2.5 did not use for training.
TABARENA_IDS_REALTABPFN25 = [
    46904,
    46905,
    46906,
    46907,
    46908,
    46910,
    46911,
    46912,
    46913,
    46915,
    46916,
    46917,
    46918,
    46919,
    46920,
    46921,
    46922,
    46923,
    46924,
    46927,
    46928,
    46929,
    46930,
    46931,
    46932,
    46933,
    46934,
    46935,
    46937,
    46938,
    46939,
    46940,
    46941,
    46942,
    46980,
    46969,
    46947,
    46949,
    46950,
    46952,
    46953,
    46954,
    46955,
    46956,
    46958,
    46960,
    46961,
    46962,
    46963,
    46964,
    46979,
]

# Combined non-member IDs (backwards-compatible alias)
NON_MEMBER_IDS_REALTABPFN25 = TABARENA_IDS_REALTABPFN25

def get_label(dataset_id: int) -> int:
    """Get membership label (1 for member, 0 for non-member)."""
    return dataset_id in MEMBER_IDS_REALTABPFN25


def get_all_ids():
    """
    Get all labeled dataset IDs (members + non-members) as tuples with family.
    
    Returns a dict with 'members' and 'non_members' as lists of (family, dataset_id) tuples.
    """
    return {
        'members': [('openml', id) for id in MEMBER_IDS_REALTABPFN25],
        'non_members': [('openml', id) for id in NON_MEMBER_IDS_REALTABPFN25],
        'all_ids': [('openml', id) for id in MEMBER_IDS_REALTABPFN25 + NON_MEMBER_IDS_REALTABPFN25]
    }


if __name__ == "__main__":
    all_ids = get_all_ids()
    print(f"RealTabPFN-2.5 Training Datasets: {len(all_ids['members'])} datasets")
    print(f"Non-member Datasets: {len(all_ids['non_members'])} datasets")
    print(f"Total Labeled: {len(all_ids['all_ids'])} datasets")

    print(f"\nSample member IDs: {all_ids['members'][:10]}")
    print(f"Sample non-member IDs: {all_ids['non_members'][:10]}")
