"""
Ground truth training dataset IDs for SAP-RPT-OSS (ContextTab).

For SAP-RPT-OSS, we use the T4 dataset as the proxy for training membership:
- Members: T4 datasets with > 150 rows
- Non-members: All other available datasets (including CC18, CTR23, OpenML)

This is a dynamic definition based on the T4 dataset structure.
"""

from .t4 import T4Dataset


def get_label(dataset_id: str) -> int:
    """
    Get membership label for a T4 dataset.
    
    Parameters:
    -----------
    dataset_id : str
        The dataset ID.
    
    Returns:
    --------
    int : 1 for member, 0 for non-member.
    """
    filepath = T4Dataset.T4_CACHE_DIR / f"{dataset_id}.parquet"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    import pandas as pd
    df = pd.read_parquet(filepath)
    return 1 if len(df) > T4Dataset.MEMBER_ROW_THRESHOLD else 0


def get_all_ids():
    """
    Get all labeled dataset IDs (members + non-members).
    
    Returns:
    --------
    Dict with 'members', 'non_members', and 'all_ids' lists of (family, id) tuples.
    """
    t4_ids = T4Dataset.get_all_ids()
    return {
        'members': [('t4', id) for id in t4_ids['members']],
        'non_members': [('t4', id) for id in t4_ids['non_members']],
        'all_ids': [('t4', id) for id in t4_ids['all_ids']]
    }


def get_member_ids():
    """Get all member dataset IDs (>150 rows)."""
    return get_all_ids()['members']


def get_non_member_ids():
    """Get all non-member dataset IDs (<=150 rows)."""
    return get_all_ids()['non_members']


if __name__ == "__main__":
    all_ids = get_all_ids()
    print(f"SAP-RPT-OSS Training Datasets: {len(all_ids['members'])} datasets")
    print(f"Non-member Datasets: {len(all_ids['non_members'])} datasets")
    print(f"Total Labeled: {len(all_ids['all_ids'])} datasets")

    print(f"\nSample member IDs: {all_ids['members'][:10]}")
    print(f"Sample non-member IDs: {all_ids['non_members'][:10]}")
