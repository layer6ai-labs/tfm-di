"""
Ground truth member/non-member dataset IDs for NanoTabPFN.

NanoTabPFN is trained on synthetic data split into 50/50 train/holdout splits:
- Classification: 50,000 train tables + 50,000 holdout tables (50x3)
- Regression: 640,000 train tables + 640,000 holdout tables (50x3)

Member datasets: Tables from the train splits
Non-member datasets: Tables from holdout splits + all OpenML + all T4
"""

from typing import Dict, List, Tuple
from functools import lru_cache


@lru_cache(maxsize=1)
def get_all_ids() -> Dict[str, List[Tuple[str, Tuple[str, str, int]]]]:
    """
    Get all NanoTabPFN member and non-member dataset IDs.

    Returns:
    --------
    Dict with keys:
        - 'members': List of (family, (task_type, split, index)) tuples
                     for train split datasets
        - 'non_members': List of (family, (task_type, split, index)) tuples
                         for holdout split datasets

    Examples:
    ---------
    >>> ids = get_all_ids()
    >>> len(ids['members'])  # ~690,000 (50k classification + 640k regression)
    >>> len(ids['non_members'])  # ~690,000 (50k classification + 640k regression)
    """
    members = []
    non_members = []
    
    # Classification train split: 50,000 tables
    for idx in range(50000):
        members.append(('synthetic', ('classification', 'train', idx)))
    
    # Classification holdout split: 50,000 tables
    for idx in range(50000):
        non_members.append(('synthetic', ('classification', 'holdout', idx)))
    
    # Regression train split: 640,000 tables
    for idx in range(640000):
        members.append(('synthetic', ('regression', 'train', idx)))
    
    # Regression holdout split: 640,000 tables
    for idx in range(640000):
        non_members.append(('synthetic', ('regression', 'holdout', idx)))
    
    return {
        'members': members,
        'non_members': non_members,
    }


# def get_synthetic_ids() -> Dict[str, List[Tuple[str, Tuple[str, str, int]]]]:
#     """
#     Get only synthetic dataset IDs (train + holdout).

#     Returns:
#     --------
#     Dict with keys:
#         - 'members': Synthetic train split IDs
#         - 'non_members': Synthetic holdout split IDs
#     """
#     return get_all_ids()


# def is_member(family: str, task_type: str, split: str, index: int) -> bool:
#     """
#     Check if a dataset is a member (in train split).

#     Parameters:
#     -----------
#     family : str
#         Dataset family ('synthetic')
#     task_type : str
#         Task type ('classification' or 'regression')
#     split : str
#         Split name ('train' or 'holdout')
#     index : int
#         Index within the split

#     Returns:
#     --------
#     bool
#         True if the dataset is in the train split (member), False otherwise
#     """
#     if family != 'synthetic':
#         return False
    
#     if task_type not in ['classification', 'regression']:
#         return False
    
#     return split == 'train'


# def decode_synthetic_id(dataset_id: Tuple[str, str, int]) -> Dict[str, str | int]:
#     """
#     Decode a synthetic dataset ID into its components.

#     Parameters:
#     -----------
#     dataset_id : tuple
#         The dataset ID in form (task_type, split, index)

#     Returns:
#     --------
#     Dict with keys: 'task_type', 'split', 'index'
#     """
#     task_type, split, index = dataset_id
#     return {
#         'task_type': task_type,
#         'split': split,
#         'index': index,
#         'is_member': split == 'train',
#     }
