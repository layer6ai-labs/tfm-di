"""
Dataset selection logic for DI.

This module acts as a dispatcher to retrieve member and non-member dataset IDs
based on the model being attacked, with optional filtering by dataset family.
"""

from typing import Dict, List, Optional
from .datasets import tabdpt_training_ids, sap_rpt_oss_training_ids, realtabpfn_training_ids, realtabpfn25_training_ids, nanotabpfn_training_ids
from .datasets.t4 import T4Dataset


SUPPORTED_MODELS = ['tabdpt', 'sap-rpt-oss', 'realtabpfn', 'realtabpfn25', 'nanotabpfn']
SUPPORTED_FAMILIES = ['openml', 't4', 't4-cls', 'synthetic', 'all']


def get_model_datasets(model_name: str, family_name: Optional[str] = None) -> Dict[str, List]:
    """
    Get member and non-member dataset IDs for a specific model.

    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'tabdpt', 'sap-rpt-oss')
    family_name : str, optional
        Dataset family to filter by ('openml' or 't4').
        If None, returns datasets from all available families for the model.

    Returns:
    --------
    Dict with 'members' and 'non_members' lists of (family, dataset_id) tuples.
    
    Examples:
    ---------
    # Get all TabDPT datasets (OpenML members + non-members from multiple families)
    get_model_datasets('tabdpt')
    # Returns: {'members': [('openml', 1138), ...], 'non_members': [('openml', 3), ('t4', 'abc'), ...]}
    
    # Get only OpenML datasets for TabDPT
    get_model_datasets('tabdpt', family_name='openml')
    # Returns: {'members': [('openml', 1138), ...], 'non_members': [('openml', 3), ...]}
    
    # Get only T4 datasets for TabDPT (all non-members since TabDPT doesn't have T4 members)
    get_model_datasets('tabdpt', family_name='t4')
    # Returns: {'members': [], 'non_members': [('t4', 'hash1'), ('t4', 'hash2'), ...]}
    """
    model_lower = model_name.lower()
    
    if model_lower not in [m.lower() for m in SUPPORTED_MODELS]:
        raise ValueError(f"Unknown model: {model_name}. Supported models: {SUPPORTED_MODELS}")
    
    if family_name is not None and family_name.lower() not in [f.lower() for f in SUPPORTED_FAMILIES]:
        raise ValueError(f"Unknown family: {family_name}. Supported families: {SUPPORTED_FAMILIES}")
    
    # Normalize family name if provided
    family_filter = family_name.lower() if family_name else None
    if family_filter == 'all':
        family_filter = None  # Treat 'all' as no filter
    
    if model_lower == 'tabdpt':
        return _get_tabdpt_datasets(family_filter)
    elif model_lower == 'sap-rpt-oss':
        return _get_sap_rpt_oss_datasets(family_filter)
    elif model_lower == 'realtabpfn':
        return _get_realtabpfn_datasets(family_filter)
    elif model_lower == 'realtabpfn25':
        return _get_realtabpfn25_datasets(family_filter)
    elif model_lower == 'nanotabpfn':
        return _get_nanotabpfn_datasets(family_filter)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}")


def _get_tabdpt_datasets(family_filter: Optional[str] = None) -> Dict[str, List]:
    """
    Get TabDPT member and non-member datasets, optionally filtered by family.
    
    TabDPT members: OpenML (123 datasets)
    TabDPT non-members: OpenML (all other) + T4 (all)
    
    Note: CC18 and CTR23 are subsets of OpenML and included in the OpenML family.
    """
    # Start with TabDPT's native OpenML datasets
    tabdpt_ids = tabdpt_training_ids.get_all_ids()
    
    # Extract raw IDs from tuples for filtering
    members = tabdpt_ids['members']  # Already [('openml', id), ...]
    non_members = tabdpt_ids['non_members']  # Already [('openml', id), ...]
    
    # If family filter is specified, only return that family
    if family_filter is not None:
        if family_filter == 'openml':
            return {
                'members': members,
                'non_members': non_members,
            }
        elif family_filter == 't4':
            # TabDPT has no T4 members, all T4 datasets are non-members
            t4_datasets = T4Dataset.get_all_ids()
            return {
                'members': [],
                'non_members': [('t4', id) for id in t4_datasets['all_ids']],
            }
    
    # No family filter: return datasets from all families
    # Add non-members from other families
    all_non_members = list(non_members)  # Start with OpenML non-members
    
    # Add all T4 datasets as non-members
    t4_datasets = T4Dataset.get_all_ids()
    all_non_members.extend([('t4', id) for id in t4_datasets['all_ids']])
    
    if len(members) == 0:
        raise ValueError("No member datasets found. DI cannot be run.")
    elif len(all_non_members) == 0:
        raise ValueError("No non-member datasets found. DI cannot be run.")
    
    return {
        'members': members,
        'non_members': all_non_members,
    }


def _get_sap_rpt_oss_datasets(family_filter: Optional[str] = None) -> Dict[str, List]:
    """
    Get SAP-RPT-OSS member and non-member datasets, optionally filtered by family.
    
    SAP-RPT-OSS members: T4 (16,199 datasets with >150 rows)
    SAP-RPT-OSS non-members: T4 (3,922 datasets with <=150 rows) + OpenML (all)
    
    Note: CC18 and CTR23 are subsets of OpenML and included in the OpenML family.
    """
    # Start with SAP-RPT-OSS's native T4 datasets
    sap_ids = sap_rpt_oss_training_ids.get_all_ids()
    
    members = sap_ids['members']  # Already [('t4', id), ...]
    non_members = sap_ids['non_members']  # Already [('t4', id), ...]
    
    # If family filter is specified, only return that family
    if family_filter is not None:
        if family_filter == 't4':
            return {
                'members': members,
                'non_members': non_members,
            }
        elif family_filter == 'openml':
            # SAP-RPT-OSS has no OpenML members, all OpenML datasets are non-members
            # Use full OpenML list (including CC18 and CTR23 as subsets)
            tabdpt_ids = tabdpt_training_ids.get_all_ids()
            all_openml = tabdpt_ids['members'] + tabdpt_ids['non_members']
            return {
                'members': [],
                'non_members': all_openml,
            }
    
    # No family filter: return datasets from all families
    # Add non-members from other families
    all_non_members = list(non_members)  # Start with T4 non-members
    
    # Add all OpenML datasets (including CC18 and CTR23) as non-members
    tabdpt_ids = tabdpt_training_ids.get_all_ids()
    all_openml = tabdpt_ids['members'] + tabdpt_ids['non_members']
    all_non_members.extend(all_openml)
    
    if len(members) == 0:
        raise ValueError("No member datasets found. DI cannot be run.")
    elif len(all_non_members) == 0:
        raise ValueError("No non-member datasets found. DI cannot be run.")
    
    return {
        'members': members,
        'non_members': all_non_members,
    }

def _get_realtabpfn_datasets(family_filter: Optional[str] = None) -> Dict[str, List]:
    """
    Get RealTabPFN member and non-member datasets, optionally filtered by family.
    RealTabPFN only supports classification, not regression.
    
    RealTabPFN members: OpenML (37 datasets) + Kaggle (To be added, 34)
    RealTabPFN non-members: OpenML (all other classification datasets)
    """
    # Start with RealTabPFN's native OpenML datasets
    realtabpfn_ids = realtabpfn_training_ids.get_all_ids()
    
    # Extract raw IDs from tuples for filtering
    members = realtabpfn_ids['members']  # Already [('openml', id), ...]
    non_members = realtabpfn_ids['non_members']  # Already [('openml', id), ...]
    
    # If family filter is specified, only return that family
    if family_filter is not None:
        if family_filter == 'openml':
            return {
                'members': members,
                'non_members': non_members,
            }
        elif family_filter == 't4-cls':
            # RealTabPFN has no T4 members, all T4 datasets are non-members
            # Can only use classification
            t4_datasets = T4Dataset.get_all_cls_ids()
            return {
                'members': [],
                'non_members': [('t4', id) for id in t4_datasets['all_ids']],
            }
    
    # No family filter: return datasets from all families
    # Add non-members from other families
    all_non_members = list(non_members)  # Start with OpenML non-members
    
    # Add all T4 cls datasets as non-members
    t4_datasets = T4Dataset.get_all_cls_ids()
    all_non_members.extend([('t4', id) for id in t4_datasets['all_ids']])
    
    if len(members) == 0:
        raise ValueError("No member datasets found. DI cannot be run.")
    elif len(all_non_members) == 0:
        raise ValueError("No non-member datasets found. DI cannot be run.")
    
    return {
        'members': members,
        'non_members': all_non_members,
    }
    
def _get_realtabpfn25_datasets(family_filter: Optional[str] = None) -> Dict[str, List]:
    """
    Get RealTabPFN-25 member and non-member datasets, optionally filtered by family.
    
    RealTabPFN-25 members: OpenML (20 datasets)
    RealTabPFN-25 non-members: OpenML (51 TabArena other) + T4 (all)
    
    Note: TabArena is a subset of OpenML and included in the OpenML family.
    """
    # Start with RealTabPFN-25's native OpenML datasets
    realtabpfn25_ids = realtabpfn25_training_ids.get_all_ids()
    
    # Extract raw IDs from tuples for filtering
    members = realtabpfn25_ids['members']  # Already [('openml', id), ...]
    non_members = realtabpfn25_ids['non_members']  # Already [('openml', id), ...]
    
    # If family filter is specified, only return that family
    if family_filter is not None:
        if family_filter == 'openml':
            return {
                'members': members,
                'non_members': non_members,
            }
        elif family_filter == 't4':
            # RealTabPFN-25 has no T4 members, all T4 datasets are non-members
            t4_datasets = T4Dataset.get_all_ids()
            return {
                'members': [],
                'non_members': [('t4', id) for id in t4_datasets['all_ids']],
            }
    
    # No family filter: return datasets from all families
    # Add non-members from other families
    all_non_members = list(non_members)  # Start with OpenML non-members
    
    # Add all T4 datasets as non-members
    t4_datasets = T4Dataset.get_all_ids()
    all_non_members.extend([('t4', id) for id in t4_datasets['all_ids']])
    
    if len(members) == 0:
        raise ValueError("No member datasets found. DI cannot be run.")
    elif len(all_non_members) == 0:
        raise ValueError("No non-member datasets found. DI cannot be run.")
    
    return {
        'members': members,
        'non_members': all_non_members,
    }

def _get_nanotabpfn_datasets(family_filter: Optional[str] = None) -> Dict[str, List]:
    """
    Get NanoTabPFN member and non-member datasets, optionally filtered by family.
    
    NanoTabPFN members: Synthetic train split (50k classification + 640k regression)
    NanoTabPFN non-members: Synthetic holdout split + OpenML (all) + T4 (all)
    
    Note: The 'synthetic' family refers to the synthetic train/holdout splits.
    """
    # Start with NanoTabPFN's native synthetic datasets
    nano_ids = nanotabpfn_training_ids.get_all_ids()
    
    members = nano_ids['members']  # Train split: [('synthetic', (task, split, idx)), ...]
    non_members = nano_ids['non_members']  # Holdout split
    
    # If family filter is specified, only return that family
    if family_filter is not None:
        if family_filter == 'synthetic':
            # Return only synthetic members (train) and non-members (holdout)
            return {
                'members': members,
                'non_members': non_members,
            }
        elif family_filter == 'openml':
            # NanoTabPFN has no OpenML members, all OpenML datasets are non-members
            tabdpt_ids = tabdpt_training_ids.get_all_ids()
            all_openml = tabdpt_ids['members'] + tabdpt_ids['non_members']
            return {
                'members': [],
                'non_members': all_openml,
            }
        elif family_filter == 't4':
            # NanoTabPFN has no T4 members, all T4 datasets are non-members
            t4_datasets = T4Dataset.get_all_ids()
            return {
                'members': [],
                'non_members': [('t4', id) for id in t4_datasets['all_ids']],
            }
    
    # No family filter: return datasets from all families
    # Add non-members from other families to synthetic non-members
    all_non_members = list(non_members)  # Start with synthetic holdout
    
    # Add all OpenML datasets as non-members
    tabdpt_ids = tabdpt_training_ids.get_all_ids()
    all_openml = tabdpt_ids['members'] + tabdpt_ids['non_members']
    all_non_members.extend(all_openml)
    
    # Add all T4 datasets as non-members
    t4_datasets = T4Dataset.get_all_ids()
    all_non_members.extend([('t4', id) for id in t4_datasets['all_ids']])
    
    if len(members) == 0:
        raise ValueError("No member datasets found. DI cannot be run.")
    elif len(all_non_members) == 0:
        raise ValueError("No non-member datasets found. DI cannot be run.")
    
    return {
        'members': members,
        'non_members': all_non_members,
    }

