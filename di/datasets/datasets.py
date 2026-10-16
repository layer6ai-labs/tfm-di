"""
Unified dataset management for DI.

This module provides centralized access to all dataset families:
- CC18: Categorical Clustering 2018 (curated OpenML datasets)
- CTR23: Click-Through Rate 2023 (curated OpenML datasets)
- T4: T4 large tabular dataset collection
- OpenML: General OpenML datasets

Each model (TabDPT, SAP-RPT-OSS) specifies which datasets are members/non-members.
"""

from typing import Dict, List
from pathlib import Path
from .tabdpt_training_ids import CC18_IDS_TABDPT, CTR23_IDS_TABDPT


class DatasetManager:
    """Central management of dataset families and their membership definitions."""
    
    # CC18 and CTR23 datasets (standard benchmarks, treated as non-members for all models)
    # These are fixed sets of curated OpenML dataset IDs
    
    @staticmethod
    def get_cc18_ids() -> List[int]:
        """Get CC18 benchmark dataset IDs."""
        return CC18_IDS_TABDPT
    
    @staticmethod
    def get_ctr23_ids() -> List[int]:
        """Get CTR23 benchmark dataset IDs."""
        return CTR23_IDS_TABDPT
    
    @staticmethod
    def get_benchmark_ids() -> List[int]:
        """Get all benchmark dataset IDs (CC18 + CTR23)."""
        return CC18_IDS_TABDPT + CTR23_IDS_TABDPT


def get_membership_label(model: str, dataset_family: str, dataset_id: str) -> int:
    """
    Get membership label for a dataset.
    
    Parameters:
    -----------
    model : str
        The model name ('tabdpt', 'sap-rpt-oss', 'contexttab')
    dataset_family : str
        The dataset family ('openml', 't4', 'cc18', 'ctr23')
    dataset_id : str
        The dataset ID within the family
    
    Returns:
    --------
    int : 1 for member, 0 for non-member
    
    Raises:
    -------
    ValueError if the model or dataset family combination is not supported
    """
    model_lower = model.lower()
    family_lower = dataset_family.lower()
    
    # TabDPT: OpenML datasets (123 members specified), everything else non-member
    if model_lower == 'tabdpt':
        if family_lower == 'openml':
            # Import here to avoid circular imports
            from .tabdpt_training_ids import get_label as tabdpt_get_label
            return tabdpt_get_label(dataset_id)
        else:
            # Non-OpenML datasets are non-members for TabDPT
            return 0
    
    # SAP-RPT-OSS (ContextTab): T4 datasets (members), everything else non-member
    elif model_lower in ['sap-rpt-oss', 'contexttab']:
        if family_lower == 't4':
            # Import here to avoid circular imports
            from .sap_rpt_oss_training_ids import get_label as sap_rpt_get_label
            return sap_rpt_get_label(dataset_id)
        else:
            # Non-T4 datasets are non-members for SAP-RPT-OSS
            return 0
    
    else:
        raise ValueError(f"Unsupported model: {model}")


def get_all_ids(model: str) -> Dict[str, List]:
    """
    Get all labeled dataset IDs (members + non-members) for a model.
    
    Parameters:
    -----------
    model : str
        The model name ('tabdpt', 'sap-rpt-oss', 'contexttab')
    
    Returns:
    --------
    Dict with 'members' and 'non_members' lists
    """
    model_lower = model.lower()
    
    if model_lower == 'tabdpt':
        from .tabdpt_training_ids import get_all_ids as tabdpt_get_all_ids
        return tabdpt_get_all_ids()
    elif model_lower in ['sap-rpt-oss', 'contexttab']:
        from .sap_rpt_oss_training_ids import get_all_ids as sap_rpt_get_all_ids
        return sap_rpt_get_all_ids()
    else:
        raise ValueError(f"Unsupported model: {model}")
