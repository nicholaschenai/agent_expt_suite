"""
Sampling utilities for experimental evaluation
"""
import random
from typing import List, Dict, Tuple

import pandas as pd

from .io import save_status_difference_logs


def sample_status_differences_data(
    csv_path: str,
    sample_size: int = 10,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Sample tasks that have different statuses between two experiments.
    
    Args:
        csv_path: Path to CSV containing status differences
        sample_size: Maximum number of tasks to sample per category
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict containing:
        - 'wrong_to_right': List of sampled task IDs where exp1 failed but exp2 passed
        - 'right_to_wrong': List of sampled task IDs where exp1 passed but exp2 failed
        - 'status_cols': List of status column names found in the CSV
    """
    # Set random seed
    random.seed(random_seed)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get column names from CSV (they should end with _status)
    status_cols = [col for col in df.columns if col.endswith('_status')]
    if len(status_cols) != 2:
        raise ValueError(f"Expected 2 status columns, found {len(status_cols)}")
    
    # Group tasks
    wrong_to_right = df[
        (df[status_cols[0]] == 'fail') & 
        (df[status_cols[1]] == 'pass')
    ]['task_id'].tolist()
    
    right_to_wrong = df[
        (df[status_cols[0]] == 'pass') & 
        (df[status_cols[1]] == 'fail')
    ]['task_id'].tolist()
    
    # Sample tasks
    wrong_to_right = random.sample(wrong_to_right, min(sample_size, len(wrong_to_right)))
    right_to_wrong = random.sample(right_to_wrong, min(sample_size, len(right_to_wrong)))
    
    return {
        'wrong_to_right': wrong_to_right,
        'right_to_wrong': right_to_wrong,
        'status_cols': status_cols
    }


def sample_status_differences(
    csv_path: str,
    exp1_name: str,
    exp2_name: str,
    model_name: str,
    output_base_dir: str,
    sample_size: int = 10,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Wrapper function that samples and saves log files for tasks that have different 
    statuses between two experiments.
    
    Args:
        csv_path: Path to CSV containing status differences
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment  
        model_name: Name of the model
        output_base_dir: Base directory to save outputs
        sample_size: Maximum number of tasks to sample per category
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict containing:
        - 'wrong_to_right': List of sampled task IDs where exp1 failed but exp2 passed
        - 'right_to_wrong': List of sampled task IDs where exp1 passed but exp2 failed
    """
    # Sample the data
    sample_data = sample_status_differences_data(csv_path, sample_size, random_seed)
    
    # Save the log files
    save_status_difference_logs(
        wrong_to_right=sample_data['wrong_to_right'],
        right_to_wrong=sample_data['right_to_wrong'],
        status_cols=sample_data['status_cols'],
        exp1_name=exp1_name,
        exp2_name=exp2_name,
        model_name=model_name,
        output_base_dir=output_base_dir
    )
    
    # Return the sampled task IDs
    return {
        'wrong_to_right': sample_data['wrong_to_right'],
        'right_to_wrong': sample_data['right_to_wrong']
    }
