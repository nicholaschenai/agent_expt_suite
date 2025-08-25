import logging

from pathlib import Path
from typing import Dict

from .io import save_status_differences_to_csv, load_evaluation_result
from .mbpp_plus import compare_dict_statuses, calculate_accuracy

logger = logging.getLogger(__name__)

def run_single_comparison(comparison: Dict[str, str]):
    """
    requires experiment config to get comparison data via get_all_comparisons()
    load MBPP plus eval results, finds status differences and saves then
    """
    exp1_name = comparison['experiment1']
    exp2_name = comparison['experiment2']
    model = comparison['model']

    exp1_path = Path(comparison['exp1_path'])
    exp2_path = Path(comparison['exp2_path'])
    
    eval_results1 = load_evaluation_result(exp1_path)
    eval_results2 = load_evaluation_result(exp2_path)

    logger.info(f"{exp1_name} accuracy: {calculate_accuracy(eval_results1):.4f}")
    logger.info(f"{exp2_name} accuracy: {calculate_accuracy(eval_results2):.4f}")

    logger.info(f"Calculating status differences for {exp1_name} vs {exp2_name} ({model})")
    status_groups = compare_dict_statuses(eval_results1, eval_results2, exp1_name, exp2_name)

    exp1_status_col = f"{exp1_name}_status"
    exp2_status_col = f"{exp2_name}_status"

    csv_fname = comparison['csv_fname']
    csv_path = comparison['output_dir'] / csv_fname
    save_status_differences_to_csv(
        status_groups, str(csv_path), exp1_status_col, exp2_status_col
    )
    logger.info(f"Saved status differences to {csv_path}")
