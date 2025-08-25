"""Configuration management for experiment comparisons."""
from typing import Dict, List, Tuple
from pathlib import Path
import itertools


class ExperimentConfig:
    """Manages configuration for experiment comparisons."""
    def __init__(
        self, 
        experiment_paths: Dict[str, Dict[str, str]], 
        base_dir: str = ".", 
        output_dir: str = "./analysis_outputs", 
        trajectories_subdir: str = "test_outputs"
        ):
        """Initialize experiment configuration.
        
        Args:
            base_dir: Base directory containing experiment folders
            output_dir: Directory to save analysis outputs
            trajectories_subdir: Subdirectory containing trajectories in each experiment
        """
        self.experiment_paths = experiment_paths
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.trajectories_subdir = trajectories_subdir
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
    def get_experiment_pairs_by_model(self) -> Dict[str, List[Tuple[str, str]]]:
        """Generate all pairwise experiment combinations for each model.
        
        Returns:
            Dictionary mapping model names to lists of experiment pairs
        """
        model_pairs = {}
        
        # Get all available models across experiments
        all_models = set()
        for exp_models in self.experiment_paths.values():
            all_models.update(exp_models.keys())
        
        # For each model, generate pairwise combinations of experiments that have that model
        for model in all_models:
            experiments_with_model = [
                exp_name for exp_name, exp_models in self.experiment_paths.items() 
                if model in exp_models
            ]
            
            # Generate all pairwise combinations
            if len(experiments_with_model) >= 2:
                model_pairs[model] = list(itertools.combinations(experiments_with_model, 2))
        
        return model_pairs
    
    def get_experiment_path(self, experiment_name: str, model_name: str) -> Path:
        """Get the full path to an experiment directory.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            
        Returns:
            Full path to the experiment directory
        """
        if experiment_name not in self.experiment_paths:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        if model_name not in self.experiment_paths[experiment_name]:
            raise ValueError(f"Model {model_name} not found in experiment {experiment_name}")
        
        relative_path = self.experiment_paths[experiment_name][model_name]
        return self.base_dir / relative_path
    
    def get_comparison_output_path(self, exp1_name: str, exp2_name: str, model_name: str) -> Path:
        """Get output path for a specific comparison.
        
        Args:
            exp1_name: First experiment name
            exp2_name: Second experiment name  
            model_name: Model name
            
        Returns:
            Path for storing this comparison's outputs
        """
        comparison_name = f"{exp1_name}_vs_{exp2_name}_{model_name}"
        return self.output_dir / comparison_name
    
    def get_all_comparisons(self) -> List[Dict[str, str]]:
        """Get all experiment comparisons to be performed.
        
        Returns:
            List of comparison configurations with experiment names and model
        """
        comparisons = []
        model_pairs = self.get_experiment_pairs_by_model()
        
        for model, exp_pairs in model_pairs.items():
            for exp1, exp2 in exp_pairs:
                comparisons.append({
                    'model': model,
                    'experiment1': exp1,
                    'experiment2': exp2,
                    'exp1_path': str(self.get_experiment_path(exp1, model)),
                    'exp2_path': str(self.get_experiment_path(exp2, model)),
                    'output_path': str(self.get_comparison_output_path(exp1, exp2, model)),
                    'csv_fname': self.get_csv_fname(exp1, exp2, model),
                    'output_dir': self.output_dir,
                })
        
        return comparisons

    def get_csv_fname(self, exp1: str, exp2: str, model:str) -> str:
        return f"status_differences_{exp1}_vs_{exp2}_{model}.csv"
