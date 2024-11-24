from .dataset_registry import dataset_registry

from cognitive_base.utils import get_cls


def data_pipeline_factory(dataset_name="MBPP_Plus", **kwargs):
    """
    Factory function to create a data pipeline instance based on the dataset name.

    Args:
        dataset_name (str): The name of the dataset for which to create the data pipeline. Default is "MBPP_Plus".
        **kwargs: Additional keyword arguments to pass to the data pipeline class.

    Returns:
        An instance of the data pipeline class corresponding to the given dataset name.

    Raises:
        ValueError: If the provided dataset name is not found in the dataset registry.
    """
    if dataset_name in dataset_registry:
        data_pipeline_cls = get_cls(dataset_registry[dataset_name])
        return data_pipeline_cls(dataset_name=dataset_name, **kwargs)
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")
