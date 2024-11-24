from .env_registry import env_registry

from cognitive_base.utils import get_cls


def env_factory(dataset_name="MBPP_Plus", **kwargs):
    """
    Creates and returns an environment instance based on the provided dataset name.

    Args:
        dataset_name (str): The name of the dataset to create the environment for. Defaults to "MBPP_Plus".
        **kwargs: Additional keyword arguments to pass to the environment class constructor.

    Returns:
        object: An instance of the environment class corresponding to the provided dataset name.

    Raises:
        ValueError: If the provided dataset name is not found in the environment registry.
    """
    if dataset_name in env_registry:
        return get_cls(env_registry[dataset_name])
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}")
