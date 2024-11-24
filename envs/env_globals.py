from .env_factory import env_factory
from .env_registry import env_registry


def init_code_env(**kwargs):
    """
    Initializes the code environment by creating an instance of the environment class.

    This function uses the provided keyword arguments to create an instance of the environment
    class using the `env_factory` function. The created instance is then assigned to the global
    variable `task_env`.

    we make the env external to agents so that we can reuse them across different task_envs 
    or do multiagent simulations

    Args:
        **kwargs: Arbitrary keyword arguments that are passed to the `env_factory` function and
                  the environment class constructor.

    Returns:
        None
    """
    global task_env
    env_cls = env_factory(**kwargs)
    task_env = env_cls(**kwargs)
