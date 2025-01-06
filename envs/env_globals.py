from .env_factory import env_factory


def init_code_env(**kwargs):
    """
    Initializes the code environment by creating an instance of the environment class.
    Supports separate train and test environments.

    we make the env external to agents to reuse them across different 
    task_envs or do multiagent simulations

    Args:
        **kwargs: Arbitrary keyword arguments including:
            - train_dataset_name (optional): Name of training dataset environment
            - test_dataset_name (optional): Name of test dataset environment
            - dataset_name (optional): Legacy fallback for both environments
            - do_train (bool): Whether training is being performed
            - do_test (bool): Whether testing is being performed
    """
    global task_env, train_env, test_env
    
    # For backwards compatibility, use dataset_name as fallback
    train_dataset = kwargs.get('train_dataset_name', kwargs.get('dataset_name'))
    test_dataset = kwargs.get('test_dataset_name', kwargs.get('dataset_name'))
    
    if kwargs.get('do_train'):
        train_kwargs = {**kwargs, 'dataset_name': train_dataset}
        env_cls = env_factory(**train_kwargs)
        train_env = env_cls(**train_kwargs)
        task_env = train_env
    
    if kwargs.get('do_test'):
        test_kwargs = {**kwargs, 'dataset_name': test_dataset}
        env_cls = env_factory(**test_kwargs)
        test_env = env_cls(**test_kwargs)
        task_env = test_env

# Initialize global variables
task_env = None
train_env = None
test_env = None
