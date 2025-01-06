import logging
from typing import Tuple, Dict, Any, Union

logger = logging.getLogger("logger")


class BaseCodeEnv:
    """
    A class to represent a coding environment that manages the execution of code snippets
    against a set of tests for specific datasets (e.g., MBPP, APPS).

    Attributes:
        timeout (int): The maximum time allowed for code execution.
        tests (list): A list of test cases for the current task.
        test_prefix (str): Prefix to add before each test case, e.g., setup code or imports.
        use_public_tests (bool): Flag to use public tests instead of private ones.
        public_tests (list): A list of public test cases.
        dataset_name (str): The name of the dataset being used.
        max_tests (int): The maximum number of tests to display in feedback.
        max_chars (int): The maximum number of characters to display per test in feedback.
    """
    def __init__(
        self,
        timeout=10,
        max_display_tests=None,
        max_display_chars=None,
        use_public_tests=False,
        dataset_name="MBPP",
        **kwargs
    ):
        """
        Initializes the CodeEnv with the given parameters.

        Parameters:
            timeout (int): The maximum time allowed for code execution.
            max_display_tests (int): The maximum number of tests to display in feedback.
            max_display_chars (int): The maximum number of characters to display per test in feedback.
            use_public_tests (bool): Flag to use public tests instead of private ones.
            do_train (bool): Flag indicating if the environment is in training mode.
            do_test (bool): Flag indicating if the environment is in testing mode.
            dataset_name (str): The name of the dataset being used.
            **kwargs: Additional keyword arguments.
        """
        # TODO: remove task specific attributes so can parallel
        self.timeout = timeout
        self.private_tests = []
        self.test_prefix = ''
        self.use_public_tests = use_public_tests
        self.public_tests = []
        self.dataset_name = dataset_name
        self.max_tests = max_display_tests
        self.max_chars = max_display_chars
        self.generic_code_env = False

    def _reset(self, task):
        pass
        # raise NotImplementedError

    def reset(self, task):
        self._reset(task)
        if 'public_test_list' in task:
            self.public_tests = task['public_test_list']

    def _step(self, full_code, use_public_tests=False):
        raise NotImplementedError

    def step(self, full_code: str, use_public_tests: bool = False) -> Tuple[str, Union[bool, int, float], bool, Dict[str, Any]]:
        """
        Executes the given code against the test cases and generates feedback.
        """
        obs, reward, done, info = self._step(full_code, use_public_tests)
        logger.info(f'obs: {obs}\nreward: {reward}\ndone: {done}\ninfo: {info}')
        return obs, reward, done, info
