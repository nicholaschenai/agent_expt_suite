import torch

from torch.utils.data import DataLoader
from typing import List


class BaseDataPipeline:
    """
    A base class for data pipelines used in machine learning experiments.
    Attributes:
        dataset_name (str): The name of the dataset.
        train (bool): Whether the pipeline is for training or testing.
        batch_size (int): The size of data batches.
        problem_dirs (list or None): Directories containing problem data.
        max_len (int): Maximum length of datapoint to manage context.
        curriculum_mode (str): Mode for curriculum.
        debug_mode (bool): Whether to run in debug mode.
        debug_subset (int): Size of the debug subset.
        dataset_type (str): Type of the dataset (e.g., 'pytorch').
        eval_later (bool): Whether to evaluate at the end of inference in bulk.
        use_public_tests (bool): Whether to use public tests.
        kwargs (dict): Additional keyword arguments.
        preprocess_fn (function): Function for preprocessing data.
    Methods:
        preprocess(data, **kwargs):
            Preprocesses the given data using the appropriate preprocessing function.
        postprocess(**kwargs):
            Placeholder method for postprocessing data after inference, for evaluation.
        get_dataset():
            Abstract method to be implemented by subclasses to return the dataset.
        get_dataloader():
            Prepares and returns the dataset and dataloader.
    """
    def __init__(
        self,
        dataset_name,
        train=True,
        batch_size=1,
        problem_dirs=None,
        max_len=36000,
        curriculum_mode="heuristic",
        debug_mode=False,
        debug_subset=50,
        dataset_type='pytorch',
        eval_later=False,
        use_public_tests=False,
        **kwargs
    ):
        self.dataset_name = dataset_name
        self.train = train
        self.batch_size = batch_size
        self.problem_dirs = problem_dirs
        self.max_len = max_len
        self.curriculum_mode = curriculum_mode
        self.debug_mode = debug_mode
        self.debug_subset = debug_subset
        self.dataset_type = dataset_type
        self.eval_later = eval_later
        self.use_public_tests = use_public_tests
        self.kwargs = kwargs
        self.preprocess_fn = generic_preprocess_train if train else generic_preprocess_test
       
    def preprocess(self, data, **kwargs):
        return self.preprocess_fn(data, **kwargs)
    
    def postprocess(self, **kwargs):
        pass

    def filter_dataset(self, dataset, dataloader):
        """
        Base filtering method that can be overridden by derived classes.
        By default, returns the dataset unchanged.
        
        Args:
            dataset: The dataset to filter
            
        Returns:
            filtered_dataset: The filtered dataset
            filter_metadata: Dict containing info about the filtering (e.g., indices kept)
        """
        return dataset, dataloader

    def get_dataset(self):
        """
        Template method that handles the dataset loading and filtering pipeline.
        Derived classes should implement _load_raw_dataset instead of overriding this.
        """
        dataset, dataloader = self._load_raw_dataset()
        
        if dataset is not None:
            dataset, dataloader = self.filter_dataset(dataset, dataloader)
            # Optionally log or store filter_metadata
        return dataset, dataloader
    
    def _load_raw_dataset(self):
        """
        Abstract method to be implemented by derived classes to load the raw dataset.
        """
        raise NotImplementedError

    def get_dataloader(self):
        """
        Retrieves and prepares the dataset and dataloader.
        This method first calls `self.get_dataset()` to obtain the initial dataset and dataloader.
        It then prepares the dataloader using the `prepare_dataloader` function with various parameters.
        Returns:
            tuple: A tuple containing the prepared dataset and dataloader.
        """
        dataset, dataloader = self.get_dataset()

        dataset, dataloader = prepare_dataloader(
            dataset,
            dataloader,
            train=self.train,
            dataset_type=self.dataset_type,
            debug_subset=self.debug_subset,
            debug_mode=self.debug_mode,
            curriculum_mode=self.curriculum_mode,
            batch_size=self.batch_size
        )
        return dataset, dataloader
    
    def attach_to_agent(self, actor):
        _, dataloader = self.get_dataloader()
        actor.dataloader = dataloader


def generic_preprocess_test(full_task, **kwargs):
    """
    Generic preprocessing of data for a given task and dataset.

    Args:
        full_task: The task to preprocess.

    Returns:
        The preprocessed task.
    """
    return full_task


def generic_preprocess_train(full_task, **kwargs):
    """
    Preprocess data for a given task and dataset, done during train or curriculum manager

    Args:
        full_task: The task to preprocess.
        code (str, optional): Code to override the default solution code. Defaults to ''.

    Returns:
        The preprocessed task with the optional code override.
    """
    return full_task


class AccedingSequenceLengthSampler(torch.utils.data.Sampler):
    """
    meant to sort pytorch dataset by length as a proxy for curriculum
    """
    def __init__(self, data: List[str]) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        sizes = torch.tensor([len(x['code']) for x in self.data])
        yield from torch.argsort(sizes).tolist()


def prepare_dataloader(
    dataset,
    dataloader,
    train,
    dataset_type,
    debug_subset,
    debug_mode,
    curriculum_mode,
    batch_size,
):
    """
    Prepares a dataloader for training or evaluation if required.
    also subsamples for debug

    Args:
        dataset (Dataset): The dataset to load data from.
        dataloader (DataLoader): The dataloader to use for loading data.
        train (bool): Whether the dataloader is for training or evaluation.
        dataset_type (str): The type of dataset ('pytorch' or 'hf').
        debug_subset (int): The number of samples to use in debug mode.
        debug_mode (bool): Whether to enable debug mode.
        curriculum_mode (str): The curriculum mode to use ('heuristic' or other).
        batch_size (int): The batch size for the dataloader.

    Returns:
        tuple: A tuple containing the possibly modified dataset and dataloader.
    """
    if dataset_type == 'pytorch':
        if debug_mode and train:
            dataset = torch.utils.data.Subset(dataset, range(debug_subset))
        l_kwargs = {}
        if curriculum_mode == "heuristic" and train:
            l_kwargs['sampler'] = AccedingSequenceLengthSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x, **l_kwargs)
    elif dataset_type == 'hf':
        if dataloader is None:
            dataloader = dataset
        if debug_mode:
            dataloader = dataloader.select(range(debug_subset))
    return dataset, dataloader
