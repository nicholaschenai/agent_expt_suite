from ..base_data_pipeline import BaseDataPipeline
from ..MBPP_Plus.MBPP_Plus_data_utils import postprocess_mbpp_plus
from ..MBPP_Plus.MBPP_Plus_Dataset import MBPPPlusDataset


class MBPPPlusDataPipeline(BaseDataPipeline):
    def __init__(self, dataset_name="MBPP_Plus", train=True, **kwargs):
        super().__init__(dataset_name, train, **kwargs)

    def postprocess(self, **kwargs):
        postprocess_mbpp_plus(**kwargs)

    def get_dataset(self):
        """
        Retrieves the dataset and dataloader.
        This method initializes the MBPPPlusDataset with the provided parameters
        and returns the dataset and dataloader. Currently, the dataloader is not
        implemented and will return None.
        Returns:
            tuple: A tuple containing the dataset (MBPPPlusDataset) and dataloader (None).
        """
        dataset, dataloader = None, None

        dataset = MBPPPlusDataset(
            dataset_name=self.dataset_name,
            eval_later=self.eval_later,
            use_public_tests=self.use_public_tests
        )
        return dataset, dataloader
