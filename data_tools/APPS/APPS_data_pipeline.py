from datasets import load_dataset

from ..base_data_pipeline import BaseDataPipeline
from ..APPS.APPS_data_utils import apps_preprocess_train, apps_preprocess_test


class APPSDataPipeline(BaseDataPipeline):
    def __init__(self, dataset_name="APPS", train=True, **kwargs):
        super().__init__(dataset_name, train, **kwargs)
        self.preprocess_fn = apps_preprocess_train if train else apps_preprocess_test

    def _load_raw_dataset(self):
        dataset, dataloader = None, None

        split = 'train' if self.train else 'test'
        if self.dataset_type == 'hf':
            dataset = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
        return dataset, dataloader
