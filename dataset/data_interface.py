"""https://github.com/westlake-repl/SaProt/blob/main/dataset/data_interface.py"""

import os

registered_datasets = {}

def register_dataset(cls):
    registered_datasets[cls.__name__] = cls
    return cls

class DataInterface:
    @classmethod
    def list_registered_datasets(cls):
        return list(registered_datasets.keys())

    @classmethod
    def init_dataset(cls, dataset_py_path: str, **kwargs):
        """
        Initialize a dataset instance.

        Args:
            dataset_py_path (str): Path to the dataset module or class.
                This can be in the format of 'module.dataset_class' or 'module.submodule.dataset_class'.
            **kwargs: Keyword arguments to pass to the dataset class constructor.

        Returns:
            DataInterface: An instance of the requested dataset class.
        """
        try:
            if '.' in dataset_py_path:
                module_path, class_name = dataset_py_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                dataset_cls = getattr(module, class_name)
            else:
                dataset_cls = registered_datasets[dataset_py_path]
        except (ImportError, KeyError, AttributeError):
            # Try to handle the 'module/submodule/dataset_class' format
            parts = dataset_py_path.split('/')
            if len(parts) > 1:
                module_path = 'dataset.' + '.'.join(parts[:-1])
                class_name = parts[-1]
                module = __import__(module_path, fromlist=[class_name])
                dataset_cls = getattr(module, class_name)
            else:
                raise ValueError(f"Invalid dataset path: {dataset_py_path}")

        return dataset_cls(**kwargs)