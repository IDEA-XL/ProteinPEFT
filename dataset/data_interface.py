"""https://github.com/westlake-repl/SaProt/blob/main/dataset/data_interface.py"""

import os
# register function as a wrapper for all models
def register_dataset(cls):
	global now_cls
	now_cls = cls
	return cls

now_cls = None
class DataInterface:
	@classmethod
	def init_dataset(cls, dataset_py_path: str, **kwargs):
		"""

        Args:
           dataset_py_path: Path to dataset file
           **kwargs: Kwargs for model initialization

        Returns: Corresponding model
        """
		sub_dirs = dataset_py_path.split(os.sep)
		cmd = f"from {'.' + '.'.join(sub_dirs[:-1])} import {sub_dirs[-1]}"
		exec(cmd)
		
		return now_cls(**kwargs)