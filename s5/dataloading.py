import torch
from pathlib import Path
import os
import shutil
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42) -> ReturnType:
	...


def make_data_loader(dset,
					 dobj,
					 seed: int,
					 batch_size: int=128,
					 shuffle: bool=True,
					 drop_last: bool=True,
					 collate_fn: callable=None):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

	# Create a generator for seeding random number draws.
	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	if dobj is not None:
		assert collate_fn is None
		collate_fn = dobj._collate_fn

	# Generate the dataloaders.
	return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
									   drop_last=drop_last, generator=rng)
									   

def create_rgr_dataset(data_dir, cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  bsz: int = 50,
										  seed: int = 42, clear_cache=False) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating Fly Courtship Regression Dataset")
	from s5.dataloaders.fly import FlyCourtshipRgr
	name = 'rgr_fly'
    
	if clear_cache and os.path.exists(os.path.join(cache_dir, name)):
		print('Clearing cache...')
		shutil.rmtree(os.path.join(cache_dir, name))

	dir_name = os.path.join('./data_s5/', data_dir)

	kwargs = {
		'n_workers': 1,
	}

	dataset_obj = FlyCourtshipRgr(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	d_out = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = len(dataset_obj.vocab)
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, d_out, SEQ_LENGTH, IN_DIM, TRAIN_SIZE
	
	
def create_clf_dataset(data_dir, cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  bsz: int = 50,
										  seed: int = 42, clear_cache=False) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating Fly Courtship Classification Dataset")
	from s5.dataloaders.fly import FlyCourtshipClf
	name = 'clf_fly'

	if clear_cache and os.path.exists(os.path.join(cache_dir, name)):
		print('Clearing cache...')
		shutil.rmtree(os.path.join(cache_dir, name))
        
	dir_name = os.path.join('./data_s5/', data_dir)

	kwargs = {
		'n_workers': 1,
	}

	dataset_obj = FlyCourtshipClf(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	d_out = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = len(dataset_obj.vocab)
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, d_out, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


Datasets = {
	"rgr": create_rgr_dataset,
	"clf": create_clf_dataset,
}
