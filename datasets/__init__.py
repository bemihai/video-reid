from datasets.dsets import Mars, iLidsVid

_all_datasets = {
    'mars': Mars,
    'ilidsvid': iLidsVid,
}


def init_dataset(name, **kwargs):
    """
    Initializes a registered ``Dataset``.

    Args:
        name (str): dataset registered key.
    """
    _datasets = list(_all_datasets.keys())
    if name not in _datasets:
        raise ValueError('Invalid Dataset key. Expected to be one of {}'.format(_datasets))
    return _all_datasets[name](**kwargs)


def register_dataset(name, dataset):
    """
    Registers a new dataset to the datasets list.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new Dataset class.
    """
    global _all_datasets
    current_datasets = list(_all_datasets.keys())
    if name in current_datasets:
        raise ValueError(
            'The given name already exists, please choose another name excluding {}'.format(current_datasets)
        )
    _all_datasets[name] = dataset
