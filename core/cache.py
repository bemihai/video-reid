import os
import pickle


# ------------------------------------------------------------------------------------------------

class Cache:
    """
    Basic caching functionality based on disk storage.
    """

    def __init__(self, root_dir):
        self._root_dir = root_dir


# ------------------------------------------------------------------------------------------------

    def _file_name(self, key):
        return os.path.join(self._root_dir, key + '.pkl')


# ------------------------------------------------------------------------------------------------

    def _ensure_dir(self, key):
        parent_dir = os.path.join(self._root_dir, *key.split('/')[:-1])
        os.makedirs(parent_dir, exist_ok=True)


# ------------------------------------------------------------------------------------------------

    def exists(self, key):
        return os.path.exists(self._file_name(key))


# ------------------------------------------------------------------------------------------------

    def put(self, key, value):
        self._ensure_dir(key)
        pickle.dump(value, open(self._file_name(key), 'wb'))


# ------------------------------------------------------------------------------------------------

    def get(self, key):
        file_name = self._file_name(key)
        if not os.path.exists(self._file_name(key)):
            return None

        return pickle.load(open(file_name, 'rb'))


# ------------------------------------------------------------------------------------------------

