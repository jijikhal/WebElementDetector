from os.path import join, isfile
from os import listdir
import random


def get_files(dir_path: str, *, shuffle: bool = False, seed: int | None = None, extension: str | None = None) -> list[str]:
    """Gets full paths of all files in a provided directory

    Args:
        dir_path (str): Relative or absolute path to a directory.
        shuffle (bool, optional): If set to True, the file order will be randomized. Defaults to False.
        seed (int | None, optional): Seed to use for shuffling. Defaults to None.
        extension (str | None, optional): If specified, only files with such extension will be returned. Defaults to None.

    Returns:
        list[str]: list of paths
    """

    files = [join(dir_path, f) for f in listdir(dir_path) if isfile(
        join(dir_path, f)) and (extension is None or f.endswith(extension))]
    if shuffle:
        random.seed(seed)
        random.shuffle(files)
    return files
