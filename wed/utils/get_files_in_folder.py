from os.path import join, isfile
from os import listdir

def get_files(foldr_path: str) -> list[str]:
    return [join(foldr_path, f) for f in listdir(foldr_path) if isfile(join(foldr_path, f))]