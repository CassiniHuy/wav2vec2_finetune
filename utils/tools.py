import os, glob, itertools, json
from types import SimpleNamespace
from configparser import ConfigParser
from ast import literal_eval
from typing import List, Union

def dict_eval(d):
    for k, v in d.items():
        try:  # number/tuple/dict/list type
            d[k] = literal_eval(v)
        except:  # str type
            pass
    return d


def parse_config(path):
    cfg = ConfigParser()
    cfg.read(path, encoding='utf-8')
    config = {}
    for section in cfg.sections():
        config[section] = SimpleNamespace(**dict_eval(dict(cfg[section])))
    return SimpleNamespace(**config)


def find_all_ext(root, ext: str or list):
    paths = []
    if type(ext) == str:
        ext = [ext]
    for dir_name, _, _ in os.walk(root):
        for item in ext:
            paths += glob.glob(os.path.join(dir_name, '*.' + item))
    return paths


def group_by_key(names: list, key=None):
    names_sorted = sorted(names, key=key)
    names_grouped = itertools.groupby(names_sorted, key=key)
    groups = [list(grouper) for _, grouper in names_grouped]
    return groups


def lists_to_one(lists: list):
    return list(itertools.chain(*lists))


def makedir(path):
    dirname = os.path.dirname(path)
    if len(dirname) != 0:
        if os.path.exists(dirname) is False:
            os.makedirs(dirname)
    return path


def join_and_make(*args: list):
    return makedir(os.path.join(*args))


def basebasename(path: str):
    return os.path.basename(os.path.dirname(path))


def concate_path(path: str, root: str = None, connector: str = '-'):
    if root is None:
        rel_path = path
    else:
        root_path = os.path.abspath(root)
        file_path = os.path.abspath(path)
        rel_path = file_path.replace(root_path, '')
    return connector.join(rel_path.strip(os.path.sep).split(os.path.sep))


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(path: str, data: dict, update=False, convert_to_str: bool = False):
    if convert_to_str is True:
        data_w = dict()
        for k, v in data.items():
            data_w[k] = str(v)
    else:
        data_w = data
    if update is True:
        info: dict = read_json(path)
        data_w = info.update(data_w)
    with open(makedir(path), 'w') as f:
        f.write(json.dumps(data_w, indent=1))


def read_file(path) -> str:
    with open(path) as f:
        return f.read().strip()


def write_file(path: str, text: Union[str, List[str]]):
    with open(makedir(path), 'w') as f:
        if isinstance(text, str):
            f.write(text)
        else:
            for line in text:
                f.write(line + '\n')

