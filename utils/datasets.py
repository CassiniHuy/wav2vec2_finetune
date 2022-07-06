import re, librosa
import random
from tqdm import tqdm
from typing import List, Tuple, Dict
from datasets import load_dataset


def split_wavs(wavs: List[Tuple[str, str]] or Dict[str, str],
                  test_ratio: float,
                  random_seed: int = 999
                  ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    random.seed(random_seed)
    if type(wavs) == dict:
        wavs = list(wavs.items())
    random.shuffle(wavs)
    return wavs[int(len(wavs) * test_ratio):], wavs[:int(len(wavs) * test_ratio)]


chars_to_ignore_regex = r'<.*?>|\[.*?\]|\(.*?\)|\{.*?\}|[\,\?\.\!\-\;\:\"]'
def filter_invalid_vocabs(
    wavs: List[Dict[str, str]] or List[Tuple[str, str]], 
    key= None) -> List[Dict[str, str]] or List[Tuple[str, str]]:
    for i, item in enumerate(wavs):
        text = re.sub(chars_to_ignore_regex, '', item[key]).strip()
        wavs[i][key] = re.sub(r'\s+', ' ', text).strip()
    return wavs


def get_vocabs(
    wavs: List[Dict[str, str]] or List[Tuple[str, str]], 
    key= None) -> list:
    vocabs = set()
    for item in wavs:
        vocabs |= set(item[key])
    return list(vocabs)


# * Prepare data
def create_dataset(
    wavs: List[Tuple[str, str]] or Dict[str, str], 
    sampling_rate: int = 16000,
    ) -> List[Dict[str, str]]:
    dataset = list()
    if type(wavs) == list:
        files = tqdm(wavs)
    elif type(wavs) == dict:
        files = tqdm(wavs.items())
    else:
        raise TypeError('Unrecognizable wavs type:', str(type(wavs)))
    for wav, text in files:
        row, audio = dict(), dict()
        array, _ = librosa.load(wav, sr=sampling_rate, mono=True)
        audio['array'] = array
        audio['path'] = wav
        audio['sampling_rate'] = sampling_rate
        row['audio'] = audio
        row['file'] = wav
        row['text'] = text
        dataset.append(row)
    return dataset


def random_select(data: list, limit: int, random_seed: int = 999):
    random.seed(random_seed)
    indices_selected = random.sample(list(range(len(data))), k=limit)
    selected = list()
    for index in indices_selected:
        selected.append(data[index])
    return selected


def load_datasets(
    dataset: str,
    train_limit: int = 7000,
    random_seed: int = 999,
    split: str = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]] or List[Dict]:
    invalid_tedlium_text = 'ignore_time_segment_in_scoring'
    invalid_vocabs = r'[0123456789&\$\+]'
    if dataset == 'tedlium':
        def is_valid(text): # ! Just filter out these data
            if text == invalid_tedlium_text: 
                return False
            if re.search(invalid_vocabs, text):
                return False
            return True
        def sanitize(split_dataset):
            split_dataset = list(filter(lambda x: is_valid(x['text']), split_dataset))
            split_dataset = filter_invalid_vocabs(split_dataset, key='text')
            return split_dataset
        
        if split is None:
            tedlium = load_dataset('LIUM/tedlium', 'release3')
            train_dataset, val_dataset, test_dataset = tedlium['train'], tedlium['validation'], tedlium['test']
            train_dataset = random_select(train_dataset, limit=train_limit, random_seed=random_seed)
            return sanitize(train_dataset), sanitize(val_dataset), sanitize(test_dataset)
        else:
            tedlium_split = load_dataset('LIUM/tedlium', 'release3', split=split)
            if split == 'train':
                tedlium_split = random_select(tedlium_split, limit=train_limit, random_seed=random_seed)
            return sanitize(tedlium_split)
    else:
        raise ValueError(f'Unrecognizable dataset: {dataset}')


def compute_dataset_length(
    dataset: List[Dict]) -> float:
    length = 0.0
    for item in dataset:
        length += item['audio']['array'].shape[-1] / item['audio']['sampling_rate']
    return length # seconds

