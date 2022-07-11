import re, random, warnings
from typing import List, Tuple
from datasets import load_dataset, Dataset, DatasetInfo, Features, Audio, Value
from datasets.dataset_dict import DatasetDict


''' Const definition '''
features = Features({
    'audio': Audio(sampling_rate=16_000, mono=True, decode=True), 
    'text': Value(dtype='string'),
    'file': Value(dtype='string')
    })
dataset_info = DatasetInfo(features=features)
AUDIO_PATH, TEXT = str, str
DATA_LIST = List[Tuple[AUDIO_PATH, TEXT]] or List[Tuple[AUDIO_PATH]]
chars_to_ignore_regex = r'<.*?>|\[.*?\]|\(.*?\)|\{.*?\}|[\,\?\.\!\-\;\:\"]'


''' Utils functions '''
def split_wavs(
    wavs: DATA_LIST,
    test_ratio: float,
    random_seed: int = 999
    ) -> Tuple[DATA_LIST, DATA_LIST]:
    random.seed(random_seed)
    random.shuffle(wavs)
    return wavs[int(len(wavs) * test_ratio):], wavs[:int(len(wavs) * test_ratio)]


def filter_invalid_vocabs(
    dataset: Dataset, 
    pattern: str = chars_to_ignore_regex
    ) -> Dataset:
    def remove_invalid_vocabs(batch):
        batch['text'] = re.sub(pattern, '', batch['text']).strip()
        batch['text'] = re.sub(r'\s+', ' ', batch['text']).strip()
        return batch
    return dataset.map(remove_invalid_vocabs)


def filter_long_audios(dataset: Dataset, max_duration: int = 10) -> Dataset:
    def is_short_audio(batch):
        if batch['audio']['array'].shape[-1] / batch['audio']['sampling_rate'] < max_duration:
            return True
        else:
            return False
    return Dataset.filter(dataset, is_short_audio)


def get_vocabs(
    dataset: Dataset) -> list:
    vocabs = set()
    for item in dataset:
        vocabs |= set(item['text'])
    return list(vocabs)


def compute_dataset_length(
    dataset: Dataset) -> float:
    length = 0.0
    for item in dataset:
        length += item['audio']['array'].shape[-1] / item['audio']['sampling_rate']
    return length # seconds


def subset_dataset(dataset: Dataset, limit: int, random_seed: int = 999) -> Dataset:
    if len(dataset) < limit:
        return dataset
    random.seed(random_seed)
    indices_selected = random.sample(list(range(len(dataset))), k=limit)
    return dataset.select(indices_selected)


''' Load dataset '''
def create_dataset(wavs: DATA_LIST) -> Dataset:
    if isinstance(wavs[0], tuple) is True:
        files = [pair[0] for pair in wavs]
        texts = [pair[1] for pair in wavs]
    elif isinstance(wavs[0], str) is True:
        files = wavs
        texts = [''] * len(wavs)
    else:
        raise TypeError(f'Unrecognizable wavs item type: {str(type(wavs[0]))}. Should be Tuple[str, str] or str.')
    data = {'audio': files, 'file': files, 'text': texts}
    return Dataset.from_dict(data, info=dataset_info)


def load_tedlium(
    small_version: bool = False,
    train_limit: int = 7000,
    random_seed: int = 999,
    split: str = None,
    ) -> DatasetDict or Dataset:
    invalid_tedlium_text = 'ignore_time_segment_in_scoring'
    invalid_vocabs = r'[0123456789&\$\+]'
    def is_valid(text: str) -> bool: # ! Just filter out these data
        if text == invalid_tedlium_text: 
            return False
        if re.search(invalid_vocabs, text):
            return False
        return True
    def sanitize(split_dataset: Dataset) -> Dataset:
        split_dataset = split_dataset.filter(lambda x: is_valid(x['text']))
        split_dataset = filter_invalid_vocabs(split_dataset)
        return split_dataset
    if small_version is False:
        tedlium = load_dataset('LIUM/tedlium', 'release3', split=split)
    else:
        tedlium = load_dataset('chengan/tedlium_small', split=split)
    if split is None:
        tedlium['train'] = subset_dataset(tedlium['train'], limit=train_limit, random_seed=random_seed)
        tedlium['train'], tedlium['validation'], tedlium['test'] = \
             sanitize(tedlium['train']), sanitize(tedlium['validation']), sanitize(tedlium['test'])
    else:
        if split == 'train':
            tedlium = subset_dataset(tedlium, limit=train_limit, random_seed=random_seed)
        tedlium = sanitize(tedlium)
    return tedlium


def load_datasets(
    dataset: str,
    train_limit: int = 7000,
    random_seed: int = 999,
    split: str = None,
    ) -> DatasetDict or Dataset:
    if dataset == 'tedlium':
        return load_tedlium(small_version=False, train_limit=train_limit, random_seed=random_seed, split=split)
    elif dataset == 'tedlium_small':
        return load_tedlium(small_version=True, train_limit=train_limit, random_seed=random_seed, split=split)
    else:
        data_splits = load_dataset(dataset, split=split)
    if split is None:
        data_splits['train'] = subset_dataset(data_splits['train'], limit=train_limit, random_seed=random_seed)
    if split == 'train':
        data_splits = subset_dataset(data_splits, limit=train_limit, random_seed=random_seed)
    warnings.warn(message=f'Dataset "{dataset}" loaded without filtering out invalid vocabs.', category=UserWarning)
    return data_splits
