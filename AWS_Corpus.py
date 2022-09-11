import os
from argparse import ArgumentParser
from utils import tools

parser = ArgumentParser()
parser.add_argument('folder', type=str)
args = parser.parse_args()

paths = tools.find_all_ext(args.folder, 'flac')
print(f'{len(paths)} audio files found (.flac)')

txts = tools.find_all_ext(args.folder, 'txt')
path2txt = dict()
for txt in txts:
    with open(txt) as txtf:
        for line in txtf:
            wav, text = line.strip().split('\t')
            path = os.path.join(args.folder, *wav.split('-')[:2], wav + '.flac')
            path2txt[path] = text

print(f'{len(path2txt)} audio files found in txt files.')

tools.write_json('AWS_Corpus.json', path2txt)

