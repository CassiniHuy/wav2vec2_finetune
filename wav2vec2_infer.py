import os, re
from utils import tools
from argparse import ArgumentParser
from wav2vec2.infer_utils import from_pretrained, obtain_transcription
from jiwer import wer as wer_metric
from utils.datasets import load_datasets, create_dataset, filter_long_audios, chars_to_ignore_regex

parser = ArgumentParser()
parser.add_argument('--wav_folder', type=str, default=None, help='folder containing wav/flac files')
parser.add_argument('--wav_json', type=str, default=None, help='json file with key/value=wav_path/text')
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--model_id', type=str, default='facebook/wav2vec2-large-960h-lv60-self') # jonatasgrosman/wav2vec2-large-xlsr-53-english
parser.add_argument('--with_lm_id', type=str, default=None) # LM: patrickvonplaten/wav2vec2-base-100h-with-lm
parser.add_argument('--batch', type=int, default=20)
parser.add_argument('--max_duration', type=int, default=20, help='filter out long audios')
parser.add_argument('--data_dir', type=str, default=None)
args = parser.parse_args()
print(args)

def alias(id: str) -> id:
    if id == 'large-model':
        return 'facebook/wav2vec2-large-960h-lv60-self'
    elif id == 'xlsr-model':
        return 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    elif id == 'base-lm':
        return 'patrickvonplaten/wav2vec2-base-100h-with-lm'
    elif id == 'xlsr-lm':
        return 'jonatasgrosman/wav2vec2-large-xlsr-53-english'
    else:
        return id
args.model_id = alias(args.model_id)
args.with_lm_id = alias(args.with_lm_id)

# * load model
print(f'Load pretrained model from: {args.model_id}')
if args.with_lm_id is not None:
    print(f'Inference with language model: {args.with_lm_id}')
else:
    print(f'Inference without language model.')
model, processor = from_pretrained(model_id=args.model_id, with_lm_id=args.with_lm_id)
model.eval()
model.cuda()

# Procedures
def compute_wer(preds, labels, filter_invalid_chars: bool = True):
    preds = [item.lower() for item in preds]
    labels = [item.lower() for item in labels]
    if filter_invalid_chars:
        func = lambda x: re.sub(chars_to_ignore_regex, '', re.sub(r'\s+', ' ', x)).strip()
        preds = [func(x) for x in preds]
        labels = [func(x) for x in labels]
    return wer_metric(truth=labels, hypothesis=preds)

def save_results(dataset, dataset_name, results, save_dir=''):
    wavs = [row['file'] for row in dataset]
    texts = [row['text'] for row in dataset]
    wer = compute_wer(results, texts) if len(''.join(texts)) != 0 else 999
    model_str = args.model_id.strip(os.path.sep).replace(os.path.sep, '-')
    lm_str = args.with_lm_id.strip(os.path.sep).replace(os.path.sep, '-') if args.with_lm_id is not None else None
    save_path = os.path.join(
        save_dir, f'{dataset_name}_wer-{wer:.4f}_{model_str}_lm-{str(lm_str)}.txt')
    tools.write_file(save_path, [';'.join([wavs[i], texts[i], results[i]]) for i in range(len(wavs))])
    print(f'Results saved to: {save_path}.')


# Inference
if args.wav_folder is not None:
    print(f'...Inference audios in {args.wav_folder}')
    wavs = tools.find_all_ext(args.wav_folder, ['flac', 'wav'])
    dataset = create_dataset(wavs)
    dataset = filter_long_audios(dataset, args.max_duration)
    print('Get audios number:', len(dataset))
    results = obtain_transcription(dataset, model, processor, args.batch)
    save_results(dataset, os.path.basename(args.wav_folder), results, os.path.dirname(args.wav_folder))

if args.wav_json is not None:
    print(f'...Inference audios from {args.wav_json}')
    path2text = tools.read_json(args.wav_json)
    dataset = create_dataset(list(path2text.items()))
    dataset = filter_long_audios(dataset, args.max_duration)
    print('Get audios number:', len(dataset))
    results = obtain_transcription(dataset, model, processor, args.batch)
    save_results(dataset, args.wav_json[:-5], results)
    
if args.dataset_name is not None:
    test_split = load_datasets(args.dataset_name, split='test', data_dir=args.data_dir)
    test_split = filter_long_audios(test_split, args.max_duration)
    print('Get audios number:', len(test_split))
    results = obtain_transcription(test_split, model, processor, args.batch)
    save_results(test_split, args.dataset_name, results)
