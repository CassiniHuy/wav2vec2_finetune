import os, time
from tqdm import tqdm
from utils import tools
from argparse import ArgumentParser
from wav2vec2.utils import from_pretrained, predict
from datasets import load_metric
from utils.datasets import load_datasets

parser = ArgumentParser()
parser.add_argument('--wav_folder', type=str, default=None, help='folder containing wav/flac files')
parser.add_argument('--wav_json', type=str, default=None, help='json file with key/value=wav_path/text')
parser.add_argument('--dataset_name', type=str, default=None, choices=['tedlium'])
parser.add_argument('--model_id', type=str, default='facebook/wav2vec2-large-960h-lv60-self') # jonatasgrosman/wav2vec2-large-xlsr-53-english
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()
print(args)

# * load model
print(f'Load pretrained model from: {args.model_id}')
model, processor = from_pretrained(args.model_id)
model.eval()
model.cuda()

# * Loop wavs
def infer_all(wavs: list, labels: list = None, verbose: bool = True):
    preds = list()
    start_time = time.time()
    if verbose is False:
        wavs = tqdm(wavs)
    for i, wav in enumerate(wavs):
        pred = predict(model, processor, wav).lower()
        preds.append(pred)
        if verbose is True:
            if labels is not None:
                text = labels[i]
                print(f'{i}: {pred.lower()} <-> {text.lower()}')
            else:
                print(f'{i}: {pred.lower()}')
    return preds, time.time() - start_time

wer_metric = load_metric("wer")
def compute_wer(preds, labels):
    preds = [item.lower() for item in preds]
    labels = [item.lower() for item in labels]
    return  wer_metric.compute(predictions=preds, references=labels)

if args.wav_folder is not None:
    print(f'...Inference audios in {args.wav_folder}')
    wavs = tools.find_all_ext(args.wav_folder, ['flac', 'wav'])
    preds, elapsed = infer_all(wavs, verbose=args.verbose)
    save_path = os.path.join(args.wav_folder, args.model_id.replace(os.path.sep, '-') + '.json')
    tools.write_json(save_path, dict([(wavs[i], preds[i]) for i in range(len(wavs))]))
    print(f'{args.wav_folder} results saved to: {save_path} ({elapsed:.4f} sec cost; {elapsed / len(wavs)} sec per wav)')

if args.wav_json is not None:
    print(f'...Inference audios from {args.wav_json}')
    path2text = tools.read_json(args.wav_json)
    preds, elapsed = infer_all(list(path2text.keys()), labels=list(path2text.values()), verbose=args.verbose)
    wer = compute_wer(preds, list(path2text.values()))
    save_path = os.path.join(
        os.path.dirname(args.wav_json), 
        args.model_id.replace(os.path.sep, '-') + '-{0}-wer-{1:.4f}.json'.format(args.wav_json.replace('.json', ''), wer))
    tools.write_json(save_path, {kv[0]: (preds[i], kv[1]) for i, kv in enumerate(path2text.items())})
    print(f'{args.wav_json} results saved to: {save_path} ({elapsed:.4f} sec cost; {elapsed / len(preds)} sec per wav)')
    
if args.dataset_name is not None:
    test_split = load_datasets(args.dataset_name, split='test')
    wavs, labels = [item['audio']['array'] for item in test_split], [item['text'] for item in test_split]
    preds, elapsed = infer_all(wavs, labels=labels, verbose=args.verbose)
    wer = compute_wer(preds, labels)
    save_path = args.model_id.strip(os.path.sep).replace(os.path.sep, '-') + f'-{args.dataset_name}-wer-{wer:.4f}.json'
    tools.write_json(save_path, {item['file']: (preds[i], item['text']) for i, item in enumerate(test_split)})
    print(f'{args.dataset_name} results saved to: {save_path} ({elapsed:.4f} sec cost; {elapsed / len(wavs)} sec per wav)')

