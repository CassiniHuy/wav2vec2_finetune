import os
from argparse import ArgumentParser
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from wav2vec2.train_utils import get_trainer, preprocess
from wav2vec2.train_config import get_training_arguments
from utils import datasets, tools

parser = ArgumentParser()
parser.add_argument('--model_id',
                    type=str,
                    default='facebook/wav2vec2-large-960h-lv60-self')
parser.add_argument('--wav_json', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='tedlium', choices=['tedlium'])
parser.add_argument('--save_path', type=str, default='./wav2vec2_models')
parser.add_argument('--num_train_epochs', type=int, default=None, help='Number of training epochs.')
parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)
parser.add_argument('--per_device_train_batch_size', type=int, default=None)
parser.add_argument('--local_files_only', action='store_true', default=False)
args = parser.parse_args()
print(args)

# * Load training arguments
output_dir = os.path.join(args.save_path, args.model_id.replace(os.path.sep, '-'))
training_args = get_training_arguments(
    output_dir=output_dir)
if args.num_train_epochs is not None:
    training_args.num_train_epochs = args.num_train_epochs
if args.per_device_train_batch_size is not None:
    training_args.per_device_train_batch_size = args.per_device_train_batch_size

# * Load model
print(f'Load pretrained model from: {args.model_id}')
model = Wav2Vec2ForCTC.from_pretrained(args.model_id,
                                       local_files_only=args.local_files_only)
processor = Wav2Vec2Processor.from_pretrained(
    args.model_id, local_files_only=args.local_files_only)

# * Load training/test data
if args.wav_json is not None:
    path2text = tools.read_json(args.wav_json)
    train_wavs, test_wavs = datasets.split_wavs(list(path2text.items()), test_ratio=0.3)
    train_dataset = datasets.create_dataset(train_wavs)
    test_dataset = datasets.create_dataset(test_wavs)
elif args.dataset_name is not None:
    splits = datasets.load_datasets(args.dataset_name)
    train_dataset, test_dataset = splits['train'], splits['validation']
else:
    raise RuntimeError('wav_json and dataset_name cannot both be None')

print(f'{len(train_dataset)} audios in train split ({datasets.compute_dataset_length(train_dataset)/3600:.2f} h)')
tools.write_json(
    os.path.join(output_dir, 'train_split.json'), 
    {item['file']: item['text'] for item in train_dataset})
print(f'{len(test_dataset)} audios in test split ({datasets.compute_dataset_length(test_dataset)/3600:.2f} h)')
tools.write_json(
    os.path.join(output_dir, 'test_split.json'), 
    {item['file']: item['text'] for item in test_dataset})
print(f'Dataset training vocabs:', list(datasets.get_vocabs(train_dataset)))

# assert 1 == 0

# * Train
model.freeze_feature_encoder()
model.train()
trainer = get_trainer(
    model, processor, training_args, 
    preprocess(processor, train_dataset), 
    preprocess(processor, test_dataset))
try:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
except KeyboardInterrupt:
    print('Training early stopped by KeyboardInterrupt.')

# * Save
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

