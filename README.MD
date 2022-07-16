# Usage

## Finetune
- Prepare your data, 2 options:
  - Dataset id from huggingface
  - A json file containing the audio file path and text. e.g., examples/AWS_Corpus.json
- Finetune. 

e.g.,
```
>> python wav2vec2_finetune.py --model_id facebook/wav2vec2-large-960h-lv60-self --dataset_name LIUM/tedlium --ckpt_path wav2vec2_models/ --model_save_path large_models/epoch-4 --num_train_epochs 4 --batch_size 10
```

# Inference
- Prepare your data, 3 options:
  - Dataset id from huggingface. It will take the test split.
  - Specify the json file like the finetuning phase.
  - Specify the wav folder. It will automatically find all files (wav, flac). This option will not yield WER as not text info provided.

```
>> python wav2vec2_infer.py --wav_folder your/audio/files/folder --model_id facebook/wav2vec2-large-960h-lv60-self --with_lm_id 'patrickvonplaten/wav2vec2-base-100h-with-lm'
```

