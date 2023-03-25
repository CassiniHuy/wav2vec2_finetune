# Installation
1. PyTorch installation: https://pytorch.org/
2. Install transformers: https://huggingface.co/docs/transformers/installation

e.g., installation by conda
```
>> conda create -n wav2vec2 python=3.8
>> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
>> conda install -c conda-forge transformers
```



# Finetune
- Prepare your data, 2 options:
  - Dataset id from huggingface
  - A json file containing the audio file path and text. e.g.:
  ```json
  {
    "/home/7303-408-0000.flac": "ECHO OPEN THE DOOR",
    "/home/7303-408-0001.flac": "ECHO OPEN THE DOOR",
    "/home/7303-408-0002.flac": "ECHO OPEN THE DOOR",
  }
  ```
- Finetune. 

e.g.,
```
python wav2vec2_finetune.py --model_id facebook/wav2vec2-large-960h-lv60-self --dataset_name LIUM/tedlium --ckpt_path wav2vec2_models/ --model_save_path large_models/epoch-4 --num_train_epochs 4 --batch_size 10
```

# Inference
- Prepare your data, 3 options:
  - Dataset id from huggingface. It will take the test split.
  - Specify the json file like the finetuning phase.
  - Specify the wav folder. It will automatically find all files (wav, flac). This option will not yield WER as not text info provided.

```
python wav2vec2_infer.py --wav_folder your/audio/files/folder --model_id facebook/wav2vec2-large-960h-lv60-self --with_lm_id 'patrickvonplaten/wav2vec2-base-100h-with-lm'
```

# Torchserve

- Prepare your model mar file, e.g., *modela.mar*:
  
  without language model:
  ```
  torch-model-archiver --model-name modela --version 1.0 --serialized-file modela/pytorch_model.bin --handler handler_v1.py --extra-files "modela/config.json,modela/preprocessor_config.json,modela/special_tokens_map.json,modela/tokenizer_config.json,modela/vocab.json"
  ```
  with language model:
  ```
  torch-model-archiver --model-name modela --version 1.0 --serialized-file modela/pytorch_model.bin --handler handler_v1.py --extra-files "modela/config.json,modela/preprocessor_config.json,modela/special_tokens_map.json,modela/tokenizer_config.json,modela/alphabet.json,modela/vocab.json,modela/language_model/attrs.json,modela/language_model/lm.binary,modela/language_model/unigrams.txt"
  ```
- Move mar file into *model_store* folder
  ```
  mkdir model_store
  mv modela.mar model_store
  ```
- Config: edit *config.properties*
- Start torchserve service
  ```
  torchserve --start --model-store model_store --models modela=modela.mar --ncs
  ```
- Visit the torchserve(*handler_v1.py*):
  ```
  curl http://127.0.0.1:8083/predictions/modela -X POST -T test_audio.wav
  ```
- Stop torchserve:
  ```
  torchserve --stop
  ```
