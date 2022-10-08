from typing import Dict, List, Tuple, Union
from ts.torch_handler.base_handler import BaseHandler
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode.alphabet import UNK_TOKEN
from torchmetrics.functional import signal_noise_ratio
from collections import OrderedDict
from datasets import load_dataset
import logging, os, shutil, json, re
import torch, librosa
import numpy as np

formatter = logging.Formatter(
    r"%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
    datefmt=r"%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler(__name__ + '.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

MAX_DURATION = 5
BATCH_SIZE = 20
SAMPLING_RATE = 16000

LM_PAD_TOKEN = ''
LM_DELIMITER = ' '
LM_BOS_TOKEN = '<s>'
LM_EOS_TOKEN = '</s>'

COMMANDS = [
    "Text my client hello",
    "Show me the payment code",
    "Your lock is broken",
    "Remind me to call Bill",
    "Turn off all alarms",
    "Turn on automatic renewal",
    "Clear the voice mail",
    "Screen capture",
    "Your flight has been canceled",
    "Navigate to Golden Gate Bridge",
]

NOSCORE = 0
WHITEBOX = 1
BLACKBOX = 2

SUCCESS = 0
RUNTIME_ERROR = 1
INVALID_INPUT = 2
REQUEST_ERROR = 3

def reset_lm_idx2vocab(encoder: dict, lm: Wav2Vec2ProcessorWithLM) -> Wav2Vec2ProcessorWithLM:
    for vocab, idx in encoder.items():
        if vocab == "<pad>":
            lm.decoder._idx2vocab[idx] = LM_PAD_TOKEN
        elif vocab == "<s>":
            lm.decoder._idx2vocab[idx] = LM_BOS_TOKEN
        elif vocab == "</s>":
            lm.decoder._idx2vocab[idx] = LM_EOS_TOKEN
        elif vocab == "<unk>":
            lm.decoder._idx2vocab[idx] = UNK_TOKEN
        elif vocab == "|":
            lm.decoder._idx2vocab[idx] = LM_DELIMITER
        else:
            lm.decoder._idx2vocab[idx] = vocab
    return lm
    

def if_succeed(prediction: str, command: str) -> bool:
    prediction = prediction.replace(UNK_TOKEN, ' ').replace(LM_BOS_TOKEN, ' ').replace(LM_EOS_TOKEN, ' ')
    prediction = prediction.replace('-', ' ').replace("'", ' ')
    prediction = prediction.strip().lower()
    if command in prediction:
        beg_idx = prediction.index(command)
        end_idx = beg_idx + len(command)
        if beg_idx > 0:
            if prediction[beg_idx - 1] != ' ':
                return False
        if end_idx < len(prediction):
            if prediction[end_idx] != ' ':
                return False
        return True
    else:
        return False

    
def parse_filename(path: str) -> Union[None, Tuple[Union[int, None], Union[int, None]]]:
    filename = os.path.splitext(os.path.basename(path))[0].lower()
    match_ojb = re.match(r'carrier([0-9]{2})_c([0-9]{2})', filename)
    if match_ojb is not None:
        source_id = int(match_ojb.group(1)) - 1
        command_id = int(match_ojb.group(2)) - 1
        if source_id < 0 or source_id >= 50:
            source_id = None
        if command_id < 0 or command_id >= 10:
            command_id = None
        return source_id, command_id
    else:
        return None

    
def compute_snr(wave: np.ndarray, source: np.ndarray) -> float:
    source_tensor = torch.tensor(source)
    wave_tensor = torch.tensor(wave[:source.shape[0]])
    wave_tensor = torch.nn.functional.pad(wave_tensor, \
        [0, source_tensor.shape[0] - wave_tensor.shape[0]])
    snr = signal_noise_ratio(wave_tensor, source_tensor).item()
    return snr

    
def whitebox_score(snr: float) -> float:
    return max(0, 1 + (snr - 20))

def blackbox_score(snr: float) -> float:
    return max(0, 3 + 3 * (snr - 12))

def compute_score(result: dict, snr: float, command: str, mode: int) -> float:
    if if_succeed(result, command) is True:
        if mode == WHITEBOX:
            score = whitebox_score(snr)
        elif mode == BLACKBOX:
            score = blackbox_score(snr)
        else:
            score = 0
    else:
        score = 0
    return score

class Wav2vec2Handler(BaseHandler):
    """
    Huggingface Wav2vec2 handler class.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        
        # self.device = torch.device(
        #     "cuda:" + str(properties.get("gpu_id"))
        #     if torch.cuda.is_available() and properties.get("gpu_id") is not None
        #     else "cpu"
        # )
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0')

        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir,  local_files_only=True)
        self.model.eval()
        self.model.to(self.device)
        logger.info(f'Model loaded from {model_dir} loaded.')
        if os.path.exists(os.path.join(model_dir, 'lm.binary')):
            os.mkdir(os.path.join(model_dir, 'language_model'))
            shutil.move(os.path.join(model_dir, 'lm.binary'), os.path.join(model_dir, 'language_model', 'lm.binary'))
            shutil.move(os.path.join(model_dir, 'attrs.json'), os.path.join(model_dir, 'language_model', 'attrs.json'))
            shutil.move(os.path.join(model_dir, 'unigrams.txt'), os.path.join(model_dir, 'language_model', 'unigrams.txt'))
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_dir, local_files_only=True)
            with open(os.path.join(model_dir, 'vocab.json')) as f:
                encoder = json.load(f)
                self.processor = reset_lm_idx2vocab(encoder, self.processor)
            logger.info(f'Processor with LM loaded from {model_dir} loaded.')
            self.with_lm = True
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_dir, local_files_only=True)
            logger.info(f'Processor without LM loaded from {model_dir} loaded.')
            self.with_lm = False
        # Load commands and sources
        wavs = load_dataset(r'yongjian/music-clips-50')['train']
        self.sources = [wavs['audio'][wavs['file'].index(name)]['array'] for name in sorted(wavs['file'])]
        self.commands = [cmd.lower().strip() for cmd in COMMANDS]
        logger.info(f'Source audios and commands loaded.')
        self.initialized = True

    def _get_batches(self, waves: List[np.ndarray]) -> list:
        batches = list()
        for i in range(len(waves) // BATCH_SIZE + 1):
            batch = waves[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            if len(batch) == 0:
                continue
            batch_i = self.processor(batch, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
            batches.append(batch_i)
        return batches

    def preprocess(self, requests) -> Dict[str, Dict[str, Union[str, torch.Tensor, bool]]]:
        '''requests:
        {
            "paths": ["path1", "path2", ...],
            "mode": 0 or 1 or 2
        }
        '''
        # logger.debug(str(requests))
        dataset, waves = OrderedDict(), list()
        for req_id, item in enumerate(requests):
            # * Get paths and mode
            try:
                data = item['body']
                audio_paths = data['paths']
            except KeyError:
                continue
            try:
                mode = int(data['mode'])
                assert mode in [NOSCORE, WHITEBOX, BLACKBOX]
            except:
                mode = NOSCORE
            # * Preprocess paths
            if isinstance(audio_paths, list) is False:   # * not a list
                audio_paths = [audio_paths]
            else:
                pass
            for path in audio_paths:
                path = str(path)
                dataset[path] = dict(mode=mode, req_id=req_id)
                if os.path.exists(path) is True:
                    # * Preprocess mode
                    try:
                        dataset[path]['array'], _ = librosa.load(path, sr=SAMPLING_RATE, duration=MAX_DURATION, mono=True)
                        dataset[path]['message'] = ''
                        dataset[path]['status'] = SUCCESS
                        waves.append(dataset[path]['array'])
                    except:
                        dataset[path]['array'] = None
                        dataset[path]['message'] = 'Audio loaded error. Please upload audio files with wav format.'
                        dataset[path]['status'] = INVALID_INPUT
                else:
                    dataset[path]['array'] = None
                    dataset[path]['message'] = 'File not found.'
                    dataset[path]['status'] = REQUEST_ERROR
        # Prepare batches
        batches = self._get_batches(waves)
        return dataset, batches
    
    def inference(self, inputs) -> OrderedDict:
        '''inputs: dataset, batches
        dataset:
        {
            "path1": {"array": np.ndarray, "mode": 0 or 1 or 2, "message": str},
            "path2": {"array": np.ndarray, "mode": 0 or 1 or 2, "message": str},
            ...
        }
        '''
        results = list()
        with torch.no_grad():
            dataset, batches = inputs
            for batch in batches:
                try:
                    if self.processor.feature_extractor.return_attention_mask is True:
                        logits = self.model(
                            batch.input_values.to(self.device), 
                            attention_mask=batch.attention_mask.to(self.device)).logits
                    else:
                        logits = self.model(batch.input_values.to(self.device)).logits
                    if self.with_lm:
                        pred_text = self.processor.batch_decode(logits.cpu().numpy()).text
                    else:
                        pred_text = self.processor.batch_decode(torch.argmax(logits, dim=-1))
                    results += pred_text
                except RuntimeError as e:
                    results += [e,] * batch.input_values.shape[0]
        for key, value in dataset.items():
            if value['status'] == SUCCESS:
                result = results[0]
                if isinstance(result, RuntimeError):
                    dataset[key]['result'] = ''
                    dataset[key]['message'] = str(result)
                    dataset[key]['status'] = RUNTIME_ERROR
                else:
                    dataset[key]['result'] = result
                    dataset[key]['status'] = SUCCESS
                results = results[1:]
            else:
                dataset[key]['result'] = ''
        return dataset
        
    def _evaluate_ae(self, path, pairs: dict) -> dict:
        # Parse filename
        parse_result = parse_filename(path)
        if parse_result is None:
            snr, score, status = np.nan, 0.0, INVALID_INPUT
            pairs['message'] = 'Invalid filename. Please use the valid filename like carrier01_c01.wav.'
        else:
            source_id, command_id = parse_result
            if source_id is None:
                snr, score, status = np.nan, 0.0, INVALID_INPUT
                pairs['message'] = 'Invalid filename. Carrier id should > 0 and <= 50.'
            else:
                # Compute snr
                snr = compute_snr(pairs['array'], self.sources[source_id])
                if command_id is None:
                    score, status = 0.0, INVALID_INPUT
                    pairs['message'] = 'Invalid filename. Command id should > 0 and <= 10.'
                else:
                    score = compute_score(pairs['result'], snr, self.commands[command_id], pairs['mode'])
                    status = SUCCESS
        pairs['snr'] = snr
        pairs['score'] = score
        pairs['status'] = status
        return pairs
    
    def postprocess(self, outputs: OrderedDict) -> str:
        '''return
        {
            "path1":{result: str, snr: float, score: float, message: str, status: int}
        }
        '''
        if len(outputs) == 0:
            return ['']
        else:
            results = [dict(),] * (list(outputs.values())[-1]['req_id'] + 1)
        for key, pairs in outputs.items():
            req_id = pairs['req_id']
            results[req_id][key] = dict()
            results[req_id][key]['result'] = pairs['result']
            if pairs['mode'] != NOSCORE:
                if pairs['status'] == SUCCESS:
                    pairs = self._evaluate_ae(key, pairs)
                    results[req_id][key]['snr'] = pairs['snr']
                    results[req_id][key]['score'] = pairs['score']
                else:
                    results[req_id][key]['snr'] = np.nan
                    results[req_id][key]['score'] = 0.0
            else:
                pass
            results[req_id][key]['message'] = pairs['message']
            results[req_id][key]['status'] = pairs['status']
        results = [json.dumps(result) for result in results]
        # logger.debug('outputs: ' + str(results))
        return results
