import io, logging, os, shutil
import torch, librosa
from ts.torch_handler.base_handler import BaseHandler
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM

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
SAMPLING_RATE = 16000

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

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # self.device = torch.device('cpu')

        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir,  local_files_only=True)
        logger.info(f'Model loaded from {model_dir} loaded.')
        if os.path.exists(os.path.join(model_dir, 'lm.binary')):
            os.mkdir(os.path.join(model_dir, 'language_model'))
            shutil.move(os.path.join(model_dir, 'lm.binary'), os.path.join(model_dir, 'language_model', 'lm.binary'))
            shutil.move(os.path.join(model_dir, 'attrs.json'), os.path.join(model_dir, 'language_model', 'attrs.json'))
            shutil.move(os.path.join(model_dir, 'unigrams.txt'), os.path.join(model_dir, 'language_model', 'unigrams.txt'))
            self.processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_dir, local_files_only=True)
            logger.info(f'Processor with LM loaded from {model_dir} loaded.')
            self.with_lm = True
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_dir, local_files_only=True)
            logger.info(f'Processor without LM loaded from {model_dir} loaded.')
            self.with_lm = False
        self.model.eval()
        self.model.to(self.device)
        self.initialized = True

    def preprocess(self, requests):
        waves = list()
        for idx, data in enumerate(requests):
            audio = io.BytesIO(data['body'])
            wave, _ = librosa.load(audio, sr=SAMPLING_RATE, duration=MAX_DURATION, mono=True)
            waves.append(wave)
        inputs = self.processor(waves, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        return inputs.input_values, inputs.attention_mask

    def inference(self, inputs):
        with torch.no_grad():
            input_values, attention_mask = inputs
            if self.processor.feature_extractor.return_attention_mask is True:
                logits = self.model(input_values.to(self.device), attention_mask=attention_mask.to(self.device)).logits
            else:
                logits = self.model(input_values.to(self.device)).logits
            if self.with_lm:
                return logits
            else:
                return torch.argmax(logits, dim=-1)

    def postprocess(self, outputs):
        if self.with_lm:
            pred_text = self.processor.batch_decode(outputs.cpu().numpy()).text
        else:
            pred_text = self.processor.batch_decode(outputs)
        return pred_text