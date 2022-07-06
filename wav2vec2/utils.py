import torch, torchaudio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from transformers import Wav2Vec2Processor, Trainer, Wav2Vec2ForCTC, TrainingArguments
from datasets import load_metric

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    

# * Compute metrics
wer_metric = load_metric("wer")
def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# * Get trainer
def get_trainer(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor, 
    training_args: TrainingArguments, 
    train_dataset: list, 
    test_dataset: list) -> Trainer:
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, processor),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
    )
    return trainer


# * Get pretrained
def from_pretrained(model_id: str) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_id,  local_files_only=True)
        processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=True)
    except OSError:
        model = Wav2Vec2ForCTC.from_pretrained(model_id,  local_files_only=False)
        processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=False)
    return model, processor


def preprocess(
    processor: Wav2Vec2Processor,
    dataset: List[Dict],
    audio_column: str = 'audio',
    transcription_column: str = 'text'
    ) -> List[Dict]:
    inputs = list()
    alphabets = ''.join(filter(lambda x: x.isalpha(), list(processor.tokenizer.decoder.values())))
    for item in dataset:
        row = dict()
        array, text = item[audio_column]['array'], item[transcription_column]
        text = text.upper() if alphabets.isupper() else text.lower()
        row["input_values"] = processor(
            array, sampling_rate=processor.feature_extractor.sampling_rate).input_values[0]
        with processor.as_target_processor():
            row["labels"] = processor(text.strip()).input_ids
        inputs.append(row)
    return inputs


def predict(
    model: Wav2Vec2ForCTC, 
    processor: Wav2Vec2Processor, 
    wav: str or np.array,
    cuda: bool = True):
    if type(wav) == str:
        torchaudio.set_audio_backend('soundfile')
        wave, sr_ori = torchaudio.load(wav)
        if sr_ori != processor.feature_extractor.sampling_rate:
            wave = torchaudio.functional.resample(
                wave, sr_ori, 
                processor.feature_extractor.sampling_rate,
                 lowpass_filter_width=64)
    else:
        wave = torch.tensor(wav, dtype=torch.float).reshape([1, wav.shape[-1]])
    if cuda is True: wave = wave.cuda()
    with torch.no_grad():
        logits = model(wave).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu()
        return processor.batch_decode(pred_ids)[0]

