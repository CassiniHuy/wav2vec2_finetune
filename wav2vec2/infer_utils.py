import torch
from tqdm import tqdm
from typing import List, Tuple, Union
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from wav2vec2.lm import reset_processor_decoder
from utils.datasets import Dataset


# * Get pretrained
def from_pretrained(model_id: str, with_lm_id: str = None
    ) -> Tuple[Wav2Vec2ForCTC, Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]]:
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_id,  local_files_only=True)
        processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=True)
    except:
        model = Wav2Vec2ForCTC.from_pretrained(model_id,  local_files_only=False)
        processor = Wav2Vec2Processor.from_pretrained(model_id, local_files_only=False)
    if with_lm_id is None:
        return model, processor
    else:
        try:
            processor_lm = Wav2Vec2ProcessorWithLM.from_pretrained(with_lm_id, local_files_only=True)
        except:
            processor_lm = Wav2Vec2ProcessorWithLM.from_pretrained(with_lm_id, local_files_only=False)
        if with_lm_id != model_id:
            processor_lm = reset_processor_decoder(processor, processor_lm)
        return model, processor_lm


def obtain_transcription(
    dataset: Dataset,
    model: Wav2Vec2ForCTC, 
    processor: Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM],
    batch_size: int = 10) -> List[str]:
    device = next(model.parameters()).device
    sample_rate = processor.feature_extractor.sampling_rate
    nparrays = [row['audio']['array'] for row in dataset]
    results = list()
    for i in tqdm(range(len(dataset) // batch_size + 1)):
        batch = nparrays[i * batch_size: (i + 1) * batch_size]
        if len(batch) == 0:
            continue
        with torch.no_grad():
            model_inputs = processor(batch, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            if processor.feature_extractor.return_attention_mask is True:
                logits = model(model_inputs.input_values.to(device), attention_mask=model_inputs.attention_mask.to(device)).logits
            else:
                logits = model(model_inputs.input_values.to(device)).logits
            if isinstance(processor, Wav2Vec2ProcessorWithLM):
                pred_text = processor.batch_decode(logits.cpu().numpy()).text
            else:
                pred_ids = torch.argmax(logits, dim=-1).cpu()
                pred_text = processor.batch_decode(pred_ids)
            results += pred_text
    return results
