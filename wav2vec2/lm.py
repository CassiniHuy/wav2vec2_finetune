from typing import Dict, List, Tuple
import warnings
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer
from pyctcdecode.alphabet import UNK_TOKEN
from pyctcdecode.decoder import BeamSearchDecoderCTC

LM_PAD_TOKEN = ''
LM_DELIMITER = ' '
LM_BOS_TOKEN = '<s>'
LM_EOS_TOKEN = '</s>'


def reset_idx2vocab(
    lm_idx2vocab: dict, tokenizer: Wav2Vec2CTCTokenizer, 
    lm_bos_token: type = LM_BOS_TOKEN, lm_eos_token: type = LM_EOS_TOKEN
    ) -> Tuple[Dict[int, str], List[str]]:
    new_lm_idx2vocab = dict()
    for lm_vocab in lm_idx2vocab.values():
        # Query the encoder to obtain the vocab's new index
        if lm_vocab == UNK_TOKEN:
            true_idx = tokenizer.unk_token_id
        elif lm_vocab == LM_PAD_TOKEN:
            true_idx = tokenizer.pad_token_id
        elif lm_vocab == LM_DELIMITER:
            true_idx = tokenizer.word_delimiter_token_id
        elif lm_vocab == lm_bos_token:
            true_idx = tokenizer.bos_token_id
        elif lm_vocab == lm_eos_token:
            true_idx = tokenizer.eos_token_id
        else:
            query = lm_vocab
            try:
                true_idx = tokenizer.encoder[query.lower()]
            except KeyError:
                try:
                    true_idx = tokenizer.encoder[query.upper()]
                except KeyError:
                    warnings.warn(message=f'Cannot find the corresponding index of vocab "{lm_vocab}"', category=RuntimeWarning)
                    continue
        new_lm_idx2vocab[true_idx] = lm_vocab
    lm_vocabs_uncovered = set(lm_idx2vocab.values()) - set(new_lm_idx2vocab.values())
    ori_vocabs_uncovered = set(tokenizer.encoder.keys()) \
        - set([tokenizer.decoder[idx] for idx in new_lm_idx2vocab.keys()])
    # Map uncovered indices to the unknown token
    if len(lm_vocabs_uncovered) != 0:
        warnings.warn(message=f"Vocabs uncovered in LM's _idx2vocab: {str(lm_vocabs_uncovered)}.", category=RuntimeWarning)
    if len(ori_vocabs_uncovered) != 0:
        for ori_vocab in ori_vocabs_uncovered:
            new_lm_idx2vocab[tokenizer.encoder[ori_vocab]] = '⁇'
        warnings.warn(message=f"Vocabs uncovered in tokenizer's decoder: {str(ori_vocabs_uncovered)}." + \
            "Corresponding indices are mapped to the unknown token.", category=RuntimeWarning)
    new_lm_labels = list(new_lm_idx2vocab.values())
    return new_lm_idx2vocab, new_lm_labels


def reset_tokenizer_vocabs(tokenizer: Wav2Vec2CTCTokenizer, lm_tokenizer: Wav2Vec2CTCTokenizer) -> Wav2Vec2CTCTokenizer:
    # Set special tokens
    tokenizer.word_delimiter_token = lm_tokenizer.word_delimiter_token
    tokenizer.bos_token = lm_tokenizer.bos_token
    tokenizer.eos_token = lm_tokenizer.eos_token
    tokenizer.pad_token = lm_tokenizer.pad_token
    tokenizer.unk_token = lm_tokenizer.unk_token
    # Get new encoder and decoder
    missing_vocabs = set(tokenizer.get_vocab()) - set(lm_tokenizer.get_vocab())
    for vocab in missing_vocabs:
        vocab_case = vocab.upper() if vocab.islower() else vocab.lower()
        if vocab_case in lm_tokenizer.encoder.keys(): # if a-z or A-Z
            tokenizer.decoder[tokenizer.encoder[vocab]] = vocab_case
            tokenizer.encoder[vocab_case] = tokenizer.encoder[vocab]
            tokenizer.encoder.pop(vocab)
        else:
            warnings.warn(message=f'Found missing alphabet token: "{vocab}".' + \
                'This vocab will be removed from encoder and mapped to the unknown token in decoder.', category=RuntimeWarning)
            tokenizer.decoder[tokenizer.encoder[vocab]] = tokenizer.unk_token
            tokenizer.encoder.pop(vocab)
    return tokenizer


def combine_lm(
    processor_nolm: Wav2Vec2Processor,
    processor_withlm: Wav2Vec2ProcessorWithLM
    ) -> Wav2Vec2ProcessorWithLM:
    # Reset the decoder mapping
    decoder: BeamSearchDecoderCTC = processor_withlm.decoder
    idx2vocab, labels = reset_idx2vocab(
        processor_withlm.decoder._idx2vocab, processor_nolm.tokenizer, 
        lm_bos_token=processor_nolm.tokenizer.bos_token,
        lm_eos_token=processor_nolm.tokenizer.eos_token)
    decoder._idx2vocab = idx2vocab
    decoder._alphabet._labels = labels
    # Reset the tokenizer mapping
    tokenizer: Wav2Vec2CTCTokenizer = reset_tokenizer_vocabs(processor_nolm.tokenizer, processor_withlm.tokenizer)
    return Wav2Vec2ProcessorWithLM(processor_nolm.feature_extractor, tokenizer, decoder)
