from model.esm_mamba_hf import EsmMambaConfig, EsmTokenizer, EsmMambaModel, EsmMambaForMaskedLM
from transformers import EsmConfig, EsmModel, EsmForMaskedLM
import copy
import torch
import random
import numpy as np

def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

	if dtype is None:
		dtype = sequences[0].dtype

	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)

	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq

	return array

def trainable_parameters(model):
    print(f"Trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")

config = EsmMambaConfig.from_pretrained('config/model/esm_mamba_8M.yaml')
model = EsmMambaForMaskedLM(config)
# Print the number of trainable parameters
trainable_parameters(model)


# esm_config = EsmConfig.from_pretrained('/cto_labs/AIDD/WEIGHTS/Protein/esm2_t12_35M_UR50D')
# esm_model = EsmForMaskedLM(esm_config)
# trainable_parameters(esm_model)
# breakpoint()

# load tokenizer
tokenizer = EsmTokenizer.from_pretrained('/cto_labs/AIDD/WEIGHTS/Protein/esm2_t6_8M_UR50D')
seq_len = 1024

"""tokenizer
(Pdb) self.tokenizer._id_to_token
{0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O', 29: '.', 30: '-', 31: '<null_1>', 32: '<mask>'}
(Pdb) self.tokenizer._token_to_id
{'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32}
"""

def encode_sequence(seq, tokenizer:EsmTokenizer, seq_len:int):
    # Encode sequence
    ids = tokenizer.encode(seq, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # Truncate sequence
    if len(ids) > seq_len:
        ids = ids[:seq_len]
        tokens = tokens[:seq_len]
    
    masked_tokens, labels = _apply_bert_mask(tokens, tokenizer)
    masked_seq = " ".join(masked_tokens)
    return masked_seq, labels

def _apply_bert_mask(tokens, tokenizer:EsmTokenizer, mask_ratio:float=0.15):
    masked_tokens = copy.copy(tokens)
    labels = torch.full((len(tokens)+2,), -100, dtype=torch.long)
    for i in range(len(tokens)):
        token = tokens[i]
        prob = random.random()
        
        if prob < mask_ratio:
            prob /= mask_ratio
            labels[i+1] = tokenizer.convert_tokens_to_ids(token)
            
            if prob < 0.8:
                token = tokenizer.mask_token
            elif prob < 0.9:
                token = random.choice(list(tokenizer.get_vocab().keys()))
            else:
                pass
            
            masked_tokens[i] = token
    return masked_tokens, labels

def collate_fn(batch, tokenizer:EsmTokenizer):
    seqs, label_ids = tuple(zip(*batch))
    label_ids = pad_sequences(label_ids, constant_value=-100) # WARN: most commonly use -100
    labels = {"labels": label_ids}
		
    encoder_info = tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
    inputs = {"inputs": encoder_info}
    
    return inputs, labels

# Example for encoding a sequence (or a batch of sequences)
seq = 'MSLFLCLYKSTESLNSEDSLSLSASLHRFFFSFLFSLSLSLLFRRAFFSIAYQQEVSPKCLSTEVSPIQLSFPHSAYQQEVSPIQKCLSTEVSPIQGFLFPKLSFPHSAYQQEKSHFQFFFLNSLSSKESLSNSGFLSLQSLQLLFFSQQESLSIQCFLFPQCLSTKVSPIQSFLLHINKNQQVFIFKRRNSAYQQESLFCFLFPLYNRAYQKKRRAFFSPQCLSPIQCFLFQCLSTRESLKTFFSPQCLSTRESLQFMLFFPTVLINKRVSPIQSFLLQCLSTRESLQFRALSFPHSAYQQESLSQFRAFFFPIVLINKRVSQFRAFFSYTVNKRLSTRKVPLQDRNQLVFFPTVLINKRVSPIQSFLFPTVLINRESLQFSAFFSPQCLSTMAKSLQFRLSFSPKQCLSTRVSLQFRAFFFFKSFLFPTVLINKRVSPIQECLNSAYQQESLSIQAFFFPTVLINKKSLQFGSLSTRESSNSFAFFPKVLIQQAVSPICFLFPTVLINRVSPIQSFLLVLINKRVSNSEVLERLFLQFINRESLQFRAFFPHSAYQQESLSNSVVSFPYSAYQQESLSNSFLLTSLQAFFSHSAYQQESLSNSELSFQCQQVSSQFRAFFSPQCLSTESLSNSGFLFSPTVLINKRVSPIQSLFLFPIVLINKRVSPIQSFLFPTVLINKRVSPIQSFLFPTVLINKRDAYQQRESLQFRAFFSPQCLSTRESLNSESLFLFPTVLINKRVSPIQSFLFPTVLINKRVSLNSVSFPIAFFSPQCLSTRESLQFRELSFLFPTVLINRESLQFRAFFSPTVLINKSLNSELFFLSTRVSPIQCFLFPHSAYQQESLSNSELSFPHSAYQQESLSNSELSFPHSAYQQESLSNSELSFPHSAYQQESLSNSVLSFPYSAYQQESLSNSELSFPHSAYQQESLSNSELSFPHSAYQQESLSNSVLSFPHSAYQQESLSNSELSFPHSAYQQESLSNSVLSFPHSAYQQESLSNSELSFLQFRAFFSFPTVLINKR'

seq1 = 'MSLFLCLYKSTESLNSEDSLSLSASLHRFFFSFLFSLSLSLLFRRAFFSIAYQQEVSPKCLSTEVSPIQLSFPHSAYQQEVSPIQKCLSTEVSPIQGFLFPKLSFPHSAYQQEKSHFQFFFLNSLSSKESLSNSGFLSLQSLQLLFFSQQESLSIQCFLFPQCLSTKVSPIQSFLLHINKNQQVFIFKRRNSAYQQESLFCFLFPLYNRAYQKKRRAFFSPQCLSPIQCFLFQCLSTRESLKTFFSPQCLSTRESLQFMLFFPTVLINKRVSPIQSFLLQCLSTRESLQFRALSFPHSAYQQESLSQFRAFFFPIVLINKRVSQFRAFFSYTVNKRLSTRKVPLQDRNQLVFFPTVLINKRVSPIQSFLFPTVLINRESLQFSAFFSPQCLSTMAKSLQFRLSFSPKQCLSTRVSLQFRAFFFFKSFLFPTVLINKRVSPIQECLNSAYQQESLSIQAFFFPTVLINKKSLQFGSLSTRESSNSFAFFPKVLIQQAVSPICFLFPTVLINRVSPIQSFLLVLINKRVSNSEVLERLFLQFINRESLQFRAFFPHSAYQQESLSNSVVSFPYSAYQQESLSNSFLLTSLQAFFSHSAYQQESLSNSELSFQCQQVSSQFRAFFSPQCLSTESLSNSGF'

masked_seq, labels = encode_sequence(seq, tokenizer, seq_len)
masked_seq1, labels1 = encode_sequence(seq1, tokenizer, seq_len)
inputs, labels = collate_fn([(masked_seq, labels), (masked_seq1, labels1)], tokenizer)
inputs = inputs["inputs"]
labels = labels["labels"]

# del inputs["attention_mask"]

"""inputs
inputs:
    input_ids: shape [B, seqlen+1]
    attention_mask: shape [B, seqlen+1] # 1 for real tokens, 0 for padding
"""

# Cast to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def cast_to_device(inputs, device):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    else:
        for k,v in inputs.items():
            inputs[k] = cast_to_device(v, device)
    return inputs

inputs = cast_to_device(inputs, device)
labels = cast_to_device(labels, device)

# Forward
with torch.no_grad():
    outputs = model(**inputs, labels=labels, output_hidden_states=True)
    print(outputs.logits.shape) # [B, seqlen+1, vocab_size]
    # if want to get hidden states
    # outputs = model(**inputs, output_hidden_states=True, labels=labels)
    # outputs.hidden_states # Tuple (B, seqlen+2, hidden_size) * # L