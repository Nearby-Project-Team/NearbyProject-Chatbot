import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from pytorch_lightning import Trainer

# koGPT2 tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 


