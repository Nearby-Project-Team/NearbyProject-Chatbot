import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

from ts.torch_handler.base_handler import BaseHandler

class ChatbotHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        
        self.U_TKN = '<usr>'
        self.S_TKN = '<sys>'
        self.BOS = '</s>'
        self.EOS = '</s>'
        self.MASK = '<unused0>'
        self.SENT = '<unused1>'
        self.PAD = '<pad>'
        
        self.koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=self.BOS, eos_token=self.EOS, unk_token='<unk>',
            pad_token=self.PAD, mask_token=self.MASK) 
    
    def initialize(self, context):
        self._context = context
        self.initialized = True
    
    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        return preprocessed_data
    
    def inference(self, data):
        model_output = self.model.forward(data)
        return model_output
    
    def postprocess(self, data):   
        return data.strip()
    
    def handle(self, data, context):
        a = ""
        while 1 :
            _data = self.preprocess(data)
            tokens = self.koGPT2_TOKENIZER.encode(self.U_TKN + _data + self.SENT + '0' + self.S_TKN + a)
            input_ids = torch.LongTensor(tokens).unsqueeze(dim=0)
            model_output = self.inference(input_ids)
            pred = pred.logits
            gen = self.koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).cpu().squeeze().numpy().tolist())[-1]
            if gen == self.EOS:
                break
            a += gen.replace("▁", " ")
        return self.postprocess(a)
    
    