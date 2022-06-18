import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from ts.torch_handler.base_handler import BaseHandler
import os

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
        
        self.model = None
    
    def initialize(self, context):
        self._context = context
        
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model_weight = torch.load(model_pt_path, map_location=self.device)
        self.model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.model.load_state_dict(self.model_weight)
        self.model.cuda()
        self.initialized = True
    
    def preprocess(self, data):
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        return preprocessed_data.decode()
    
    def inference(self, data):
        model_output = self.model.forward(data)
        return model_output
    
    def postprocess(self, data):   
        return data.strip()
    
    def handle(self, data, context):
        a = ""
        while 1 :
            _data = self.preprocess(data)
            print(_data)
            tokens = self.koGPT2_TOKENIZER.encode(self.U_TKN + _data + self.SENT + '0' + self.S_TKN + a)
            input_ids = torch.LongTensor(tokens).unsqueeze(dim=0).to(device=self.device)
            model_output = self.inference(input_ids)
            model_output = model_output.logits
            gen = self.koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(model_output, dim=-1).cpu().squeeze().numpy().tolist())[-1]
            if gen == self.EOS:
                break
            a += gen.replace("▁", " ")
        print(a)
        return [self.postprocess(a)]
    
    