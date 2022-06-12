import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from ChatbotDataset import ChatbotDataset, collate_batch
from torch.utils.data import DataLoader
from tqdm import tqdm

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "<\s>"
EOS = "<\e>"
UNK = "<unk>"
PAD = "<pad>"
MASK = "<mask>"
SENT = '<unused1>'

MODEL_NAME = "./checkpoint/Nearby-Model-"

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token=UNK,
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

if __name__ == "__main__":
    with torch.no_grad():
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            a = ""
            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + '0' + A_TKN + a)).unsqueeze(dim=0)
                pred = model(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            print("Chatbot > {}".format(a.strip()))