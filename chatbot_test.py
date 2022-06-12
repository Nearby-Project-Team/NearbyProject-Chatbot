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
model_weight = torch.load('./checkpoint/Nearby-Model-16.pt')
model.load_state_dict(model_weight["model_state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()

if __name__ == "__main__":
    with torch.no_grad():
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            a = ""
            while 1:
                tokens = koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + '0' + A_TKN + a)
                print(tokens)
                input_ids = torch.LongTensor(tokens).unsqueeze(dim=0).to(device=device)
                pred = model(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            print("Chatbot > {}".format(a.strip()))