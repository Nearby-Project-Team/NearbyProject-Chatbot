import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

MODEL_NAME = "./checkpoint/Nearby-Model-"

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model_weight = torch.load('./checkpoint/Nearby-Model-20.pt')
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
                tokens = koGPT2_TOKENIZER.encode(U_TKN + q + SENT + '0' + S_TKN + a)
                input_ids = torch.LongTensor(tokens).unsqueeze(dim=0).to(device=device)
                pred = model(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).cpu().squeeze().numpy().tolist())[-1]
                if gen == EOS or len(a) > 60:
                    break
                a += gen.replace("▁", " ")
            print("Chatbot > {}".format(a.strip()))
