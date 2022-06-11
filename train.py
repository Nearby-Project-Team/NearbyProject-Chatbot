from calendar import EPOCH
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from ChatbotDataset import ChatbotDataset, collate_batch
from torch.utils.data import DataLoader
from tqdm import tqdm

BOS = "<\s>"
EOS = "<\e>"
UNK = "<unk>"
PAD = "<pad>"
MASK = "<mask>"

MODEL_NAME = "./checkpoint/Nearby-Model-"

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token=UNK,
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

Chatbot_Data = pd.read_csv("ChatBotData.csv")
# Test 용으로 300개 데이터만 처리한다.
Chatbot_Data = Chatbot_Data[:300]
Chatbot_Data.head()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ChatbotDataset(Chatbot_Data, max_len=40)
#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, batch_size=16, num_workers=0, shuffle=True, collate_fn=collate_batch,)

model.to(device)
model.train()

learning_rate = 3e-3
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

EPOCH = 20
Sneg = -1e18

print ("start")
for epoch in range(EPOCH):
    print("Epoch ", epoch)
    for batch_idx, samples in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)
        out = model(token_ids)
        out = out.logits      #Returns a new tensor with the logit of the elements of input
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2).to(device)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out)).to(device)
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        # 학습 끝
        optimizer.step()
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, MODEL_NAME + str(epoch) + ".pt")
    
    print("Epoch: ", epoch, "loss: ", avg_loss.value)
print ("end")