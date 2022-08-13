from calendar import EPOCH
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
from transformers.optimization import get_cosine_schedule_with_warmup
from ChatbotDataset import ChatbotDataset, collate_batch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

MODEL_NAME = "./checkpoint/Nearby-Model-"

EPOCH = 100
Sneg = -1e18

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

chatdata_dir = glob.glob('./ChatData/*.csv')
ChatbotData = []
for chat in chatdata_dir:
    Chatbot_Data = pd.read_csv(chat)
    Chatbot_Data.head()
    ChatbotData.append(Chatbot_Data)
ChatListData = pd.concat(ChatbotData)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ChatbotDataset(ChatListData, max_len=80)
#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True, collate_fn=collate_batch,)

model.to(device)
model.train()

learning_rate = 5e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

num_train_steps = len(train_dataloader) * EPOCH
num_warmup_steps = int(num_train_steps * 0.1)

scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

print ("start")
for epoch in range(EPOCH):
    print("Epoch ", epoch + 1)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # 학습 끝
        optimizer.step()
    
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                }, MODEL_NAME + str(epoch + 1) + ".pt")
        torch.save(model.state_dict(), MODEL_NAME + str(epoch + 1) + ".pth")
    
    print(f"Epoch {epoch + 1}, loss: {avg_loss:.3f}")
print ("end")
