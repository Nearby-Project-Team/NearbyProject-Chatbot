from transformers import PreTrainedTokenizerFast
from main import KoGPT2Chat
import torch
import argparse

parser = argparse.ArgumentParser(description='TorchScript Serializer KoGPT-2 model')
parser.add_argument('--model_params',
                    type=str,
                    default='checkpoint/model_-last.ckpt',
                    help='model binary for starting chat')

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

if __name__ == "__main__":
    
    parser = KoGPT2Chat.add_model_specific_args(parser)
    args = parser.parse_args()
    model: KoGPT2Chat = KoGPT2Chat.load_from_checkpoint(args.model_params)
    torch.save(model.kogpt2.state_dict(), "./checkpoint/model-best.pth")
