import torch.nn as nn
from transformers import BertTokenizer
from src.generate.encoder import Encoder
from utils.utils import Cross_attention
import torch


class ClsModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        encoder = Encoder(args)
        self.encoder = encoder.encoder
        self.tokenizer = encoder.tokenizer  # 整体的tokenizer用encoder的tokenizer
        # 创建decoder
        # 定义decoder tokenizer
        special_tokens = [f"[unused{_}]" for _ in range(
            1, 100)] + ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<S>', '<T>']
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})
        self.tokenizer.bos_token = '<S>'
        self.tokenizer.eos_token = '<T>'
        self.lin1 = nn.Linear(768,128)
        self.lin2 = nn.Linear(128,32)
        self.predict = nn.Linear(32,args.num_classes)
        self.dropout = nn.Dropout(args.dropout_prob)



    def forward(self, input_texts):
        input_texts_embs = self.encoder(
            input_texts["input_ids"], attention_mask=input_texts["attention_mask"], token_type_ids=input_texts["token_type_ids"])[0]
        seq_embs = torch.transpose(input_texts_embs[0],0,1)
        seq_embs = seq_embs[0]
        x = self.lin1(seq_embs)
        x = self.dropout(x)
        pred = self.lin2(x)
        return self.predict(pred)


