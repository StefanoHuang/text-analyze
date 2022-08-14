from asyncio.log import logger
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer


class Encoder():
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
        config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=args.pretrain_model, output_hidden_states=True)
        self.encoder = BertModel.from_pretrained(
            args.pretrain_model, config=config)
        if len(args.special_tokens) > 0:
            with torch.no_grad():
                expand_embs = []
                clean_tokens = []
                for special_token in args.special_tokens:
                    if special_token not in self.tokenizer.vocab:
                        clean_tokens.append(special_token)
                        token_emb = self.tokenizer(special_token, return_tensors="pt")
                        token_emb = self.encoder(token_emb["input_ids"], attention_mask=token_emb["attention_mask"], token_type_ids=token_emb["token_type_ids"])[0]
                        expand_embs.append(token_emb[:, 0])    # use [cls] embedding
                if len(clean_tokens):
                    expand_embs = torch.cat(expand_embs, dim=0)
                    ori_weight = self.encoder.embeddings.word_embeddings.weight
                    self.encoder.embeddings.word_embeddings = nn.Embedding(ori_weight.size(0) + len(clean_tokens), ori_weight.size(1), padding_idx=0)
                    self.encoder.embeddings.word_embeddings.weight[:ori_weight.size(0)] = ori_weight
                    self.encoder.embeddings.word_embeddings.weight[ori_weight.size(0):] = expand_embs
                    self.tokenizer.add_special_tokens({"additional_special_tokens": clean_tokens})
                    logger.info(f"Success reuse the embedding for new special tokens")

