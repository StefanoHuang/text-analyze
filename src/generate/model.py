import torch.nn as nn
from transformers import BertTokenizer
from src.generate.encoder import Encoder
from src.generate.decoder import Decoder
from utils.utils import Cross_attention
import torch


class GenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 创建encoder
        encoder = Encoder(args)
        self.encoder = encoder.encoder
        self.tokenizer = encoder.tokenizer  # 整体的tokenizer用encoder的tokenizer
        # 创建decoder
        self.decoder = Decoder(args)
        self.dec_config = self.decoder.config
        # 定义decoder tokenizer
        special_tokens = [f"[unused{_}]" for _ in range(
            1, 100)] + ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<S>', '<T>']
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})
        self.tokenizer.bos_token = '<S>'
        self.tokenizer.eos_token = '<T>'
        if args.share_tokenizer:
            self.dec_tokenizer = self.tokenizer
        else:
            self.dec_tokenizer = BertTokenizer(args.dec_tokenizer_vocab)
            self.dec_tokenizer.add_special_tokens(
                {'additional_special_tokens': special_tokens})
            self.dec_tokenizer.bos_token = '<S>'
            self.dec_tokenizer.eos_token = '<T>'
        dec_vocab_size = len(self.dec_tokenizer)

        self.dropout = nn.Dropout(args.dropout_prob)
        self._build_output(dec_vocab_size)

    def _build_output(self, vocab_size):
        self.match_classifier = nn.Linear(self.dec_config.hidden_size, 1)
        if not self.args.share_word_embedding:
            self.dec_classifer = nn.Linear(
                self.dec_config.hidden_size, vocab_size)
            if self.args.init_from_encoder_embedding and self.args.share_tokenizer:
                self.dec_classifer.weight = self.encoder.embeddings.word_embeddings.weight
        else:
            assert self.args.share_tokenizer == True
        self.copy_classifier = Cross_attention(self.dec_config.hidden_size)

    def _forward_output(self, dec_output, copy_emb, copy_mask):
        if hasattr(self, "dec_classifer"):
            fixed_scores = self.dec_classifer(dec_output)
        else:
            fix_vocab_weight = self.encoder.embeddings.word_embeddings.weight
            fix_vocab_weight = fix_vocab_weight.unsqueeze(0).expand(
                dec_output.size(0), -1, -1).transpose(1, 2)
            fixed_scores = torch.bmm(dec_output, fix_vocab_weight)
        copy_scores = self.copy_classifier(dec_output, copy_emb, copy_mask)
        scores = torch.cat([fixed_scores, copy_scores], dim=-1)
        return scores

    def forward(self, input_texts, output_texts):
        if hasattr(self, "dec_classifer"):
            fix_vocab_weight = self.dec_classifer.weight
        else:
            fix_vocab_weight = self.encoder.embeddings.word_embeddings.weight
        input_texts_embs = self.encoder(
            input_texts["input_ids"], attention_mask=input_texts["attention_mask"], token_type_ids=input_texts["token_type_ids"])[0]
        attention_mask = input_texts["attention_mask"]
        if self.args.copy_source == "category":
            copy_mask = 1 - input_texts["token_type_ids"].float()
        elif self.args.copy_source == "body":
            copy_mask = input_texts["token_type_ids"].float()
        else:
            copy_mask = input_texts["attention_mask"]
        if self.training:
            content_output, dec_output = self.decoder(input_texts_embs, attention_mask=attention_mask, fixed_ans_emb=fix_vocab_weight,
                                                      prev_ids=output_texts["input_ids"][:, :-1], prev_mask=output_texts["attention_mask"][:, :-1])
            scores = self._forward_output(
                dec_output, input_texts_embs, copy_mask)
        else:
            start_ids = input_ids = output_texts["input_ids"].clone()
            start_mask = input_mask = output_texts["attention_mask"].clone()
            for t in range(1, self.args.max_output_size):
                content_output, dec_output = self.decoder(input_texts_embs, attention_mask=attention_mask, fixed_ans_emb=fix_vocab_weight,
                                                          prev_ids=input_ids, prev_mask=input_mask)
                scores = self._forward_output(dec_output, input_texts_embs, copy_mask)
                output_text_ids = scores.argmax(dim=-1)
                input_ids = torch.cat([start_ids, output_text_ids], dim=-1)
                input_mask = torch.cat([start_mask, input_mask], dim=-1)
        match_pred = self.match_classifier(input_texts_embs[: 0]) # encoder cls token
        return scores, match_pred

