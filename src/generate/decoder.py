from transformers import BertPreTrainedModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(BertPreTrainedModel):
    def __init__(self, args):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path=args.pretrain_model,
                                            output_hidden_states=True, num_hidden_layers=args.num_hidden_layers)
        super().__init__(config)
        self.prev_pred_embeddings = PrevPredEmbeddings(config, args)
        self.encoder = BertEncoder(config)
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.hidden_size))  # 用来做多任务学习-分类任务
        self.cls_mask = torch.ones(1,1) # 同上
        self.init_weights()  # 初始化模型参数

    def forward(self, paragraph_embs, attention_mask, fixed_ans_emb, prev_ids, prev_mask):
        batch = paragraph_embs.size(0)
        device = paragraph_embs.device

        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, paragraph_embs, prev_ids)

        cls_emb = self.cls_token.expand(batch, -1, -1).to(device)
        encoder_inputs = torch.cat([cls_emb, paragraph_embs, dec_emb], dim=1)
        dec_max_num = dec_emb.size(1)
        content_max_num = paragraph_embs.size(1)
        dec_mask = torch.zeros(batch, dec_max_num, dtype=torch.float32, device=device)
        cls_mask = self.cls_mask.expand(batch, -1).to(device)
        attention_mask = torch.cat([cls_mask, attention_mask, dec_mask], dim=1)

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # bert的attention_mask维度为 [batch_size, num_heads, from_seq_length, to_seq_length], from_seq_length第二层，to_seq_length第一层
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # 增加num_heads、from_seq_length
        extended_attention_mask = extended_attention_mask.repeat(1, 1, from_seq_length, 1)
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_casual_mask(dec_max_num, encoder_inputs.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        
        encoder_outputs = self.encoder(encoder_inputs, extended_attention_mask)[0]
        cls_output = encoder_outputs[:, 0]
        content_output = encoder_outputs[:, 1:content_max_num+1]
        dec_output = encoder_outputs[:, -dec_max_num:]

        return content_output, dec_output


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        MAX_DEC_LENGTH = args.max_output_size
        MAX_TYPE_NUM = args.type_vocab_size
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)
        
        BertLayerNorm = torch.nn.LayerNorm
        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.copy_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, ans_emb, copy_emb, prev_ids):
        assert prev_ids.dim() == 2 and prev_ids.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_ids.size(0)
        seq_length = prev_ids.size(1)
        ans_num = ans_emb.size(0)

        ans_emb = self.ans_layer_norm(ans_emb)  # 词表embedding向量，用于输出从词表中生成词
        copy_emb = self.copy_layer_norm(copy_emb)   # encoder输出向量，用于输出从输入句子复制词
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_copy_emb_cat = torch.cat([ans_emb, copy_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_copy_emb_cat, prev_ids)
        
        position_ids = torch.arange(seq_length, dtype=torch.long, device=copy_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_ids = prev_ids.ge(ans_num).long()   # Token type ids: 0 for ans_emb， 1 for copy_emb
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _batch_gather(x, ids):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.contiguous().view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=ids.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == ids.dim()
    ids_flat = batch_offsets + ids
    results = F.embedding(ids_flat, x_flat)
    return results

def _get_casual_mask(seq_length, device):
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask