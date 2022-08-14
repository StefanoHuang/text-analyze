from torch.utils.data import Dataset, DataLoader
from utils.sample import Sample, SampleList
import json
from utils.logger import Log
import torch

logger = Log(__name__).getlog()


class MyDataset(Dataset):
    def __init__(self, tokenizer, dec_tokenizer, args, dataset_path, dataset_flag):
        self.tokenizer = tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.args = args
        self.dataset_flag = dataset_flag
        with open(dataset_path, "r", encoding="utf-8") as f:
            annos = f.readlines()
        self.all_anno = [json.loads(_) for _ in annos]
        logger.info(f"{dataset_path} dataset length: {len(self.all_anno)}")

    def copy_target(self, question, content):
        copy_array = torch.zeros(
            [self.args.max_output_size, self.args.max_input_size], dtype=torch.float32)
        for i, q in enumerate(question[1: self.args.max_output_size+1]):
            for j, t in enumerate(content[: self.args.max_input_size]):
                if q == t:
                    copy_array[i, j] = 1
        return copy_array

    def __len__(self):
        return len(self.all_anno)

    def __getitem__(self, idx):
        anno = self.all_anno[idx]
        data_id, output_text, category, input_text = anno["data_id"], anno[
            "output_text"], anno["category"], anno["input_text"]
        # 处理输出数据
        if self.dataset_flag != "train":
            output_text_tokens = self.dec_tokenizer.bos_token   # dev和test状态时，只有<s>作为输入
        else:
            output_text_tokens = self.dec_tokenizer.bos_token + output_text + \
                self.dec_tokenizer.eos_token + self.dec_tokenizer.sep_token
        output_text_tokens = self.dec_tokenizer.tokenize(
            output_text_tokens)    # 对原文进行分词
        # 处理输入数据
        input_text_tokens = self.tokenizer.cls_token + category + \
            self.tokenizer.sep_token + input_text + self.tokenizer.sep_token
        # input_text_tokens = self.tokenizer.cls_token + input_text + self.tokenizer.sep_token
        input_text_tokens = self.tokenizer.tokenize(input_text_tokens)

        copy_target = self.copy_target(output_text_tokens, input_text_tokens)
        # if "output_text_str" in anno.keys():
        #     output_text_str = anno["output_text_str"]
        # else:
        #     output_text_str = output_text
        anno_dict = {"category_str": category, "input_text_body_str": input_text, "input_text_str": [category, input_text], "output_text_str": output_text,
                     "id": data_id,"input_text_tokens": input_text_tokens, "output_text_tokens": output_text_tokens, "copy_target": copy_target}
        return Sample(anno_dict)

    def collate_fn(self, data):
        data = SampleList(data)
        data.input_texts = self.tokenizer(data.input_text_tokens, is_split_into_words=True, padding=True, max_length=self.args.max_input_size,
        truncation=True, return_tensors="pt", add_special_tokens=False)
        for i, tokens in enumerate(data.input_text_tokens):
            sep_idx = tokens.index("[SEP]")
            token_type_ids = data.input_texts["token_type_ids"][i]
            attention_mask = data.input_texts["attention_mask"][i]
            token_type_ids[sep_idx:] = 1
            data.input_texts["token_type_ids"][i] = token_type_ids * attention_mask
        data.output_texts = self.dec_tokenizer(data.output_text_tokens, is_split_into_words=True, padding=True, max_length=self.args.max_output_size,
        truncation=True, return_tensors="pt", add_special_tokens=False)
        input_texts_size = data.input_texts.input_ids.size(1)
        output_texts_size = data.output_texts.input_ids.size(1)
        data.copy_target = data.copy_target[:, :output_texts_size, :input_texts_size]
        if self.args.copy_source == "category":
            copy_mask = 1 - data.input_texts["token_type_ids"].float()
        elif self.args.copy_source == "body":
            copy_mask = data.input_texts["token_type_ids"].float()
        else:
            copy_mask = data.input_texts["attention_mask"]
        data.copy_target = data.copy_target * copy_mask.unsqueeze(1)
        return data


def BuildDataloader(dataset, batch_size, shuffle, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=dataset.collate_fn, num_workers=num_workers)
    return dataloader
