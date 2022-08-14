from torch.utils.data import Dataset, DataLoader
from utils.sample import Sample, SampleList
import json
from utils.logger import Log
import torch

logger = Log(__name__).getlog()


class MyDataset(Dataset):
    def __init__(self, tokenizer, args, dataset_path, dataset_flag,label_encoder):
        self.tokenizer = tokenizer
        self.args = args
        self.dataset_flag = dataset_flag
        with open(dataset_path, "r", encoding="utf-8") as f:
            annos = f.readlines()
        self.all_anno = [json.loads(_) for _ in annos]
        self.label_encoder = label_encoder
        logger.info(f"{dataset_path} dataset length: {len(self.all_anno)}")
        if self.dataset_flag == 'train':
            self.fit_label()

    def __len__(self):
        return len(self.all_anno)

    def fit_label(self):
        category = [anno["category"] for anno in self.all_anno]
        self.label_encoder.fit(category)

    def get_label_encoder(self):
        return self.label_encoder

    def __getitem__(self, idx):
        anno = self.all_anno[idx]
        category, input_text = anno["category"], anno["input_text"]
        # 处理输出数据
        # 处理输入数据
        input_text_tokens = self.tokenizer.cls_token + category + \
            self.tokenizer.sep_token + input_text + self.tokenizer.sep_token
        # input_text_tokens = self.tokenizer.cls_token + input_text + self.tokenizer.sep_token
        input_text_tokens = self.tokenizer.tokenize(input_text_tokens)
        # if "output_text_str" in anno.keys():
        #     output_text_str = anno["output_text_str"]
        # else:
        #     output_text_str = output_text
        label = self.label_encoder.transform(category)

        anno_dict = {"category_str": category, "input_text_body_str": input_text, "input_text_str": [category, input_text],"input_text_tokens": input_text_tokens,"label":label}
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
        return data


def BuildDataloader(dataset, batch_size, shuffle, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=dataset.collate_fn, num_workers=num_workers)
    return dataloader
