import argparse

class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.use_data_args()
        self.adjust_base_args()
        self.adjust_hyper_args()

    def get_parser(self):
        return self.parser.parse_args()

    def use_data_args(self):
        self.parser.add_argument("--dataset_trian", type=str, default="./data/cls/train.txt", help="the dir of train dataset")
        self.parser.add_argument("--dataset_dev", type=str, default="./data/cls/dev.txt", help="the dir dir of dev dataset")
        self.parser.add_argument("--dataset_test", type=str, default="./data/cls/test.txt", help="the dir dir of test dataset")
        self.parser.add_argument("--output", type=str, default="./model/exp1", help="save the checkpoints and logs")
        self.parser.add_argument("--resume_file", type=str, default="", help="resume the checkpoint")
        self.parser.add_argument("--num_classes", type=int, default=2, help="Number for classification")

    def adjust_base_args(self):
        self.parser.add_argument("--task", type=str, default="") # gen、cls
        self.parser.add_argument("--seed", type=int, default=10)
        self.parser.add_argument("--num_workers", type=int, default=8)
        self.parser.add_argument("--pretrain_model", type=str, default=f"../BERT_model/chinese_roberta_wwm_ext", 
                                help="the chinese pretrained model")
        self.parser.add_argument("--running_type", type=list, default=["trian", "dev", "test"])
        

    def adjust_hyper_args(self):
        # model hyper parameters
        self.parser.add_argument("--vocab_size", type=int, default=1)
        self.parser.add_argument("--special_tokens", type=list, default=["测试"])
        self.parser.add_argument("--max_input_size", type=int, default=192)
        self.parser.add_argument("--max_output_size", type=int, default=175)
        self.parser.add_argument("--num_hidden_layers", type=int, default=4)
        # training hyper parameters
        self.parser.add_argument("--lr", type=float, default=1e-5)
        self.parser.add_argument("--lr_scale", type=float, default=0.01)
        self.parser.add_argument("--dropout_prob", type=float, default=0.1)
        self.parser.add_argument("--weight_decay", type=float, default=2e-4)
        self.parser.add_argument("--accum_iter", type=int, default=4)
        self.parser.add_argument("--epochs", type=int, default=20)
        self.parser.add_argument("--train_bs", type=int, default=8)
        self.parser.add_argument("--dev_bs", type=int, default=8)
        self.parser.add_argument("--copy_source", type=str, default="content")  # title、passage       
        self.parser.add_argument("--share_tokenizer", type=bool, default=True)
        self.parser.add_argument("--share_word_embedding", type=bool, default=True)
        self.parser.add_argument("--using_RL", type=bool, default=False)    
        # 需要丢弃的参数
        self.parser.add_argument("--hidden_size", type=int, default=768)
        self.parser.add_argument("--num_attention_heads", type=int, default=12)
        self.parser.add_argument("--num_beams", type=int, default=5)
        self.parser.add_argument("--shuffle_terms", type=bool, default=False)
        self.parser.add_argument("--compute_matching", type=bool, default=False)
        self.parser.add_argument("--replace_ratio", type=float, default=0)
        self.parser.add_argument("--using_lac", type=bool, default=False)
        self.parser.add_argument("--copy_wo_gen", type=float, default=0)
        self.parser.add_argument("--title_reformat", type=bool, default=False)
        self.parser.add_argument("--title_shuffle", type=bool, default=False)
        self.parser.add_argument("--paragraph_shuffle_ratio", type=float, default=0)

flags = Flags()
args_cls = flags.get_parser()
