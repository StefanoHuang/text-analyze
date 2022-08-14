import torch
from src.generate.model import GenModel
from utils.logger import Log
from torch.cuda.amp import autocast, GradScaler
from src.generate.optimizer import Optimizer
from src.generate.dataset import MyDataset, BuildDataloader
from src.generate.dataset_path import dataset_path
from utils.utils import DecodingBCEWithMaskLoss, GenerateOOV, AverageMeter
import os
import time
import torch.nn as nn
from src.generate.eval import eval_model
from utils.utils import data_list_to_excel

logger = Log(__name__).getlog()

class Generator():
    def __init__(self):
        logger.info(f"Generator is processing")
    def run(self, args):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GenModel(args)
        # 加载模型
        if args.resume_file:
            ori_model_state_dict = torch.load(args.resume_file)
            model.load_state_dict(ori_model_state_dict, strict=True)
            logger.info("successfully load the previous checkpoint from {args.resume_file}")
        model = model.to(device)    # model中的tensor不会转到devcie，只有变量才会转到devcie

        scaler = GradScaler()
        # 数据加载
        dataset_trian_path, dataset_dev_path, dataset_test_path = dataset_path(args.dataset_name)
        train_set = MyDataset(model.tokenizer, model.dec_tokenizer, args, dataset_trian_path, dataset_flag="train")
        train_loader = BuildDataloader(train_set, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
        dev_set = MyDataset(model.tokenizer, model.dec_tokenizer, args, dataset_dev_path, dataset_flag="dev")
        dev_loader = BuildDataloader(dev_set, batch_size=args.dev_bs, shuffle=False, num_workers=args.num_workers)
        test_set = MyDataset(model.tokenizer, model.dec_tokenizer, args, dataset_test_path, dataset_flag="test")
        test_loader = BuildDataloader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

        # 优化器加载
        steps_per_epoch = len(train_loader)
        optimizer_class = Optimizer(args, [model.encoder], model, steps_per_epoch)
        optimizer, scheduler = optimizer_class.get_optimizer()

        dec_lossfn = DecodingBCEWithMaskLoss(len(model.dec_tokenizer))

        min_loss = 1e6
        best_bleu_dev = best_bleu_test = 0
        best_dev_epoch = 0
        generator = GenerateOOV(model.dec_tokenizer)

        # 创建输出目录
        output_path = f"{args.output}/{args.exp_name}"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if args.exp_name == "debug":
            save_prefix = "debug-"
            if args.resume_file:
                save_prefix += args.resume_file.split("/")[-1]
        else:
            save_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())
        train_model_output_path = f"{output_path}/{save_prefix}-min_loss.pth"
        dev_model_output_path = f"{output_path}/{save_prefix}-dev.pth"
        test_model_output_path = f"{output_path}/{save_prefix}-test.pth"
        dev_rslt_output_path = f"{output_path}/{save_prefix}-dev.xlsx"
        test_rslt_output_path = f"{output_path}/{save_prefix}-test.xlsx"

        for epoch in range(args.epochs):
            if "train" in args.running_type:
                epoch_losses = AverageMeter()
                model.train()
                optimizer.zero_grad()
                for step, data in enumerate(train_loader):
                    data.input_texts = data.input_texts.to(device)
                    data.output_texts = data.output_texts.to(device)
                    data.copy_target = data.copy_target.to(device)
                    with autocast():
                        scores_pred, match_pred = model(data.input_texts, data.output_texts)
                        generate_loss = dec_lossfn(scores_pred, data.output_texts["input_ids"][:,1:], data.copy_target[:,:-1], loss_mask=data.output_texts["attention_mask"][:,:-1])
                        total_loss = generate_loss
                    assert torch.isnan(total_loss).sum() == 0
                    # 正常梯度回传
                    # optimizer.zero_grad()
                    # total_loss.backward()
                    # optimizer.step()
                    # scheduler.step()
                    scaler.scale(total_loss).backward()
                    if ((step + 1) % args.accum_iter == 0) or ((step + 1) == len(train_loader)):
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                        scaler.step(optimizer)
                        scaler.update() 
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    batch = scores_pred.size(0)
                    epoch_losses.update(total_loss.item(), batch)
                    if step % 100 == 0:
                        logger.info(f"{epoch}-{step} | avg_loss: {epoch_losses.avg}, cur_loss: {total_loss.item()}")

                if epoch_losses.avg < min_loss and args.exp_name != "debug":
                    min_loss = epoch_losses.avg
                    torch.save(model.state_dict(), train_model_output_path)
            
            if "dev" in args.running_type:  # TO LOOK
                model.eval()
                dev_all_predictions, dev_blue_score = eval_model(model, model.dec_tokenizer, dev_loader, args, generator)
                logger.info(f"epoch: {epoch}, dev_blue_score: {dev_blue_score}")
                if dev_blue_score > best_bleu_dev:
                    best_bleu_dev = dev_blue_score
                    best_dev_epoch = epoch
                    data_list_to_excel(dev_all_predictions, dev_rslt_output_path)
                    if args.exp_name != "debug":
                        torch.save(model.state_dict(), dev_model_output_path)

            if "test" in args.running_type:
                model.eval()
                test_all_predictions, test_blue_score = eval_model(model, model.dec_tokenizer, test_loader, args, generator)
                logger.info(f"epoch: {epoch}, test_blue_score: {test_blue_score}")
                if test_blue_score > best_bleu_test:
                    best_bleu_test = test_blue_score
                    data_list_to_excel(test_all_predictions, test_rslt_output_path)
                    if args.exp_name != "debug":
                        torch.save(model.state_dict(), test_model_output_path)
        if "dev" in args.running_type:
            logger.info(f"best dev epoch is {best_dev_epoch}, the best dev bleu is {best_bleu_dev}")



        
        