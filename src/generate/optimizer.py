from transformers import AdamW, get_cosine_schedule_with_warmup


class Optimizer():
    def __init__(self, args, finetune_model=[], all_model=None, steps_per_epoch=0):
        self.args = args
        self.finetune_model = finetune_model
        self.all_model = all_model
        self.steps_per_epoch = steps_per_epoch
    
    def get_optimizer_parameters(self):
        optimizer_params_groups = []
        finetune_params_set = set()
        if self.args.finetune:
            for m in self.finetune_model:
                optimizer_params_groups.append({"params": list(m.parameters()), "lr": self.args.lr * self.args.lr_scale})
                finetune_params_set.update(list(m.parameters()))
        remaining_params = [p for p in self.all_model.parameters() if p not in finetune_params_set]
        optimizer_params_groups.insert(0, {"params": remaining_params})
        return optimizer_params_groups

    def get_optimizer(self):
        optimizer = AdamW(self.get_optimizer_parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        warm_up_step = self.steps_per_epoch // self.args.accum_iter
        all_step = self.args.epochs * warm_up_step
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_step, all_step)
        return optimizer, scheduler