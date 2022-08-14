import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model, dataloader,  break_num=None):
    count = 0
    accurate = 0.0
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            if break_num and step > break_num:
                break
            data.input_texts = data.input_texts.to(device)

            scores_pred = model(data.input_texts)
            y_pred = scores_pred.argmax(dim=-1)
            count += 1
            accurate += torch.eq(y_pred, data.label.squeeze(dim=-1)).float()
    return accurate/count
