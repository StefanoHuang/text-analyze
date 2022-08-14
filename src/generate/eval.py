from sacrebleu import corpus_bleu, sentence_bleu
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model, dec_tokenizer, dataloader, args, generator, break_num=None):
    all_predictions = []
    all_cands = []
    all_refs = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            if break_num and step > break_num:
                break
            data.input_texts = data.input_texts.to(device)
            data.output_texts = data.output_texts.to(device)

            scores_pred, match_pred = model(data.input_texts, data.output_texts)
            output_text_ids = scores_pred.argmax(dim=-1).tolist()

            for i in range(len(output_text_ids)):
                output_text_ids_each = output_text_ids[i]
                final_token, pred_token = generator.generate(
                    output_text_ids_each, data.input_text_tokens[i])
                cand = " ".join(list(final_token))  # 取final_token来计算bleu
                ref = " ".join(list(data.output_text_str[i]))
                bleu = sentence_bleu(cand, [ref]).score
                batch_sample = {"id": data.id[i], "category": data.category_str[i], "input_text_body": data.input_text_body_str[i],
                                "target": data.output_text_str[i], "generation": final_token, "generation_mid": pred_token, "bleu": bleu}
                all_predictions.append(batch_sample)
                all_cands.append(cand)
                all_refs.append(ref)
    blue_score = corpus_bleu(all_cands, [all_refs])
    blue_score_data = blue_score.score

    return all_predictions, blue_score_data
