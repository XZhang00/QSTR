import argparse
import os, math
import torch
from data_utils import FastFTDataset_roberta
from transformers import RobertaTokenizer, AutoConfig, WEIGHTS_NAME, CONFIG_NAME, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import Adam, AdamW
import random
import logging
import numpy as np
from torch import nn
import time, json
from SimFT_model import SimFT_ffn_Roberta
from numpy import mean
sigmoid = nn.Sigmoid()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loss_mse_sort(args, output, ori_scores, cur_device):
    MSEloss = nn.MSELoss(reduction='sum')

    pred = sigmoid(output).reshape(-1, args.select_num)
    scores = ori_scores.unsqueeze(-1).reshape(-1, args.select_num)
    # print("----pred", pred.size(), scores.size())

    src_indice = torch.tensor([args.select_num - 2]).to(cur_device)
    src_preds = torch.index_select(pred, -1, src_indice).squeeze(-1)
    # print(f"src_preds: {src_preds}")
    ref_indice = torch.tensor([args.select_num - 1]).to(cur_device)
    ref_preds = torch.index_select(pred, -1, ref_indice).squeeze(-1)
    # print(f"ref_preds: {ref_preds}")

    src_scores = torch.index_select(scores, -1, src_indice).squeeze(-1)
    ref_scores = torch.index_select(scores, -1, ref_indice).squeeze(-1)

    # mse_loss = torch.mean(MSEloss(dis_z, pred).sum(-1))
    mse_loss = MSEloss(scores, pred)

    dis_diff = scores.unsqueeze(2) - scores.unsqueeze(2).transpose(1, 2)
    dis_mask = dis_diff.lt(0)  # 取小于0的部分，pred的差中小于0的是正确的，取大于0的值即为loss
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(2).transpose(1, 2)
    pred_mask = pred_diff.lt(0)

    rate = dis_mask.ne(pred_mask).sum() / (2 * dis_mask.sum() + 1e-9)

    loss_tmp = torch.maximum((pred_diff + (dis_diff * -1)) * dis_mask, torch.zeros_like(dis_mask).to(cur_device))
    # print("+++++loss-tmp", loss_tmp.size(), loss_tmp.sum(-1).sum(-1).size())
    loss_sort = torch.mean(loss_tmp.sum(-1).sum(-1))

    return loss_sort + mse_loss, loss_sort, mse_loss, torch.mean(src_scores), torch.mean(ref_scores), torch.mean(src_preds), torch.mean(ref_preds), rate


def get_loss_mse_sort_v2(args, output, ori_scores, cur_device):

    src_indice = torch.tensor([args.select_num - 2]).to(cur_device)
    src_preds = torch.index_select(sigmoid(output).reshape(-1, args.select_num), -1, src_indice).squeeze(-1)
    ref_indice = torch.tensor([args.select_num - 1]).to(cur_device)
    ref_preds = torch.index_select(sigmoid(output).reshape(-1, args.select_num), -1, ref_indice).squeeze(-1)
    src_scores = torch.index_select(ori_scores.unsqueeze(-1).reshape(-1, args.select_num), -1, src_indice).squeeze(-1)
    ref_scores = torch.index_select(ori_scores.unsqueeze(-1).reshape(-1, args.select_num), -1, ref_indice).squeeze(-1)

    pred = sigmoid(output)
    scores = ori_scores.unsqueeze(-1)
    # print("----pred", pred.size(), scores.size())

    MSEloss = nn.MSELoss(reduction='sum')
    mse_loss = MSEloss(scores, pred)

    dis_diff = scores - scores.transpose(0, 1)
    dis_mask = dis_diff.lt(0)  # 取小于0的部分，pred的差中小于0的是正确的，取大于0的值即为loss
    pred_diff = pred - pred.transpose(0, 1)
    pred_mask = pred_diff.lt(0)
    # print(f"dis_diff: {dis_diff.size()}, pred_diff: {pred_diff.size()}")
    rate = dis_mask.ne(pred_mask).sum() / (2 * dis_mask.sum() + 1e-9)

    loss_tmp = torch.maximum((pred_diff + (dis_diff * -1)) * dis_mask, torch.zeros_like(dis_mask).to(cur_device))
    # print("+++++loss-tmp", loss_tmp.size(), loss_tmp.sum(-1).size())
    loss_sort = torch.mean(loss_tmp.sum(-1))

    return  loss_sort + mse_loss, loss_sort, mse_loss, torch.mean(src_scores), torch.mean(ref_scores), torch.mean(src_preds), torch.mean(ref_preds), rate



def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())
    fr.close()
    return res


def eavluate(args, tokenizer, model, syn_vocab2id, device, path, parse_path):
    """
    计算当前模型在src/ref/tgt句法模板上的pred
    """
    sigmoid = nn.Sigmoid()
    pad_id = 1
    input_sens = read_file(path)
    res = []
    for _tree_type in ["src", "ref", "tgt"]:
        input_trees_path = parse_path.replace("src", _tree_type)
        input_trees = read_file(input_trees_path)

        preds = []
        cnt = int(len(input_sens) / args.eval_batch_size) + 1
        for _e in range(cnt):
            start_e = _e * args.eval_batch_size
            end_e = min((_e+1) * args.eval_batch_size, len(input_sens))
            if end_e == start_e: continue
            sen_input_ids = torch.LongTensor(tokenizer(input_sens[start_e: end_e], padding=True)["input_ids"]).to(device)
            sen_attention_mask = sen_input_ids.ne(pad_id).to(device)
            sen_token_type_ids = torch.zeros_like(sen_input_ids).to(device)
            sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1).to(device)
            
            syns = []
            for i_tree in input_trees[start_e:end_e]:
                tmp_syn = '<s> ' + i_tree.replace("(", "( ").replace(")", ") ") + ' </s>'
                syn = [syn_vocab2id[i] if i in syn_vocab2id else syn_vocab2id["<unk>"] for i in tmp_syn.split()]
                syns.append(syn)

            max_syn_len = max([len(i) for i in syns])
            max_syn_len = min(max_syn_len, args.max_syn_len)  
            syn_input_ids = torch.LongTensor([i[:max_syn_len] + [pad_id] * max(0, max_syn_len - len(i)) for i in syns]).to(device)
            syn_attention_mask = syn_input_ids.ne(pad_id).to(device)
            syn_token_type_ids = torch.zeros_like(syn_input_ids).to(device)
            syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1).to(device)

            batch_data = {
                'sen_input_ids': sen_input_ids,
                "sen_attention_mask": sen_attention_mask,
                "sen_token_type_ids": sen_token_type_ids,
                "sen_position_ids": sen_position_ids,
                "syn_input_ids": syn_input_ids,
                "syn_attention_mask": syn_attention_mask,
                "syn_token_type_ids": syn_token_type_ids,
                "syn_position_ids": syn_position_ids
            }
            with torch.no_grad():
                cur_output = model(
                    sem_input_ids=batch_data["sen_input_ids"],
                    sem_attention_mask=batch_data["sen_attention_mask"],
                    sem_token_type_ids=batch_data["sen_token_type_ids"],
                    sem_position_ids=batch_data["sen_position_ids"],
                    syn_input_ids=batch_data["syn_input_ids"],
                    syn_attention_mask=batch_data["syn_attention_mask"],
                    syn_token_type_ids=batch_data["syn_token_type_ids"],
                    syn_position_ids=batch_data["syn_position_ids"]
                )["output"]
            preds.extend(sigmoid(cur_output.squeeze(-1)).cpu().tolist())
        
        res.append(mean(preds))
    assert len(res) == 3
    return res


def get_pearson(X, Y):
    assert len(X) == len(Y)
    X_ = mean(X)
    Y_ = mean(Y)
    sum_up = 0
    sum_down_left = 0
    sum_down_right = 0
    for xi, yi in zip(X, Y):
        sum_up += (xi - X_) * (yi - Y_)
        sum_down_left += (xi - X_) * (xi - X_)
        sum_down_right += (yi - Y_) * (yi - Y_)
    r = sum_up / (math.sqrt(sum_down_left * sum_down_right) + 1e-10)

    return r


def cal_pearson(input_sens_path, input_random_trees_path, input_scores_path, tokenizer, model, syn_vocab2id, device):
    """
    计算当前模型在test/val的随机保存的30句上的pred和parascore值的相关系数
    """
    input_sens = read_file(input_sens_path)
    input_random_trees = []
    with open(input_random_trees_path, 'r', encoding='utf-8') as fr2:
            tmp_trees = []
            for line in fr2.readlines():
                tmp_trees.append(line.strip())
                if len(tmp_trees) == 30:
                    input_random_trees.append(tmp_trees)
                    tmp_trees = []
    fr2.close()

    input_scores = []
    with open(input_scores_path, 'r', encoding='utf-8') as fr3:
            tmp_scores = []
            for line in fr3.readlines():
                tmp_scores.append(float(line.strip()))
                if len(tmp_scores) == 30:
                    input_scores.append(tmp_scores)
                    tmp_scores = []
    fr3.close()
    assert len(input_sens) == len(input_random_trees) == len(input_scores)
    pearson_value = []
    
    pad_id = 1
    for i in range(len(input_sens)):
        cur_input_sens = [input_sens[i]] * 30
        cur_input_trees = input_random_trees[i]
        cur_input_scores = input_scores[i]
        assert len(cur_input_scores) == len(cur_input_sens) == len(cur_input_trees)

        sen_input_ids = torch.LongTensor(tokenizer(cur_input_sens, padding=True)["input_ids"]).to(device)
        sen_attention_mask = sen_input_ids.ne(pad_id).to(device)
        sen_token_type_ids = torch.zeros_like(sen_input_ids).to(device)
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1).to(device)
        
        syns = []
        for i_tree in cur_input_trees:
            tmp_syn = '<s> ' + i_tree.replace("(", "( ").replace(")", ") ") + ' </s>'
            syn = [syn_vocab2id[i] if i in syn_vocab2id else syn_vocab2id["<unk>"] for i in tmp_syn.split()]
            syns.append(syn)

        max_syn_len = max([len(i) for i in syns])
        max_syn_len = min(max_syn_len, args.max_syn_len)  
        syn_input_ids = torch.LongTensor([i[:max_syn_len] + [pad_id] * max(0, max_syn_len - len(i)) for i in syns]).to(device)
        syn_attention_mask = syn_input_ids.ne(pad_id).to(device)
        syn_token_type_ids = torch.zeros_like(syn_input_ids).to(device)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1).to(device)

        # print("****", sen_input_ids.size(), syn_input_ids.size())

        batch = {
            'sen_input_ids': sen_input_ids,
            "sen_attention_mask": sen_attention_mask,
            "sen_token_type_ids": sen_token_type_ids,
            "sen_position_ids": sen_position_ids,
            "syn_input_ids": syn_input_ids,
            "syn_attention_mask": syn_attention_mask,
            "syn_token_type_ids": syn_token_type_ids,
            "syn_position_ids": syn_position_ids
        }
        with torch.no_grad():
            output = model(
                sem_input_ids=batch["sen_input_ids"],
                sem_attention_mask=batch["sen_attention_mask"],
                sem_token_type_ids=batch["sen_token_type_ids"],
                sem_position_ids=batch["sen_position_ids"],
                syn_input_ids=batch["syn_input_ids"],
                syn_attention_mask=batch["syn_attention_mask"],
                syn_token_type_ids=batch["syn_token_type_ids"],
                syn_position_ids=batch["syn_position_ids"]
            )["output"]
        
        preds = sigmoid(output.squeeze(-1)).cpu().tolist()

        pearson_value.append(get_pearson(cur_input_scores, preds))

    return mean(pearson_value)


def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    if args.local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    syn_vocab2id = json.load(open(args.syn_vocab_path))
    model_name = args.model_name
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    print("&&&&&", args.is_continue_train)
    if not args.is_continue_train:
        model_config.bert_path = args.bert_path
        model = SimFT_ffn_Roberta(model_config, True, args.bert_path)
    else:
        model = SimFT_ffn_Roberta.from_pretrained(model_name)
        print("@@@@@", model_name, "loaded!")

    # dataset 
    train_dataset = FastFTDataset_roberta(
        src_path=args.src_path, 
        src_random_parse_path=args.src_random_parse_path,
        score_path=args.scores_path,
        tokenizer=tokenizer,
        syn_vocab2id=syn_vocab2id,
        max_sen_len=args.max_sen_len,
        max_syn_len=args.max_syn_len, 
        select_num=args.select_num
    )
    print(len(train_dataset))
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    batch_size = int(args.batch_size / args.n_gpu) if args.local_rank != -1 else args.batch_size
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  collate_fn=train_dataset.collate,
                                  num_workers=args.num_workers,
                                  worker_init_fn=worker_init_fn
                                  )

    total_steps = int((len(train_dataset) / args.batch_size / args.gradient_accumulation_steps + 1) * args.n_epochs)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warm_up_steps, 
                                                num_training_steps=total_steps)

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        model.cuda()
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True, 
                    broadcast_buffers=False)
    elif args.n_gpu > 1:
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    print("model loaded!")
    

    if args.local_rank in [-1, 0]:
        logging.info("***** Running training *****")
        logging.info(f"  lr: {args.lr}, warm_up_steps: {args.warm_up_steps}")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {args.n_epochs}")
        logging.info(f"  Instantaneous batch size per device = {int(args.batch_size / args.n_gpu)}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.batch_size * args.gradient_accumulation_steps}")
        logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logging.info(f"  Optimization steps in each epoch = {int(len(train_dataset) / args.batch_size / args.gradient_accumulation_steps + 1)}")
        logging.info(f"  Total optimization steps = {total_steps}")

    iter_num = 0
    last_out_iter_num = 0
    for i_epoch in range(args.n_epochs):
        model.train()
        out_loss_sort = []
        out_loss_mse = []
        out_src_preds = []
        out_ref_preds = []
        out_src_scores = []
        out_ref_scores = []
        out_rate = []

        for train_step, batch in enumerate(train_dataloader):
            if i_epoch < args.skip_epochs: 
                optimizer.step()
                scheduler.step()
                iter_num += 1
                continue
            for key in batch.keys():
                # print(key, batch[key].size())
                batch[key] = batch[key].to(model.device)
                # print(key, batch[key][0])
            output = model(
                sem_input_ids=batch["sen_input_ids"],
                sem_attention_mask=batch["sen_attention_mask"],
                sem_token_type_ids=batch["sen_token_type_ids"],
                sem_position_ids=batch["sen_position_ids"],
                syn_input_ids=batch["syn_input_ids"],
                syn_attention_mask=batch["syn_attention_mask"],
                syn_token_type_ids=batch["syn_token_type_ids"],
                syn_position_ids=batch["syn_position_ids"]
            )
            # print("@@@@@@", output.device)
            # print("@@@@@-output", output["output"].size())
            # print(batch['aesop_src_inputs'][:2])
            
            loss_output, sort_loss, mse_loss, src_scores, ref_scores, src_preds, ref_preds, rate = get_loss_mse_sort_v2(args, output["output"], batch["scores"], model.device)
            # print(f"loss {loss_output}")
            # loss = loss_output["loss"] / args.gradient_accumulation_steps
            loss = loss_output / args.gradient_accumulation_steps
            loss.backward()

            out_loss_sort.append((sort_loss / args.gradient_accumulation_steps).item())
            out_loss_mse.append((mse_loss / args.gradient_accumulation_steps).item())

            out_src_preds.append(src_preds.item())
            out_ref_preds.append(ref_preds.item())
            out_src_scores.append(src_scores.item())
            out_ref_scores.append(ref_scores.item())
            out_rate.append(rate.item())


            if (train_step+1) % args.gradient_accumulation_steps == 0 and train_step != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                iter_num += 1
                if iter_num % args.log_interval == 0 and iter_num != last_out_iter_num and args.local_rank in [-1, 0]:
                    logging.info(f"epoch: {i_epoch+1}, step: {iter_num}, lr: {optimizer.param_groups[-1]['lr']:.7f}, sort_loss: {mean(out_loss_sort):.4f}, mse_loss: {mean(out_loss_mse):.4f}, src_scores: {mean(out_src_scores):.4f}, ref_scores: {mean(out_ref_scores):.4f}, src_preds: {mean(out_src_preds):.4f}, ref_preds: {mean(out_ref_preds):.4f}, rate: {mean(out_rate):.6f}")
                    last_out_iter_num = iter_num
                    out_loss_sort = []
                    out_loss_ce = []

                if args.local_rank in [-1, 0] and iter_num % args.save_interval == 0 and iter_num != 0:
                    output_dir = args.saved_model_path + '/epoch' + str(i_epoch + 1) + "_iter" + str(iter_num) + "/"
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_dir, CONFIG_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_dir)
                    logging.info(f"{output_dir} ->model saved {args.local_rank}")

                    model.eval()
                    eval_res_val = eavluate(args, tokenizer, model, syn_vocab2id, model.device, args.val_path, args.val_parse_path)
                    mean_pearson_val = cal_pearson(args.val_path, args.val_random_trees_path, args.val_random_scores_path, tokenizer, model, syn_vocab2id, model.device)
                    logging.info(f"{output_dir} -> eavluate in val set, results: src: {eval_res_val[0]:.6f}, ref: {eval_res_val[1]:.6f}, tgt: {eval_res_val[2]:.6f}, mean_pearson_val: {mean_pearson_val:.6f}")
                    eval_res_test = eavluate(args, tokenizer, model, syn_vocab2id, model.device, args.test_path, args.test_parse_path)
                    mean_pearson_test = cal_pearson(args.test_path, args.test_random_trees_path, args.test_random_scores_path, tokenizer, model, syn_vocab2id, model.device)
                    logging.info(f"{output_dir} -> eavluate in test set, results: src: {eval_res_test[0]:.6f}, ref: {eval_res_test[1]:.6f}, tgt: {eval_res_test[2]:.6f}, mean_pearson_test: {mean_pearson_test:.6f}")
                    model.train()

        if i_epoch < args.skip_epochs: 
            if args.local_rank in [-1, 0]:
                logging.info(f"epoch-{i_epoch+1} has skipped! ")
            continue

        if args.local_rank in [-1, 0]:
            output_dir = args.saved_model_path + '/epoch' + str(i_epoch + 1) + "/"
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(output_dir)
            logging.info(f'{i_epoch+1} epoch training finished，{output_dir} ->model saved')
            
            model.eval()
            eval_res_val = eavluate(args, tokenizer, model, syn_vocab2id, model.device, args.val_path, args.val_parse_path)
            mean_pearson_val = cal_pearson(args.val_path, args.val_random_trees_path, args.val_random_scores_path, tokenizer, model, syn_vocab2id, model.device)
            logging.info(f"{output_dir} -> eavluate in val set, results: src: {eval_res_val[0]:.6f}, ref: {eval_res_val[1]:.6f}, tgt: {eval_res_val[2]:.6f}, mean_pearson_val: {mean_pearson_val:.6f}")
            eval_res_test = eavluate(args, tokenizer, model, syn_vocab2id, model.device, args.test_path, args.test_parse_path)
            mean_pearson_test = cal_pearson(args.test_path, args.test_random_trees_path, args.test_random_scores_path, tokenizer, model, syn_vocab2id, model.device)
            logging.info(f"{output_dir} -> eavluate in test set, results: src: {eval_res_test[0]:.6f}, ref: {eval_res_test[1]:.6f}, tgt: {eval_res_test[2]:.6f}, mean_pearson_test: {mean_pearson_test:.6f}")
            model.train()
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default=' /facebook-bart-base', type=str)
    parser.add_argument("--bert_path", default='/bert-base-uncased', type=str)
    
    parser.add_argument("--src_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--src_random_parse_path", default="", type=str)
    parser.add_argument("--scores_path", default="", type=str)

    parser.add_argument("--val_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--val_parse_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--val_random_trees_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--val_random_scores_path", default=' /data-QQPPos/train/src.txt', type=str)


    parser.add_argument("--test_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--test_parse_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--test_random_trees_path", default=' /data-QQPPos/train/src.txt', type=str)
    parser.add_argument("--test_random_scores_path", default=' /data-QQPPos/train/src.txt', type=str)
    
    parser.add_argument("--saved_model_path", default='./save-model-test', type=str)
    parser.add_argument("--syn_vocab_path", default=' /data-QQPPos/syns2id.json', type=str)

    parser.add_argument("--is_continue_train", action="store_true")
    parser.add_argument("--skip_epochs", default=0, type=int)

    parser.add_argument("--select_num", default=10, type=int)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=400, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--thres_ce", default=1, type=float)
    parser.add_argument("--margin", default=0.05, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warm_up_steps", default=500, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--save_interval", default=5000, type=int)

    parser.add_argument("--max_sen_len", default=64, type=int)
    parser.add_argument("--max_syn_len", default=192, type=int)

    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--n_gpu", default=torch.cuda.device_count() if torch.cuda.is_available() else 0, type=int)

    args = parser.parse_args()

    if args.local_rank in [-1, 0]:
        print(args)

    main(args)