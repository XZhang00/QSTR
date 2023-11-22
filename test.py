import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from data_utils import FTDataset
from transformers import BertTokenizer, AutoConfig, WEIGHTS_NAME, CONFIG_NAME, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.optim import Adam, AdamW
from numpy import *
import logging
import numpy as np
import time, json
from parascore import ParaScorer
from SimFT_model import SimFT_ffn
device = torch.device("cuda")
scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)
from tqdm import tqdm
from torch import nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = nn.Sigmoid()

def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())

    return res


def main(args, batch, model, aesop_model, aesop_tokenizer):
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

        # print(len(batch['aesop_src_inputs']), args.aesop_batch_size)
        # cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, batch['aesop_src_inputs'], args.aesop_batch_size)
        return sigmoid(output.squeeze(-1)).cpu().tolist() # , cur_pgs
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='../outputs-QQPPos/SimFT_ffn-mse+sort/epoch1_iter20', type=str)
    parser.add_argument("--aesop_model_name", default='../qqppos-h4-d', type=str)
    parser.add_argument("--aesop_batch_size", default=400, type=int)
    
    parser.add_argument("--input_path", default='../data-QQPPos/test/src.txt', type=str)
    parser.add_argument("--input_trees_path", default='../data-QQPPos/test/src.tree_h5', type=str)
    
    parser.add_argument("--syn_vocab_path", default='../data-QQPPos/syn2id.json', type=str)

    parser.add_argument("--batch_size", default=300, type=int)
    parser.add_argument("--max_sen_len", default=64, type=int)
    parser.add_argument("--max_syn_len", default=192, type=int)

    args = parser.parse_args()

    # path = "../outputs-QQPPos/sim-mse"
    # for i_name in os.listdir(path=path):
    #     # print(i_name)
    #     if "epoch" not in i_name: continue
    #     model_name = path + '/' + i_name
    #     # print(model_name)
    model_name = args.model_name
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = SimFT_ffn.from_pretrained(model_name).to(device)
    # print(model)
    model_config = AutoConfig.from_pretrained(model_name)
    model.eval()

    aesop_model = AutoModelForSeq2SeqLM.from_pretrained(args.aesop_model_name).to(device).half()
    aesop_model.eval()
    aesop_tokenizer = AutoTokenizer.from_pretrained(args.aesop_model_name)
    syn_vocab2id = json.load(open(args.syn_vocab_path))

    input_sens = read_file(args.input_path)
    for _tree_type in ["src", "ref", "tgt"]:
        input_trees_path = args.input_trees_path.replace("src", _tree_type)
        input_trees = read_file(input_trees_path)
        # input_sens = [input_sens[2865]]
        # input_trees = [input_trees[2865]]
        # aesop_input = [aesop_input[2865]]
        # print(input_sens)

        preds = []
        pgs = []
        parascores = []

        cnt = int(len(input_sens) / args.batch_size) + 1
        for _e in range(cnt):
            start_e = _e * args.batch_size
            end_e = min((_e+1) * args.batch_size, len(input_sens))
            if end_e == start_e: continue
            sen_input_ids = torch.LongTensor(tokenizer(input_sens[start_e: end_e], padding=True)["input_ids"]).to(device)
            # print(sen_input_ids.size())
            sen_attention_mask = sen_input_ids.ne(0).to(device)
            sen_token_type_ids = torch.zeros_like(sen_input_ids).to(device)
            sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1).to(device)
            
            syns = []
            for i_tree in input_trees[start_e:end_e]:
                tmp_syn = '<s> ' + i_tree.replace("(", "( ").replace(")", ") ") + ' </s>'
                syn = [syn_vocab2id[i] if i in syn_vocab2id else syn_vocab2id["<unk>"] for i in tmp_syn.split()]
                syns.append(syn)

            max_syn_len = max([len(i) for i in syns])
            max_syn_len = min(max_syn_len, args.max_syn_len) 
            # max_syn_len = args.max_syn_len
            syn_input_ids = torch.LongTensor([i[:max_syn_len] + [0] * max(0, max_syn_len - len(i)) for i in syns]).to(device)
            syn_attention_mask = syn_input_ids.ne(0).to(device)
            syn_token_type_ids = torch.zeros_like(syn_input_ids).to(device)
            syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1).to(device)
            # print(syn_token_type_ids.size(), syn_attention_mask.size(), syn_position_ids.size())

            batch_data = {
                'sen_input_ids': sen_input_ids,
                "sen_attention_mask": sen_attention_mask,
                "sen_token_type_ids": sen_token_type_ids,
                "sen_position_ids": sen_position_ids,
                "syn_input_ids": syn_input_ids,
                "syn_attention_mask": syn_attention_mask,
                "syn_token_type_ids": syn_token_type_ids,
                "syn_position_ids": syn_position_ids,
                "src_sens": input_sens[start_e: end_e]
            }
            cur_pred = main(args, batch_data, model, aesop_model, aesop_tokenizer)
            preds.extend(cur_pred)
            # pgs.extend(cur_pgs)
            # print(cur_pred[:5])

            # cur_parascore = scorer.free_score(cur_pgs, input_sens[start_e: end_e])
            # print(cur_parascore[:5])
            # parascores.extend(cur_parascore)

        # print(len(preds), len(pgs))
        # print(i_name, input_trees_path.split("/")[-1], mean(parascores), mean(preds))
        # print(i_name, input_trees_path.split("/")[-1], mean(preds))
        print(model_name, input_trees_path.split("/")[-1], mean(preds), preds[111], input_sens[111], input_trees[111])



