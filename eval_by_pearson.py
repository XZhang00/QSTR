import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import BertTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from numpy import mean
import time, json
from SimFT_model import SimFT_ffn
device = torch.device("cuda")
from tqdm import tqdm
from torch import nn
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = nn.Sigmoid()
import math


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='../outputs-QQPPos/SimFT_ffn-mse+sort/epoch1_iter20', type=str)
    
    parser.add_argument("--input", default='../data-QQPPos/test-pgs-all.json', type=str)
    
    parser.add_argument("--syn_vocab_path", default='../data-QQPPos/syn2id.json', type=str)

    parser.add_argument("--select_num", default=2000, type=int)
    parser.add_argument("--max_syn_len", default=192, type=int)

    args = parser.parse_args()

    syn_vocab2id = json.load(open(args.syn_vocab_path))

    # path = "../outputs-QQPPos/sim_ffn_softmax-mse+sort2"
    # for i_name in os.listdir(path=path):
    #     # print(i_name)
    #     if "epoch" not in i_name or 'iter' in i_name: continue
    #     model_name = path + '/' + i_name
    
    model_name = args.model_name
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = SimFT_ffn.from_pretrained(model_name).to(device)
    model_config = AutoConfig.from_pretrained(model_name)
    model.eval()

    input_ori_dict = json.load(open(args.input))

    pearson_value = []

    for i in tqdm(range(len(input_ori_dict))):
        input_sens = []
        input_trees = []
        input_scores = []
        preds = []
        for _ls in input_ori_dict[str(i)][:30]:
            input_sens.append(_ls[0])
            input_trees.append(_ls[1])
            input_scores.append(_ls[3])

        sen_input_ids = torch.LongTensor(tokenizer(input_sens, padding=True)["input_ids"]).to(device)
        sen_attention_mask = sen_input_ids.ne(0).to(device)
        sen_token_type_ids = torch.zeros_like(sen_input_ids).to(device)
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1).to(device)
        
        syns = []
        cur_aesop_inputs = []
        for i_tree in input_trees:
            tmp_syn = '<s> ' + i_tree.replace("(", "( ").replace(")", ") ") + ' </s>'
            syn = [syn_vocab2id[i] if i in syn_vocab2id else syn_vocab2id["<unk>"] for i in tmp_syn.split()]
            syns.append(syn)

        max_syn_len = max([len(i) for i in syns])
        max_syn_len = min(max_syn_len, args.max_syn_len)  
        syn_input_ids = torch.LongTensor([i[:max_syn_len] + [0] * max(0, max_syn_len - len(i)) for i in syns]).to(device)
        syn_attention_mask = syn_input_ids.ne(0).to(device)
        syn_token_type_ids = torch.zeros_like(syn_input_ids).to(device)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1).to(device)

        batch = {
            'sen_input_ids': sen_input_ids,
            "sen_attention_mask": sen_attention_mask,
            "sen_token_type_ids": sen_token_type_ids,
            "sen_position_ids": sen_position_ids,
            "syn_input_ids": syn_input_ids,
            "syn_attention_mask": syn_attention_mask,
            "syn_token_type_ids": syn_token_type_ids,
            "syn_position_ids": syn_position_ids,
            "src_sens": input_sens,
            "aesop_src_inputs": cur_aesop_inputs
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
        
        preds.extend(sigmoid(output.squeeze(-1)).cpu().tolist())

        pearson_value.append(get_pearson(input_scores, preds))


    print("/".join(model_name.split("/")[-2:]), mean(pearson_value), max(pearson_value), min(pearson_value), len(pearson_value))

    # fw = open("../outputs-QQPPos/Pearson-output/" + "&".join(model_name.split("/")[-2:]) + "-pearson-res2.json", 'w', encoding='utf-8')
    # fw.write(json.dumps(pearson_value, ensure_ascii=False))
    # fw.close()    