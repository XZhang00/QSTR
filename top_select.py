import argparse
from transformers import BertTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer
from numpy import mean
import time, json
from SimFT_model import SimFT_ffn, SimFT_ffn_Roberta
from tqdm import tqdm
from data_utils import normalize_tree, tree_edit_distance




def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='../outputs-QQPPos-fast-ref015/ref015-SimFT_ffn-sort_batch+mse-lr_2e-5/epoch10', type=str)
    parser.add_argument("--output_path", default='../retrieve-results-fast-ref015', type=str)

    
    # parser.add_argument("--aesop_model_name", default='../qqppos-h4-d', type=str)
    # parser.add_argument("--aesop_batch_size", default=300, type=int)
    
    parser.add_argument("--src_path", default='../data-QQPPos/data-QQPPos/test/src.txt', type=str)

    parser.add_argument("--input_aesop_src", default='../data-QQPPos/test/aesop-level5-src.source', type=str)
    parser.add_argument("--input_aesop_ref", default='../data-QQPPos/test/aesop-level5-ref.source', type=str)

    
    parser.add_argument("--all_parse_path", default='../data-QQPPos/candidate_trees_h5.json', type=str)
    parser.add_argument("--syn_vocab_path", default='../data-QQPPos/syn2id.json', type=str)

    parser.add_argument("--batch_size", default=5000, type=int)
    parser.add_argument("--max_sen_len", default=64, type=int)
    parser.add_argument("--max_syn_len", default=192, type=int)
    parser.add_argument("--per_save_num", default=1000, type=int)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--per_device_num", default=375, type=int)
    parser.add_argument("--pad_id", default=0, type=int)




    args = parser.parse_args()

    syn_vocab2id = json.load(open(args.syn_vocab_path))

    candidate_trees = json.load(open(args.all_parse_path))
    
    print(len(candidate_trees))
    
    number = args.device_id
    s_num = number * args.per_device_num
    e_num = ( number + 1) * args.per_device_num
    aesop_input_srcs = read_file(args.input_aesop_src)[s_num:e_num]
    aesop_input_refs = read_file(args.input_aesop_ref)[s_num:e_num]
    
    device_id = str(number)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    import torch
    device = torch.device("cuda")
    from torch import nn
    sigmoid = nn.Sigmoid()

    model_name = args.model_name
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model_config = AutoConfig.from_pretrained(model_name)
        model = SimFT_ffn_Roberta.from_pretrained(model_name).to(device)
        args.pad_id = 1
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model_config = AutoConfig.from_pretrained(model_name)
        model = SimFT_ffn.from_pretrained(model_name).to(device)
        args.pad_id = 0
    model.eval()


    syn_embeds = torch.load(args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-syn_embeds.pt")
    syn_mask = torch.load(args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-syn_mask.pt").to(device)
    print("syn loaded!")
    fw = open(args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-parse"+ device_id +".txt", 'w', encoding='utf-8')
    fw_s = open(args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-score"+ device_id +".txt", 'w', encoding='utf-8')


    for i_src, i_ref in zip(aesop_input_srcs, aesop_input_refs):
        src_sen, _, _ = i_src.split(" <sep> ")
        _, _, ref_parse = i_ref.split(" <sep> ")
        print(src_sen)
        sen_input_ids = torch.LongTensor(tokenizer([src_sen], padding=True)["input_ids"]).to(device)
        sem_attention_mask = sen_input_ids.ne(args.pad_id).to(device)
        sen_token_type_ids = torch.zeros_like(sen_input_ids).to(device)
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1).to(device)
        
        preds = []
        with torch.no_grad():
            # st = time.time()
            sem_encoder_output = model.encoder_sem(
                input_ids=sen_input_ids,
                attention_mask=sem_attention_mask,
                position_ids=sen_position_ids,
                token_type_ids=sen_token_type_ids,
                return_dict=True
            )["last_hidden_state"]

            # # 1. 在cpu上计算，每个句子需要17s
            # sem1 = model.ffn_sem(sem_encoder_output).repeat_interleave(len(candidate_trees), dim=0).cpu()
            # sem_attention_mask = sen_attention_mask.repeat_interleave(len(candidate_trees), dim=0).cpu()
            # print(sem1.size(), sem_attention_mask.size())
            
            # sim_matrix = torch.matmul(sem1, syn1.transpose(-1, -2))
            # sem_embed = ((sem1 * torch.max(sim_matrix, dim=-1, keepdim=True).values) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
            # syn_embed = ((syn1 * torch.max(sim_matrix.transpose(-1, -2), dim=-1, keepdim=True).values) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)

            # output = model.linear(torch.cat((sem_embed.to(device), syn_embed.to(device)), dim=-1))
            # z_output = sigmoid(output.squeeze(-1)).cpu().tolist()
            # print(len(z_output))
            # et = time.time()
            # print(f"cost: {et-st} s")

            # 在gpu上计算 5000-4.3s; 6000-4.28s; 7000-3.9s; 
            sem_ffn = model.ffn_sem(sem_encoder_output)
            cnt = int(len(candidate_trees) / args.batch_size) + 1
            for _e in range(cnt):
                start_e = _e * args.batch_size
                end_e = min((_e+1) * args.batch_size, len(candidate_trees))
                if end_e == start_e: continue
                syn_ffn = syn_embeds[start_e: end_e].to(device)
                syn_attention_mask = syn_mask[start_e: end_e]

                sim_matrix = torch.matmul(sem_ffn, syn_ffn.transpose(-1, -2))  # bsz*sen-len*syn-len  语义维度会自动广播

                sem_sim_mask = (1.0 - sem_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min
                syn_sim_mask = (1.0 - syn_attention_mask.int()) * torch.finfo(sim_matrix.dtype).min

                sem_syn_sim_score = sim_matrix + syn_sim_mask.unsqueeze(1)
                syn_sem_sim_score = sim_matrix.transpose(-1, -2) + sem_sim_mask.unsqueeze(1)

                sem_embed = ((sem_ffn * torch.max(sem_syn_sim_score, dim=-1, keepdim=True).values) * sem_attention_mask.unsqueeze(-1)).sum(1) / sem_attention_mask.unsqueeze(-1).sum(1)
                syn_embed = ((syn_ffn * torch.max(syn_sem_sim_score, dim=-1, keepdim=True).values) * syn_attention_mask.unsqueeze(-1)).sum(1) / syn_attention_mask.unsqueeze(-1).sum(1)

                output = model.linear(torch.cat((sem_embed, syn_embed), dim=-1))
                preds.extend(sigmoid(output.squeeze(-1)).cpu().tolist())

        # print(len(preds))
        # et = time.time()
        # print(f"cost: {et-st} s")
            
        preds_item = [[i, j] for i, j in enumerate(preds)]
        preds_sort_item = sorted(preds_item, key= lambda x:x[1], reverse=True)
        for j in preds_sort_item[:args.per_save_num]:
            # fw.write(aesop_inputs[j[0]] + '\n')
            # fw.flush()
            cur_ted_4 = tree_edit_distance(normalize_tree(ref_parse, 4), normalize_tree(candidate_trees[j[0]], 4))
            cur_ted_3 = tree_edit_distance(normalize_tree(ref_parse, 3), normalize_tree(candidate_trees[j[0]], 3))
            cur_ted_2 = tree_edit_distance(normalize_tree(ref_parse, 2), normalize_tree(candidate_trees[j[0]], 2))

            fw.write(f"{candidate_trees[j[0]]}\n")
            fw_s.write(f"pred: {j[1]:.8f}\tcur_ted_4: {cur_ted_4:.4f}\tcur_ted_3: {cur_ted_3:.4f}\tcur_ted_2: {cur_ted_2:.4f}\n")
            fw.flush()
            fw_s.flush()
    fw.close()
    fw_s.close()



# nohup python -u top_select.py >> log/select.log0 2>&1 &


