import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from transformers import BertTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer
from numpy import mean
import time, json
from SimFT_model import SimFT_ffn, SimFT_ffn_Roberta
device = torch.device("cuda")
from tqdm import tqdm
from torch import nn
import random
import matplotlib.pyplot as plt 
# from sentence_transformers import SentenceTransformer, util
# sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = nn.Sigmoid()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='../outputs-QQPPos-fast-ref015-roberta/sort_batch+mse-lr_3e-5/epoch10', type=str)
    parser.add_argument("--output_path", default='../retrieve-res_for_next_train', type=str)

   
    parser.add_argument("--all_parse_path", default='../data-QQPPos/candidate_trees_h5_sort_by_freq.json', type=str)
    parser.add_argument("--syn_vocab_path", default='../data-QQPPos/syn2id-roberta.json', type=str)

    parser.add_argument("--batch_size", default=400, type=int)
    parser.add_argument("--max_syn_len", default=192, type=int)
    parser.add_argument("--pad_id", default=0, type=int)


    args = parser.parse_args()

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

    syn_vocab2id = json.load(open(args.syn_vocab_path))

    num = 4000
    candidate_trees = json.load(open(args.all_parse_path))[:num]
    print(len(candidate_trees))

    st = time.time()
    save_syn_embed = []
    save_syn_mask = []
    cnt = int(len(candidate_trees) / args.batch_size) + 1
    for _e in tqdm(range(cnt)):
        start_e = _e * args.batch_size
        end_e = min((_e+1) * args.batch_size, len(candidate_trees))
        if end_e == start_e: continue

        syns = []
        for i_tree in candidate_trees[start_e:end_e]:
            tmp_syn = '<s> ' + i_tree.replace("(", "( ").replace(")", ") ") + ' </s>'
            syn = [syn_vocab2id[i] if i in syn_vocab2id else syn_vocab2id["<unk>"] for i in tmp_syn.split()]
            syns.append(syn)
            # aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")

        max_syn_len = args.max_syn_len
        syn_input_ids = torch.LongTensor([i[:max_syn_len] + [args.pad_id] * max(0, max_syn_len - len(i)) for i in syns]).to(device)
        syn_attention_mask = syn_input_ids.ne(args.pad_id).to(device)
        syn_token_type_ids = torch.zeros_like(syn_input_ids).to(device)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1).to(device)

        save_syn_mask.append(syn_attention_mask.cpu())
        with torch.no_grad():
            syn_encoder_output = model.encoder_syn(
                input_ids=syn_input_ids,
                attention_mask=syn_attention_mask,
                position_ids=syn_position_ids,
                token_type_ids=syn_token_type_ids,
                return_dict=True
            )["last_hidden_state"]
           
            syn1 = model.ffn_syn(syn_encoder_output)
            save_syn_embed.append(syn1.cpu())

    z_save_syn_embeds = torch.cat(save_syn_embed, dim=0)
    print(z_save_syn_embeds.size())
    embed_path = args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-syn_embeds-" + str(num) + ".pt"
    torch.save(z_save_syn_embeds, embed_path) 

    z_save_syn_mask = torch.cat(save_syn_mask, dim=0)
    print(z_save_syn_mask.size())
    mask_path = args.output_path + '/' + "-".join(args.model_name.split("/")[-2:]) + "-syn_mask-" + str(num) + ".pt"
    torch.save(z_save_syn_mask, mask_path)

    et = time.time()
    print(f"cost: {et-st} s")