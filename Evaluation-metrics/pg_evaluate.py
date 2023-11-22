import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# os.environ["USE_TF"] = 'None'
import torch
device = torch.device('cuda')
from sacrebleu.metrics import BLEU
bleu = BLEU(tokenize='intl')
import re
from sentence_transformers import SentenceTransformer, util
sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
# sim_model.encode(s, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
import subprocess
import argparse
import time
from numpy import mean
from utils import cal_tree_dis, stanford_parsetree_extractor, run_multi_bleu


def cos_sim(input_file, ref_file):
    inputs = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
           inputs.append(line.strip())
    fr.close() 
    ref = []
    with open(ref_file, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
           ref.append(line.strip())
    fr.close()
    assert len(inputs) == len(ref)

    inputs_embeds = sim_model.encode(inputs, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)
    ref_embeds = sim_model.encode(ref, batch_size=128, convert_to_tensor=True, normalize_embeddings=True)

    res = torch.sum(inputs_embeds * ref_embeds, dim=1)

    return torch.mean(res).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate results")
    # parser.add_argument('--', type=str, default=None)
    parser.add_argument('--src', type=str, default='../QQP-Pos-test-results/src.txt')
    parser.add_argument('--ref', type=str, default='../QQP-Pos-test-results/ref.txt')
    parser.add_argument('--tgt', type=str, default='../QQP-Pos-test-results/tgt.txt')
    parser.add_argument('--gen_pg', type=str, default='../output-qqppos-new/ref015-score-0.1/AESOP-tgt-checkpoint-13400-pg.txt')
    parser.add_argument('--ibleu_a', type=float, default=0.8)
    parser.add_argument('--max_parse_depth', type=int, default=1000)
    args = parser.parse_args()

    src_bleu, ref_bleu, ibleu, ted_src, ted_tgt, ted_ref, cos_sim_src, cos_sim_ref = ['/'] * 8

    s1 = time.time()

    src_bleu = run_multi_bleu(args.gen_pg, args.src)
    ref_bleu = run_multi_bleu(args.gen_pg, args.ref)
    a = args.ibleu_a
    ibleu = a * ref_bleu - ((1-a) * src_bleu)
    cos_sim_src = cos_sim(args.gen_pg, args.src)
    cos_sim_ref = cos_sim(args.gen_pg, args.ref)
    cos_sim_src_ref = cos_sim(args.src, args.ref)

    # path = '../QQP-Pos-test-results'
    # spe = stanford_parsetree_extractor()
    # tmp_ll = os.listdir(path)
    # tmp_ll.sort()
    # for _name in tmp_ll:
    #     if 'SI-SCP' in _name or 'SGCP' in _name: continue
    #     # print(_name)
    #     args.gen_pg = path + '/' + _name
    #     ted_src, ted_tgt, ted_ref = '/', '/', '/'
    #     if 'pg' in args.gen_pg.split("/")[-1]: 

    spe = stanford_parsetree_extractor()
    if 'src' in args.gen_pg.split("/")[-1]:
        ted_src = cal_tree_dis(spe, args.gen_pg, args.src, args.max_parse_depth)
    elif 'tgt' in args.gen_pg.split("/")[-1]:
        ted_tgt = cal_tree_dis(spe, args.gen_pg, args.tgt, args.max_parse_depth)
        ted_ref = cal_tree_dis(spe, args.gen_pg, args.ref, args.max_parse_depth)
    elif 'ref' in args.gen_pg.split("/")[-1]:
        ted_ref = cal_tree_dis(spe, args.gen_pg, args.ref, args.max_parse_depth)
    else:
        ted_src = cal_tree_dis(spe, args.gen_pg, args.src, args.max_parse_depth)
        ted_ref = cal_tree_dis(spe, args.gen_pg, args.ref, args.max_parse_depth)
                    
            # print(args.gen_pg.split("/")[-1])
            # print("******", ted_src, ted_tgt, ted_ref)

    print(args.gen_pg)
    print('src_bleu', 'ref_bleu', 'iBLEU', 'TED-src', 'TED-tgt', 'TED-ref', 'cos-sim-src', 'cos-sim-ref')
    print(src_bleu, ref_bleu, ibleu, ted_src, ted_tgt, ted_ref, cos_sim_src, cos_sim_ref)
    
    e1 = time.time()
    print("cost:", e1-s1, "s")


