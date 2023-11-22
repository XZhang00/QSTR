import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
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
from parascorer_new import ParaScorer
from numpy import mean
from utils import run_multi_bleu


def cos_sim(inputs, ref):
    
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
    
    parser.add_argument('--gen_pg', type=str, default='../QQPPos-diversity-results/AESOP-siscp-diverse-0.2-pgs.txt')

    args = parser.parse_args()

    print(args.gen_pg)
    scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)

    srcs = []
    with open(args.src, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            srcs.append(line.strip())
    fr.close()
    refs = []
    with open(args.ref, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            refs.append(line.strip())
    fr.close()

    all_10_pgs = []
    for i in range(10):
        all_10_pgs.append([])
    # print("+++", len(all_10_pgs))
    with open(args.gen_pg, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            all_10_pgs[i % 10].append(line.strip())
    fr.close()
    for i in all_10_pgs:
        print(len(i))
    
    s1 = time.time()
    cnt = 1
    for cur_pgs in all_10_pgs:
        # tmp_dir = "./tmp-eval/"
        # if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
        # tmp_name = tmp_dir + "tmp-" + str(cnt) + ".txt"
        # fw = open(tmp_name, 'w', encoding='utf-8')
        # for i_pg in cur_pgs:
        #     fw.write(i_pg + '\n')
        # fw.flush()
        # fw.close()
        # src_bleu = run_multi_bleu(tmp_name, args.src)
        # ref_bleu = run_multi_bleu(tmp_name, args.ref)

        cos_sim_src = cos_sim(cur_pgs, srcs)
        cos_sim_ref = cos_sim(cur_pgs, refs)
        
        score_free = scorer.free_score(cur_pgs, srcs, 0.4)
        # score_ref = scorer.base_score(cur_pgs, srcs, refs, 0.15)
        assert len(score_free) == len(srcs) 

        # a = 0.8
        # ibleu = a * ref_bleu - ((1-a) * src_bleu)

        # print("sort-" + str(cnt), src_bleu, ref_bleu, ibleu, '/', '/', '/', cos_sim_src, cos_sim_ref, mean([cos_sim_src, cos_sim_ref]), mean(score_free), mean(score_ref))
        
        cnt += 1

        print(mean([cos_sim_src, cos_sim_ref]), mean(score_free))
    
    e1 = time.time()
    print("cost:", e1-s1, "s")
    print(args.gen_pg)


