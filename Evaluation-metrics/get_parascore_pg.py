from parascorer_new import ParaScorer
import argparse
import time
from numpy import mean
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
device = torch.device('cuda')


parser = argparse.ArgumentParser()
parser.add_argument('--pg_file', '-pg', type=str, default='../QQP-Pos-test-results/AESOP-level5-ref_sep_extract-pg.txt')
parser.add_argument('--src_file', '-s', type=str, default='../QQP-Pos-test-results/src.txt')
parser.add_argument('--ref_file', '-r', type=str, default='../QQP-Pos-test-results/ref.txt')
parser.add_argument('--tgt_file', '-t', type=str, default='../QQP-Pos-test-results/tgt.txt')
args = parser.parse_args()


scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)

srcs = []
with open(args.src_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        srcs.append(line.strip())
fr.close()

refs = []
with open(args.ref_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        refs.append(line.strip())
fr.close()

pgs = []
with open(args.pg_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        pgs.append(line.strip())
fr.close()

tmp_free_score = scorer.free_score(refs, srcs, 0.4)
# score_ref015 = scorer.base_score(pgs, srcs, refs, 0.15)


# print(mean(tmp_free_score), mean(score_ref015), mean(score_ref))
# print(mean(tmp_free_score), mean(score_ref015))
print(mean(tmp_free_score))



