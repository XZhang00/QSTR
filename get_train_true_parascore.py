from parascorer_new import ParaScorer
import argparse
import time
from numpy import mean
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
device = torch.device('cuda')


parser = argparse.ArgumentParser()
parser.add_argument('--src_file', '-s', type=str, default='../data-QQPPos/level5/train.source')
parser.add_argument('--ref_file', '-r', type=str, default='../data-QQPPos/level5/train.target')
parser.add_argument('--output_path', '-t', type=str, default='../data-QQPPos/level5/train.score-true-ref015')
args = parser.parse_args()


scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)

srcs = []
with open(args.src_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        srcs.append(line.strip().split(" <sep> ")[0])
fr.close()
print(srcs[:2])

refs = []
with open(args.ref_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        refs.append(line.strip().split(" <sep> ")[1])
fr.close()
print(refs[:2])


# tmp_free_score = scorer.free_score(pgs, srcs, 0.35)
score_ref015 = scorer.base_score(refs, srcs, refs, 0.15)

# score_ref = scorer.base_score(pgs, srcs, refs, 0.10)
# print(mean(tmp_free_score), mean(score_ref015), mean(score_ref))
print(mean(score_ref015))

assert len(srcs) == len(refs) == len(score_ref015)

preds = []
fw = open(args.output_path, 'w', encoding='utf-8')

for _pred in score_ref015:
    fw.write(str(_pred) + '\n')
fw.flush()




