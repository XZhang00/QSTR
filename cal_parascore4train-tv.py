from parascorer_new import ParaScorer
import argparse
import time
from numpy import mean
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
device = torch.device('cuda')


def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())
    fr.close()
    return res



parser = argparse.ArgumentParser()
parser.add_argument('--src_file', '-s', type=str, default='../data-QQPPos/test/src.txt')
parser.add_argument('--ref_file', '-r', type=str, default='../data-QQPPos/test/ref.txt')
parser.add_argument('--pg_file', '-p', type=str, default='../data-QQPPos/test/save4test-1W/src_random.pg')

parser.add_argument('--output_path', '-t', type=str, default='../data-QQPPos/test/save4test-1W/src_score.ref015')
args = parser.parse_args()


scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)

num = 30

srcs = []
with open(args.src_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        for i in range(num):
            srcs.append(line.strip())
fr.close()

refs = []
with open(args.ref_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        for i in range(num):
            refs.append(line.strip())
fr.close()

pgs = read_file(args.pg_file)

assert len(srcs) == len(refs) == len(pgs)

score_ref015 = scorer.base_score(pgs, srcs, refs, 0.15)

print(mean(score_ref015))

assert len(srcs) == len(refs) == len(score_ref015)

fw = open(args.output_path, 'w', encoding='utf-8')
for _pred in score_ref015:
    fw.write(str(_pred) + '\n')
fw.flush()
fw.close()




