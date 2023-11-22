from parascorer_new import ParaScorer
import argparse
import time
from numpy import mean
import os
device_id = 7
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
import torch
device = torch.device('cuda')


def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())
    fr.close()
    return res


device_id = device_id
parser = argparse.ArgumentParser()
parser.add_argument('--src_file', '-s', type=str, default='../data-QQPPos/train/src.txt')
parser.add_argument('--ref_file', '-r', type=str, default='../data-QQPPos/train/ref.txt')
parser.add_argument('--pg_file', '-p', type=str, default='../data-QQPPos/train/save4train-1W/src_random.pg' + str(device_id))

parser.add_argument('--output_path', '-t', type=str, default='../data-QQPPos/train/save4train-1W/src_random.score-ref015-' + str(device_id))
args = parser.parse_args()


scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)
fw = open(args.output_path, 'w', encoding='utf-8')

num = 10

train_src_ori = read_file(args.src_file)

start_num = int(device_id) * 11433
end_num = min((int(device_id) + 1) * 11433, len(train_src_ori))
print(device_id, start_num, end_num)

srcs = []
with open(args.src_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines()[start_num : end_num]:
        for i in range(num):
            srcs.append(line.strip())
fr.close()

refs = []
with open(args.ref_file, 'r', encoding='utf-8') as fr:
    for line in fr.readlines()[start_num : end_num]:
        for i in range(num):
            refs.append(line.strip())
fr.close()



pgs = read_file(args.pg_file)

assert len(srcs) == len(refs) == len(pgs)
print(len(srcs), len(pgs))

score_ref015 = scorer.base_score(pgs, srcs, refs, 0.15)

print(mean(score_ref015))

assert len(srcs) == len(refs) == len(score_ref015)

for _pred in score_ref015:
    fw.write(str(_pred) + '\n')
fw.flush()
fw.close()

print(args.pg_file)



