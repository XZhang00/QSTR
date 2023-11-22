from utils import run_multi_bleu
import argparse
from numpy import mean
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pg_file', '-pg', type=str, default='../QQPPos-diversity-results/AESOP-siscp-diverse-0.2-pgs.txt')

args = parser.parse_args()

def cal_bleu_diverse(pg_file):
    all_10_pgs = []
    for i in range(10):
        all_10_pgs.append([])
    # print("+++", len(all_10_pgs))
    with open(pg_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            all_10_pgs[i % 10].append(line.strip())
    fr.close()
    for i in all_10_pgs:
        assert len(i) == 3000

    cnt = 1
    all_pg_name = []
    for cur_pgs in all_10_pgs:
        tmp_dir = "./tmp-bleu/"
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
        tmp_name = tmp_dir + "tmp-" + str(cnt) + ".txt"
        all_pg_name.append(tmp_name)
        fw = open(tmp_name, 'w', encoding='utf-8')
        for i_pg in cur_pgs:
            fw.write(i_pg + '\n')
        fw.flush()
        fw.close()
        cnt += 1

    st = time.time()
    bleu_all = []
    for i in range(len(all_pg_name)):
        for j in range(i+1, len(all_pg_name)):
            ij = run_multi_bleu(all_pg_name[i], all_pg_name[j])
            ji = run_multi_bleu(all_pg_name[j], all_pg_name[i])
            bleu_all.append(ij)
            bleu_all.append(ji)

    et = time.time()
    # print(len(bleu_all), et-st, "s")
    return mean(bleu_all)

res = cal_bleu_diverse(args.pg_file)
print(res)
print(args.pg_file)