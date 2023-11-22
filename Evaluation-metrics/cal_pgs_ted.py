from utils import stanford_parsetree_extractor, tree_edit_distance, from_out2parses, get_nomal_trees
import argparse
from numpy import mean
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pg_file', '-pg', type=str, default='../QQPPos-diverse-pg/AESOP-roberta-sort_batch+mse-lr_3e-5-epoch19-parse-top10.source-pgs.txt')
parser.add_argument('--parses_file', '-parse', type=str, default='../QQPPos-retrieve-parses/roberta-sort_batch+mse-lr_3e-5-epoch19-parse-top10.txt')

args = parser.parse_args()


def cal_pgs_ted(pg_file, parses_file):
    all_10_pgs = []
    for i in range(10):
        all_10_pgs.append([])
    # print("+++", len(all_10_pgs))
    with open(pg_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            tmp = line.strip()
            if tmp == "":
                all_10_pgs[i % 10].append(".")
            else:
                all_10_pgs[i % 10].append(tmp)
    fr.close()
    for i in all_10_pgs:
        assert len(i) == 3000

    all_10_parses = []
    for i in range(10):
        all_10_parses.append([])
    # print("+++", len(all_10_pgs))
    with open(parses_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            all_10_parses[i % 10].append(line.strip())
    fr.close()
    for i in all_10_parses:
        assert len(i) == 3000

    spe = stanford_parsetree_extractor()
    cnt = 1
    all_dis = []
    for cur_pgs, cur_parses in zip(all_10_pgs, all_10_parses):
        # cur_pgs = all_10_pgs[cnt-1]
        # cur_parses = all_10_parses[cnt-1]
        tmp_dir = "./tmp-out/"
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
        tmp_name = tmp_dir + "tmp-" + str(cnt) + ".txt"
        fw = open(tmp_name, 'w', encoding='utf-8')
        for i_pg in cur_pgs:
            fw.write(i_pg + '\n')
        fw.flush()
        fw.close()
        pg_parses = from_out2parses(spe.run(tmp_name))

        nomal_trees_s = get_nomal_trees(pg_parses, 5)
        nomal_trees_r = get_nomal_trees(cur_parses, 5)
        assert len(nomal_trees_s) == len(nomal_trees_r)
        dis = []
        for i,j in zip(nomal_trees_s, nomal_trees_r):
            dis.append(tree_edit_distance(i, j))
        all_dis.append(mean(dis))
        cnt += 1
    
    # print(all_dis)
    for _dis in all_dis:
        print(_dis)
    return mean(all_dis)
        


st = time.time()
res = cal_pgs_ted(args.pg_file, args.parses_file)
et = time.time()
print(f"cost: {et-st} s")
print(res)
print(args.pg_file, args.parses_file)
