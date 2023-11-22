import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, AutoConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers import BartForConditionalGeneration
import copy
import subprocess
from apted import APTED
from apted.helpers import Tree
from numpy import mean
import random


STANFORD_CORENLP = './tools/stanford-corenlp-full-2018-10-05'
class stanford_parsetree_extractor:
    def __init__(self, out_path="./tmp-out"):
        self.stanford_corenlp_path = os.path.join(STANFORD_CORENLP, "*")
        # print("standford corenlp path:", self.stanford_corenlp_path)
        self.output_dir = out_path
        self.cmd = ['java', '-cp', self.stanford_corenlp_path,
                    '-Xmx2G', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                    '-parse.model', 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
                    '-annotators', 'tokenize,ssplit,pos,parse',
                    '-ssplit.eolonly', '-outputFormat', 'text',
                    '-outputDirectory', self.output_dir,
                    '-file', None]
                    # '-parse.model', 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',     
                    # 测试SI-SCP和SGCP的结果时需要把SR模型删除，这两篇工作用的是默认的ECPG模型；
    def run(self, file):
        # print("parsing file:", file)
        self.cmd[-1] = file
        out = subprocess.run(
            self.cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        # print(out)
        parsed_file = \
            os.path.join(
                self.output_dir,
                os.path.split(file)[-1] + ".out")
        return parsed_file

    # def cleanup(self):
    #     self.output_dir.cleanup()


MULTI_BLEU_PERL = './tools/multi-bleu.perl'
def run_multi_bleu(input_file, reference_file, MULTI_BLEU_PERL=MULTI_BLEU_PERL):
    bleu_output = subprocess.check_output(
        "perl {} -lc {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    # print(bleu_output)
    bleu = float(
        bleu_output.strip().split("\n")[-1].split(",")[0].split("=")[1][1:])
    return bleu



def from_out2parses(out_file):
    parse_list = []
    with open(out_file, 'r', encoding='utf-8') as fr:
        is_parse = False
        cur_parse = []
        for line in fr.readlines():
            if line[:12] == "Constituency":
                is_parse = True
            if is_parse and line.strip() == "": 
                is_parse = False
                parse_list.append(" ".join(cur_parse))
                cur_parse = []
            if is_parse: 
                if line[:12] != "Constituency":
                    cur_parse.append(line.strip())
    # print(parse_list[0])
    return parse_list


def deleaf(tree):
    def is_paren(tok):
        return tok == ")" or tok == "("
    
    nonleaves = ''
    for w in tree.replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '
    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr).split()
    return " ".join(nonleaves)


def get_control_tree(parses):
    trees = []
    for p in parses:
        _tree = deleaf(p)
        trees.append(_tree)
    assert len(trees) == len(parses)
    return trees


def normalize_tree(tree_string, max_depth=5):
    res = []
    depth = -1
    leaf = False
    for c in tree_string:
        if c in ['{', '}']:
            continue
        if c == '(':
            leaf=False
            depth += 1

        elif c == ')':
            leaf=False
            depth -= 1
            if depth < max_depth:
                res.append('}')
                continue
                
        elif c == ' ':
            leaf=True
            continue

        if depth <= max_depth and not leaf and c != ')':
            res.append(c if c != '(' else '{')
        
    return ''.join(res)


def get_nomal_trees(parses, depth=1000):
    nomal_parses = []
    for _p in parses:
        nomal_parses.append(normalize_tree(_p, max_depth=depth))
    assert len(nomal_parses) == len(parses)
    return nomal_parses


def tree_edit_distance(lintree1, lintree2):
    tree1 = Tree.from_text(lintree1)
    tree2 = Tree.from_text(lintree2)
    n_nodes_t1 = lintree1.count('{')
    n_nodes_t2 = lintree2.count('{')
    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    return ted / (n_nodes_t1 + n_nodes_t2)


def cal_tree_dis(spe, path_s, path_r, depth=1000):
    parses_s = from_out2parses(spe.run(path_s))
    parses_r = from_out2parses(spe.run(path_r))
    nomal_trees_s = get_nomal_trees(parses_s, depth)
    nomal_trees_r = get_nomal_trees(parses_r, depth)
    assert len(nomal_trees_s) == len(nomal_trees_r)
    dis = []
    for i,j in zip(nomal_trees_s, nomal_trees_r):
        dis.append(tree_edit_distance(i, j))
    return mean(dis)


def cal_tree_dis2(spe, path_s, path_r, depth=1000):
    parses_s = from_out2parses(spe.run(path_s))
    parses_r = from_out2parses(spe.run(path_r))
    nomal_trees_s = get_nomal_trees(parses_s, depth)
    nomal_trees_r = get_nomal_trees(parses_r, depth)
    assert len(nomal_trees_s) == len(nomal_trees_r)
    dis = []
    for i,j in zip(nomal_trees_s, nomal_trees_r):
        dis.append(tree_edit_distance(i, j))
    return dis


if __name__ == "__main__":
    # build_syn_vocab()
    # dis = cal_tree_dis('../data-QQPPos/epoch20-tgt-pg.txt', 
                    #    '../QQP-Pos-test-results/tgt.txt', 5)
    # print(dis)

    # tmp_parses = from_out2parses('../data-QQPPos/train/SR.out/ref.txt.out')
    # print(len(tmp_parses))
    # control_trees = get_control_tree(tmp_parses)
    # nomal_trees = get_nomal_trees(tmp_parses, 1000)
    # assert len(control_trees) == len(nomal_trees)
    # save_dict = {}
    # cnt = 0
    # for _c, _n in zip(control_trees, nomal_trees):
    #     if _c in save_dict:
    #         cnt += 1
    #     else:
    #         save_dict[_c] = _n
    
    # print(cnt)
    # with open('../data-QQPPos/all_trees-F.json', 'w', encoding='utf-8') as fw:
    #     fw.write(json.dumps(save_dict, ensure_ascii=False))
    # fw.close()
    # print(len(save_dict))

    # tmp_parses = from_out2parses('../data-QQPPos/train/SR.out/src.txt.out')
    # print(len(tmp_parses))
    # control_trees = get_control_tree(tmp_parses)
    # nomal_trees = get_nomal_trees(tmp_parses, 1000)
    # assert len(control_trees) == len(nomal_trees)
    # save_list = []
    # cnt = 0
    # for _c, _n in zip(control_trees, nomal_trees):
    #     save_list.append([_c, _n])
    
    # print(cnt)
    # with open('../data-QQPPos/src_trees-F.json', 'w', encoding='utf-8') as fw:
    #     fw.write(json.dumps(save_list, ensure_ascii=False))
    # fw.close()
    # print(len(save_list))

    # print()

    # tmp_parses = from_out2parses('../data-QQPPos/train/SR.out/src.txt.out')
    # print(len(tmp_parses))
    # fw = open('../data-QQPPos/train/src.parse', 'w', encoding='utf-8')
    # for i in tmp_parses:
    #     fw.write(i + '\n')
    # fw.close()


    fw = open("../data-QQPPos/val/tgt.parse", 'w', encoding='utf-8')

    res = from_out2parses('../data-QQPPos/val/SR.out/tgt.txt.out')
    for i in res:
        fw.write(i + '\n')
    fw.close()
