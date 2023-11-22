import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import copy
from apted import APTED
from apted.helpers import Tree
from numpy import mean
import random
from tqdm import tqdm


class FTDataset(Dataset):
    def __init__(self, src_path, src_parse_path, ref_parse_path, all_parse_path, tokenizer, syn_vocab2id, max_sen_len=64, max_syn_len=192, select_num=10):
        src_sens = []
        with open(src_path, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                src_sens.append(line.strip())
        fr.close()
        src_parses_list = json.load(open(src_parse_path))
        ref_parses_list = json.load(open(ref_parse_path))
        all_trees_list = json.load(open(all_parse_path))
        print("***", len(src_parses_list), len(ref_parses_list), len(all_trees_list))
        
        assert len(src_sens) == len(src_parses_list) == len(ref_parses_list)
        self.src_sens = src_sens
        self.src_parses_list = src_parses_list
        self.ref_parses_list = ref_parses_list
        self.all_trees_list = all_trees_list
        self.tokenizer = tokenizer
        self.syn_vocab2id = syn_vocab2id
        self.max_sen_len = max_sen_len
        self.max_syn_len = max_syn_len
        self.pad_id = 0
        self.select_num = select_num
        self.processed_data = []
        self.process()

        print(len(self.processed_data))

    def __len__(self):
        return len(self.src_sens)
    
    def process(self):
        for idx in tqdm(range(len(self.src_sens))):
            src = self.tokenizer(self.src_sens[idx])
            aesop_select_parses = random.sample(self.all_trees_list, (self.select_num - 2))
            assert len(aesop_select_parses) == self.select_num - 2
            aesop_select_parses.append(self.src_parses_list[idx][1])
            aesop_select_parses.append(self.ref_parses_list[idx][1])
            aesop_src_full = self.src_parses_list[idx][0]

            assert len(aesop_select_parses) == self.select_num
            syns = []
            for _p in aesop_select_parses:
                tmp_syn = '<s> ' + _p.replace("(", "( ").replace(")", ") ") + ' </s>'
                syn = [self.syn_vocab2id[i] if i in self.syn_vocab2id else self.syn_vocab2id["<unk>"] for i in tmp_syn.split()]
                syns.append(syn)
            assert len(aesop_select_parses) == self.select_num == len(syns)

            src_sens = [self.src_sens[idx]] * self.select_num
            aesop_src_inputs = []
            for i_src, i_parse in zip(src_sens, aesop_select_parses):
                aesop_src_inputs.append(f"{i_src} <sep>  {aesop_src_full} <sep> {i_parse}")

            cur_input = {
                "input_ids": src['input_ids'],
                "syn_input_ids": syns, 
                "src_sens": src_sens,
                "aesop_src_inputs": aesop_src_inputs
            }
            self.processed_data.append(cur_input)

    def __getitem__(self, idx):        
        return self.processed_data[idx]
    
    def padding(self, 
                instance_list: list,
                data_type: str) -> torch.Tensor:

        if 'syn' in data_type:
            max_seq_len = self.max_syn_len
            tensor = torch.LongTensor([j[:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(j)) for i in instance_list for j in i[data_type]])
        elif data_type == 'input_ids':
            max_seq_len = self.max_sen_len
            tensor = torch.LongTensor([i[data_type][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(i[data_type])) for i in instance_list for z in range(self.select_num)])
        elif data_type == 'src_sens' or data_type == "aesop_src_inputs" :
            tensor = [j for i in instance_list for j in i[data_type]]

        return tensor

    def collate(self, batch: list) -> torch.Tensor:
        
        sen_input_ids = self.padding(batch, 'input_ids')
        syn_input_ids = self.padding(batch, 'syn_input_ids')

        sen_attention_mask = sen_input_ids.ne(self.pad_id)
        syn_attention_mask = syn_input_ids.ne(self.pad_id)

        sen_token_type_ids = torch.zeros_like(sen_input_ids)
        syn_token_type_ids = torch.zeros_like(syn_input_ids)
        
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1)
        
        src_sens = self.padding(batch, 'src_sens')
        aesop_src_inputs = self.padding(batch, 'aesop_src_inputs')

        batch_data = {
            'sen_input_ids': sen_input_ids,
            "sen_attention_mask": sen_attention_mask,
            "sen_token_type_ids": sen_token_type_ids,
            "sen_position_ids": sen_position_ids,
            "syn_input_ids": syn_input_ids,
            "syn_attention_mask": syn_attention_mask,
            "syn_token_type_ids": syn_token_type_ids,
            "syn_position_ids": syn_position_ids,
            "src_sens": src_sens,
            "aesop_src_inputs": aesop_src_inputs
        }
        return batch_data


class FastFTDataset(Dataset):
    def __init__(self, src_path, src_random_parse_path, score_path, tokenizer, syn_vocab2id, max_sen_len=64, max_syn_len=192, select_num=10):
        src_sens = []
        with open(src_path, 'r', encoding='utf-8') as fr1:
            for line in fr1.readlines():
                src_sens.append(line.strip())
        fr1.close()

        src_random_trees = []
        with open(src_random_parse_path, 'r', encoding='utf-8') as fr2:
            tmp_trees = []
            for line in fr2.readlines():
                tmp_trees.append(line.strip())
                if len(tmp_trees) == 10:
                    src_random_trees.append(tmp_trees)
                    tmp_trees = []
        fr2.close()

        scores = []
        with open(score_path, 'r', encoding='utf-8') as fr3:
            tmp_scores = []
            for line in fr3.readlines():
                tmp_scores.append(float(line.strip()))
                if len(tmp_scores) == 10:
                    scores.append(tmp_scores)
                    tmp_scores = []
        fr3.close()
        assert len(src_random_trees) == len(src_sens) == len(scores)
        print("***", len(src_sens), len(src_random_trees), len(scores))
        
        self.src_sens = src_sens
        self.src_random_trees = src_random_trees
        self.scores = scores
        self.tokenizer = tokenizer
        self.syn_vocab2id = syn_vocab2id
        self.max_sen_len = max_sen_len
        self.max_syn_len = max_syn_len
        self.pad_id = 0
        self.select_num = select_num
        self.process_data = []
        self.process()
        print(len(self.process_data))


    def __len__(self):
        return len(self.src_sens)

    def process(self):
        for idx in tqdm(range(len(self.src_sens))):
            src = self.tokenizer(self.src_sens[idx])
            select_parses = self.src_random_trees[idx]
            para_scores = self.scores[idx]
            assert len(select_parses) == len(para_scores) == self.select_num    

            syns = []
            for _p in select_parses:
                tmp_syn = '<s> ' + _p.replace("(", "( ").replace(")", ") ") + ' </s>'
                syn = [self.syn_vocab2id[i] if i in self.syn_vocab2id else self.syn_vocab2id["<unk>"] for i in tmp_syn.split()]
                syns.append(syn)
            assert self.select_num == len(syns)

            cur_input = {
                "input_ids": src['input_ids'],
                "syn_input_ids": syns, 
                "scores": para_scores
            }
            self.process_data.append(cur_input)

    def __getitem__(self, idx):

        return self.process_data[idx]
    
    def padding(self, 
                instance_list: list,
                data_type: str) -> torch.Tensor:

        if 'syn' in data_type:
            max_seq_len = self.max_syn_len
            tensor = torch.LongTensor([j[:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(j)) for i in instance_list for j in i[data_type]])
        elif data_type == 'input_ids':
            max_seq_len = self.max_sen_len
            tensor = torch.LongTensor([i[data_type][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(i[data_type])) for i in instance_list for z in range(self.select_num)])
        elif data_type == 'scores':
            tensor = torch.tensor([j for i in instance_list for j in i[data_type]])

        return tensor

    def collate(self, batch: list) -> torch.Tensor:
        
        sen_input_ids = self.padding(batch, 'input_ids')
        syn_input_ids = self.padding(batch, 'syn_input_ids')

        sen_attention_mask = sen_input_ids.ne(self.pad_id)
        syn_attention_mask = syn_input_ids.ne(self.pad_id)

        sen_token_type_ids = torch.zeros_like(sen_input_ids)
        syn_token_type_ids = torch.zeros_like(syn_input_ids)
        
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1)
        
        scores = self.padding(batch, 'scores')

        batch_data = {
            'sen_input_ids': sen_input_ids,
            "sen_attention_mask": sen_attention_mask,
            "sen_token_type_ids": sen_token_type_ids,
            "sen_position_ids": sen_position_ids,
            "syn_input_ids": syn_input_ids,
            "syn_attention_mask": syn_attention_mask,
            "syn_token_type_ids": syn_token_type_ids,
            "syn_position_ids": syn_position_ids,
            "scores": scores
        }
        return batch_data


class FastFTDataset_roberta(Dataset):
    # roberta 模型的pad字符是1
    def __init__(self, src_path, src_random_parse_path, score_path, tokenizer, syn_vocab2id, max_sen_len=64, max_syn_len=192, select_num=10):
        src_sens = []
        with open(src_path, 'r', encoding='utf-8') as fr1:
            for line in fr1.readlines():
                src_sens.append(line.strip())
        fr1.close()

        src_random_trees = []
        with open(src_random_parse_path, 'r', encoding='utf-8') as fr2:
            tmp_trees = []
            for line in fr2.readlines():
                tmp_trees.append(line.strip())
                if len(tmp_trees) == 10:
                    src_random_trees.append(tmp_trees)
                    tmp_trees = []
        fr2.close()

        scores = []
        with open(score_path, 'r', encoding='utf-8') as fr3:
            tmp_scores = []
            for line in fr3.readlines():
                tmp_scores.append(float(line.strip()))
                if len(tmp_scores) == 10:
                    scores.append(tmp_scores)
                    tmp_scores = []
        fr3.close()
        assert len(src_random_trees) == len(src_sens) == len(scores)
        print("***", len(src_sens), len(src_random_trees), len(scores))
        
        self.src_sens = src_sens
        self.src_random_trees = src_random_trees
        self.scores = scores
        self.tokenizer = tokenizer
        self.syn_vocab2id = syn_vocab2id
        self.max_sen_len = max_sen_len
        self.max_syn_len = max_syn_len
        self.pad_id = 1
        self.select_num = select_num
        self.process_data = []
        self.process()
        print(len(self.process_data))


    def __len__(self):
        return len(self.src_sens)

    def process(self):
        for idx in range(len(self.src_sens)):
            src = self.tokenizer(self.src_sens[idx])
            select_parses = self.src_random_trees[idx]
            para_scores = self.scores[idx]
            assert len(select_parses) == len(para_scores) == self.select_num    

            syns = []
            for _p in select_parses:
                tmp_syn = '<s> ' + _p.replace("(", "( ").replace(")", ") ") + ' </s>'
                syn = [self.syn_vocab2id[i] if i in self.syn_vocab2id else self.syn_vocab2id["<unk>"] for i in tmp_syn.split()]
                syns.append(syn)
            assert self.select_num == len(syns)

            cur_input = {
                "input_ids": src['input_ids'],
                "syn_input_ids": syns, 
                "scores": para_scores
            }
            self.process_data.append(cur_input)

    def __getitem__(self, idx):

        return self.process_data[idx]
    
    def padding(self, 
                instance_list: list,
                data_type: str) -> torch.Tensor:

        if 'syn' in data_type:
            max_seq_len = self.max_syn_len
            tensor = torch.LongTensor([j[:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(j)) for i in instance_list for j in i[data_type]])
        elif data_type == 'input_ids':
            max_seq_len = self.max_sen_len
            tensor = torch.LongTensor([i[data_type][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(i[data_type])) for i in instance_list for z in range(self.select_num)])
        elif data_type == 'scores':
            tensor = torch.tensor([j for i in instance_list for j in i[data_type]])

        return tensor

    def collate(self, batch: list) -> torch.Tensor:
        
        sen_input_ids = self.padding(batch, 'input_ids')
        syn_input_ids = self.padding(batch, 'syn_input_ids')

        sen_attention_mask = sen_input_ids.ne(self.pad_id)
        syn_attention_mask = syn_input_ids.ne(self.pad_id)

        sen_token_type_ids = torch.zeros_like(sen_input_ids)
        syn_token_type_ids = torch.zeros_like(syn_input_ids)
        
        sen_position_ids = torch.LongTensor([i_posi for i_posi in range(sen_input_ids.shape[1])]).repeat(sen_input_ids.shape[0]).reshape(sen_input_ids.shape[0], -1)
        syn_position_ids = torch.LongTensor([i_posi for i_posi in range(syn_input_ids.shape[1])]).repeat(syn_input_ids.shape[0]).reshape(syn_input_ids.shape[0], -1)
        
        scores = self.padding(batch, 'scores')

        batch_data = {
            'sen_input_ids': sen_input_ids,
            "sen_attention_mask": sen_attention_mask,
            "sen_token_type_ids": sen_token_type_ids,
            "sen_position_ids": sen_position_ids,
            "syn_input_ids": syn_input_ids,
            "syn_attention_mask": syn_attention_mask,
            "syn_token_type_ids": syn_token_type_ids,
            "syn_position_ids": syn_position_ids,
            "scores": scores
        }
        return batch_data



def normalize_tree(tree_string, max_depth=50):
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


def tree_edit_distance(lintree1, lintree2):
    tree1 = Tree.from_text(lintree1)
    tree2 = Tree.from_text(lintree2)
    n_nodes_t1 = lintree1.count('{')
    n_nodes_t2 = lintree2.count('{')
    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    return ted / (n_nodes_t1 + n_nodes_t2)



if __name__ == "__main__":
    # fr = open("../data-QQPPos/train/src.parse", 'r', encoding='utf-8')
    # src_parses = []
    # for line in fr.readlines():
    #     src_parses.append(line.strip())
    # fr.close()

    # print(len(src_parses))
    # src_aesop_trees_h5 = json.load(open("../data-QQPPos/src_aesop_trees_h5.json"))
    # print(len(src_aesop_trees_h5))

    # new_ls = []
    # for i, j in zip(src_parses, src_aesop_trees_h5):
    #     new_ls.append([i, j[1]])
    
    # fw = open("../data-QQPPos/train/src_trees_h5.json", 'w', encoding='utf-8')
    # fw.write(json.dumps(new_ls, ensure_ascii=False))
    # fw.close()
    # print("+++")

    fr = open("../data-QQPPos/test/aesop-level5-ref.source", 'r', encoding='utf-8')
    src_parses = []
    for i in fr.readlines():
        src_parses.append(i.split(" <sep> ")[-1])
    fr.close()

    fw = open("../data-QQPPos/test/ref.tree_h5", 'w', encoding='utf-8')
    for i in src_parses:
        fw.write(i)
    fw.close()
    print()