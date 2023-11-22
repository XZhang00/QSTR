from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def read_file(path):
    res = []
    fr = open(path, 'r', encoding='utf-8') 
    for line in fr.readlines():
        res.append(line.strip())
    fr.close()
    return res

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def deal_non_sep(string):
    if "ROOT" not in string:
        # did not try to generate the syntactic parse at all
        final_str = string
    else:
        last_para_count = 0
        # find the last element with "(" in it
        for i in range(0, len(string.split(" "))):
            item = string.split(" ")[i]
            if "(" in item or ")" in item:
                last_para_count = i
        valid_tokens = string.split(" ")[last_para_count + 1:]
        final_str = " ".join(token for token in valid_tokens)
    if final_str == "":
        final_str = "."
    return final_str

def get_pgs_from_AESOP(model, tokenizer, src_examples, batch_size=10):
    gen_pgs = []
    for examples_chunk in list(chunks(src_examples, batch_size)):
        batch = tokenizer(examples_chunk, return_tensors="pt", truncation=True, padding="longest").to(model.device)
        summaries = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_length=256,
            num_beams=4
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            if "<sep>" in hypothesis:
                gen_pgs.append(hypothesis.split('<sep>')[1].lstrip())
            else:
                gen_pgs.append(deal_non_sep(hypothesis))
        # print(len(gen_pgs))
    assert len(gen_pgs) == len(src_examples)   
    
    return gen_pgs



if __name__ == "__main__":
    import json
    import random
    import os
    import copy
    import argparse
    from tqdm import tqdm
    # from parascore import ParaScorer
    # scorer = ParaScorer(lang="en", model_type = 'bert-base-uncased', device=device)
    parser = argparse.ArgumentParser()

    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--device_num", default=8, type=int)
    args = parser.parse_args()


    train_src_ori = read_file('../data-QQPPos/train/src.txt')
    src_parses_ls_ori = read_file("../data-QQPPos/train/src.parse")    # 为了构造AESOP的输入
    candidate_trees = json.load(open("../data-QQPPos/candidate_trees_h5_1W.json"))
    src_trees_ls_ori = read_file("../data-QQPPos/train/src.tree_h5")
    ref_trees_ls_ori = read_file("../data-QQPPos/train/ref.tree_h5")
    batch_num = 8
    
    length = len(train_src_ori)
    per_device_num = int(length/args.device_num) + 1
    print("per_device_num", per_device_num)
    number = args.device_id
    start_num = number * per_device_num
    end_num = min((number + 1) * per_device_num, length)
    print(number, start_num, end_num)

    fw_pg = open("../data-QQPPos/train/save4train-1W/src_random.pg" + str(number), 'w', encoding='utf-8')
    fw_tree = open("../data-QQPPos/train/save4train-1W/src_random.tree" + str(number), 'w', encoding='utf-8')

    train_src = copy.deepcopy(train_src_ori[start_num : end_num])
    src_parses_ls = copy.deepcopy(src_parses_ls_ori[start_num : end_num])
    src_trees_ls = copy.deepcopy(src_trees_ls_ori[start_num : end_num])
    ref_trees_ls = copy.deepcopy(ref_trees_ls_ori[start_num : end_num])

    print(len(train_src), len(src_parses_ls), len(src_trees_ls))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    import torch
    device = torch.device("cuda")

    aesop_model_name = '../pretrained-models/qqppos-h4-d'
    aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    aesop_model.eval()
    aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    i = 0
    for i_src in tqdm(train_src):
        src_sen = i_src
        input_trees = random.sample(candidate_trees, batch_num)
        aesop_src_parse = src_parses_ls[i]
        src_tree = src_trees_ls[i]
        ref_tree = ref_trees_ls[i]
        input_trees.append(src_tree)
        input_trees.append(ref_tree)

        cur_aesop_inputs = []
        for i_tree in input_trees:
            cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")

        cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num + 2)

        for _tree, _pg in zip(input_trees, cur_pgs):
            if _pg == ".":
                fw_pg.write(src_sen + '\n')
            else:
                fw_pg.write(_pg + '\n')
            fw_tree.write(_tree + '\n')
            fw_pg.flush()
            fw_tree.flush()
        i += 1 



    # qqppos数据集根据新检索的模板继续训练
    # train_src_ori = read_file('../data-QQPPos/train/src.txt')
    # src_parses_ls_ori = read_file("../data-QQPPos/train/src.parse")
    # top_trees_ls_ori = read_file("../retrieve-res_for_next_train/train/src_top.tree")
    # fw_pg = open("../retrieve-res_for_next_train/train/src_top.pg" + str(args.device_id), 'w', encoding='utf-8')
    # batch_num = 10

    
    # length = len(train_src_ori)
    # per_device_num = int(length/args.device_num) + 1
    # print("per_device_num", per_device_num)
    # number = args.device_id
    # start_num = number * per_device_num
    # end_num = min((number + 1) * per_device_num, length)
    # print(number, start_num, end_num)

    # train_src = copy.deepcopy(train_src_ori[start_num : end_num])
    # src_parses_ls = copy.deepcopy(src_parses_ls_ori[start_num : end_num])
    # top_trees_ls = copy.deepcopy(top_trees_ls_ori[start_num*batch_num : end_num*batch_num])

    # print(len(train_src), len(src_parses_ls), len(top_trees_ls))

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    # import torch
    # device = torch.device("cuda")

    # aesop_model_name = '../pretrained-models/qqppos-h4-d'
    # aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    # aesop_model.eval()
    # aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    # i = 0
    # for i_src in tqdm(train_src):
    #     src_sen = i_src     
    #     aesop_src_parse = src_parses_ls[i]

    #     cur_aesop_inputs = []
    #     print((i*batch_num), (i+1)*batch_num)
    #     for i_tree in top_trees_ls[(i*batch_num):(i+1)*batch_num]:
    #         cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")
    #     print(len(cur_aesop_inputs))
    #     cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num + 2)

    #     for _pg in cur_pgs:
    #         if _pg == ".":
    #             fw_pg.write(src_sen + '\n')
    #         else:
    #             fw_pg.write(_pg + '\n')
    #         fw_pg.flush()
    #     i += 1 

    # fw_pg.close()




    # # # ParaNMT数据集的处理

    # fw_tree = open("../data-ParaNMT/train/save4train/src_random.tree0", 'w', encoding='utf-8')
    # fw_pg = open("../data-ParaNMT/train/save4train/src_random.pg0", 'w', encoding='utf-8')

    # # 给训练集的每个句子随机保留有效的10条句法解析树包括src，ref，分别将句法解析树和生成的复述句存起来
    # train_src_ori = read_file('../data-ParaNMT/train/src.txt')
    # candidate_trees = json.load(open("../data-ParaNMT/candidate_trees_h5.json"))
    # src_parses_ls_ori = read_file("../data-ParaNMT/train/src.parse")
    # src_trees_ls_ori = read_file("../data-ParaNMT/train/src.tree_h5")
    # ref_trees_ls_ori = read_file("../data-ParaNMT/train/ref.tree_h5")

    # start_num = 0
    # end_num = 20

    # train_src = copy.deepcopy(train_src_ori[start_num : end_num])
    # src_parses_ls = copy.deepcopy(src_parses_ls_ori[start_num : end_num])
    # src_trees_ls = copy.deepcopy(src_trees_ls_ori[start_num : end_num])
    # ref_trees_ls = copy.deepcopy(ref_trees_ls_ori[start_num : end_num])

    # print(len(train_src), len(candidate_trees), len(src_parses_ls), len(src_trees_ls), len(ref_trees_ls))

    # batch_num = 8
    # aesop_model_name = '../pretrained-models/paranmt-h4'
    # aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    # aesop_model.eval()
    # aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    # i = 0
    # for i_src in tqdm(train_src):
    #     src_sen = i_src
    #     input_trees = random.sample(candidate_trees, batch_num)
    #     aesop_src_parse = src_parses_ls[i]
    #     src_tree = src_trees_ls[i]
    #     ref_tree = ref_trees_ls[i]
    #     input_trees.append(src_tree)
    #     input_trees.append(ref_tree)

    #     cur_aesop_inputs = []
    #     for i_tree in input_trees:
    #         cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")
    #     print(len(cur_aesop_inputs))
    #     cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num + 2)

    #     for _tree, _pg in zip(input_trees, cur_pgs):
    #         if _pg == ".":
    #             fw_pg.write(src_sen + '\n')
    #         else:
    #             fw_pg.write(_pg + '\n')
    #         fw_tree.write(_tree + '\n')
    #         fw_pg.flush()
    #         fw_tree.flush()
    #     i += 1 

    # fw_pg.close()
    # fw_tree.close()

    # # 给测试集和验证集每个句子选30条句法解析树生成复述句，并计算parascore，存起来用来算每个epoch的相关系数
    # fw_tree = open("../data-ParaNMT/val/save4val/src_random.tree", 'w', encoding='utf-8')
    # fw_pg = open("../data-ParaNMT/val/save4val/src_random.pg", 'w', encoding='utf-8')
    
    # test_src = read_file('../data-ParaNMT/val/aesop-level5-src.source')
    # candidate_trees = json.load(open("../data-ParaNMT/candidate_trees_h5.json"))
    # print(len(test_src), len(candidate_trees))

    # save_dict = {}
    # batch_num = 40
    # save_num = 30
    # aesop_model_name = '../pretrained-models/paranmt-h4'
    # aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    # aesop_model.eval()
    # aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    # i = 0
    # for i_src in tqdm(test_src):
    #     src_sen, aesop_src_parse, src_parse = i_src.split(" <sep> ")
    #     input_trees = random.sample(candidate_trees, batch_num)
    #     cur_aesop_inputs = []
    #     for i_tree in input_trees:
    #         cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")

    #     cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num)

    #     save_trees = []
    #     save_pgs = []
    #     for _tree, _pg in zip(input_trees, cur_pgs):
    #         if _pg == ".": continue
    #         save_pgs.append(_pg)
    #         save_trees.append(_tree)
    #     assert len(save_trees) == len(save_pgs)
    #     assert len(save_pgs) > save_num

    #     for z_tree, z_pg in zip(save_trees[:save_num], save_pgs[:save_num]):
    #         fw_pg.write(z_pg + '\n')
    #         fw_tree.write(z_tree + '\n')
    #         fw_pg.flush()
    #         fw_tree.flush()
    #     i += 1 

    # fw_pg.close()
    # fw_tree.close()

    # QQPPos 数据集的处理过程

    # fw_tree = open("../data-QQPPos/train/save4train/src_random.tree7", 'w', encoding='utf-8')
    # fw_pg = open("../data-QQPPos/train/save4train/src_random.pg7", 'w', encoding='utf-8')

    # # 给训练集的每个句子随机保留有效的10条句法解析树包括src，ref，分别将句法解析树和生成的复述句存起来
    # train_src_ori = read_file('../data-QQPPos/train/src.txt')
    # candidate_trees = json.load(open("../data-QQPPos/candidate_trees_h5.json"))
    # src_parse_ls_ori = json.load(open("../data-QQPPos/train/src_trees_h5.json"))
    # ref_parse_ls_ori = json.load(open("../data-QQPPos/train/ref_trees_h5.json"))

    # train_src = copy.deepcopy(train_src_ori[240079 : 274370])
    # src_parse_ls = copy.deepcopy(src_parse_ls_ori[240079 : 274370])
    # ref_parse_ls = copy.deepcopy(ref_parse_ls_ori[240079 : 274370])

    # print(len(train_src), len(candidate_trees), len(src_parse_ls), len(ref_parse_ls))

    # batch_num = 8
    # aesop_model_name = '../qqppos-h4-d'
    # aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    # aesop_model.eval()
    # aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    # i = 0
    # for i_src in tqdm(train_src):
    #     src_sen = i_src
    #     input_trees = random.sample(candidate_trees, batch_num)
    #     aesop_src_parse = src_parse_ls[i][0]
    #     src_tree = src_parse_ls[i][1]
    #     ref_tree = ref_parse_ls[i][1]
    #     input_trees.append(src_tree)
    #     input_trees.append(ref_tree)

    #     cur_aesop_inputs = []
    #     for i_tree in input_trees:
    #         cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")

    #     cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num + 2)

    #     for _tree, _pg in zip(input_trees, cur_pgs):
    #         if _pg == ".":
    #             fw_pg.write(src_sen + '\n')
    #         else:
    #             fw_pg.write(_pg + '\n')
    #         fw_tree.write(_tree + '\n')
    #         fw_pg.flush()
    #         fw_tree.flush()
    #     i += 1 

    # 一次处理多个句子，126小时约
    # for i in tqdm(range(0, len(train_src), 10)):
    #         src_sens = train_src[i : i+10]
    #         cur_aesop_inputs = []
    #         cur_input_trees_all = []
    #         for j, src_sen in enumerate(src_sens) :
    #             print(i+j)
    #             input_trees = random.sample(candidate_trees, select_num)
    #             aesop_src_parse = src_parse_ls[i+j][0]
    #             src_tree = src_parse_ls[i+j][1]
    #             ref_tree = ref_parse_ls[i+j][1]
    #             input_trees.append(src_tree)
    #             input_trees.append(ref_tree)
    #             cur_input_trees_all.extend(input_trees)

    #             for i_tree in input_trees:
    #                 cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")
    #         print(len(cur_aesop_inputs))

    #         cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num)

    #         for _tree, _pg in zip(cur_input_trees_all, cur_pgs):
    #             if _pg == ".":
    #                 fw_pg.write(src_sen + '\n')
    #             else:
    #                 fw_pg.write(_pg + '\n')
    #             fw_tree.write(_tree + '\n')
    #             fw_pg.flush()
    #             fw_tree.flush()

    # 给测试集和验证集每个句子选30条句法解析树生成复述句，并计算parascore，存起来用来算每个epoch的相关系数
    # fw_tree = open("../data-QQPPos/test/save4test/src_random.tree", 'w', encoding='utf-8')
    # fw_pg = open("../data-QQPPos/test/save4test/src_random.pg", 'w', encoding='utf-8')
    
    # test_src = read_file('../data-QQPPos/test/aesop-level5-src.source')
    # candidate_trees = json.load(open("../data-QQPPos/candidate_trees_h5.json"))
    # print(len(test_src), len(candidate_trees))

    # save_dict = {}
    # batch_num = 40
    # save_num = 30
    # aesop_model_name = '../qqppos-h4-d'
    # aesop_model = AutoModelForSeq2SeqLM.from_pretrained(aesop_model_name).to(device)
    # aesop_model.eval()
    # aesop_tokenizer = AutoTokenizer.from_pretrained(aesop_model_name)

    # i = 0
    # for i_src in tqdm(test_src):
    #     src_sen, aesop_src_parse, src_parse = i_src.split(" <sep> ")
    #     input_trees = random.sample(candidate_trees, batch_num)
    #     cur_aesop_inputs = []
    #     for i_tree in input_trees:
    #         cur_aesop_inputs.append(f"{src_sen} <sep> {aesop_src_parse} <sep> {i_tree}")

    #     cur_pgs = get_pgs_from_AESOP(aesop_model, aesop_tokenizer, cur_aesop_inputs, batch_num)

    #     save_trees = []
    #     save_pgs = []
    #     for _tree, _pg in zip(input_trees, cur_pgs):
    #         if _pg == ".": continue
    #         save_pgs.append(_pg)
    #         save_trees.append(_tree)
    #     assert len(save_trees) == len(save_pgs)
    #     assert len(save_pgs) > save_num

    #     for z_tree, z_pg in zip(save_trees[:save_num], save_pgs[:save_num]):
    #         fw_pg.write(z_pg + '\n')
    #         fw_tree.write(z_tree + '\n')
    #         fw_pg.flush()
    #         fw_tree.flush()
    #     i += 1 

    # fw_pg.close()
    # fw_tree.close()


