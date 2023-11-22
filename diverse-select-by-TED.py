from apted import APTED
from apted.helpers import Tree
from numpy import mean
import random
import time

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


def tree_edit_distance(lintree1, lintree2):
    tree1 = Tree.from_text(lintree1)
    tree2 = Tree.from_text(lintree2)
    n_nodes_t1 = lintree1.count('{')
    n_nodes_t2 = lintree2.count('{')
    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    return ted / (n_nodes_t1 + n_nodes_t2)



parse_file = "../QQPPos-retrieve-parses/SISCP-siscp-retrieve-parses-per1000.txt"
thres = 0.2
output_file = "../QQPPos-diversity-results/siscp-diverse-"+ str(thres) + ".txt"

fw = open(output_file, 'w', encoding='utf-8')

all_parses = []
with open(parse_file, "r", encoding='utf-8') as fr:
    for line in fr.readlines():
        all_parses.append(line.strip())

fr.close()
print(len(all_parses))       

st = time.time()
selected_parse = []
for i in range(0, len(all_parses), 1000):
    print(i)
    start = i
    end = i+1000
    cur_parses = all_parses[start:end]
    assert len(cur_parses) == 1000
    cur_selected_parses = []
    cur_selected_parses.append(cur_parses[0])
    for _p in cur_parses[1:]:
        ted_past = []
        for _zp in cur_selected_parses:
            normal_p = normalize_tree(_p)
            normal_zp = normalize_tree(_zp)
            ted_past.append(tree_edit_distance(normal_p, normal_zp))
        # print(ted_past)
        # print(mean(ted_past))
        if min(ted_past) > thres:
            cur_selected_parses.append(_p)
            if len(cur_selected_parses) == 10:
                break    

    if len(cur_selected_parses) < 10:
        print(f"{i} - warning! ")
        tmp_r = random.sample(cur_parses[:100], 10-len(cur_selected_parses))
        cur_selected_parses.extend(tmp_r)
    assert len(cur_selected_parses) == 10
    for j in cur_selected_parses:
        fw.write(j + "\n")
    fw.flush()

fw.close()

et = time.time()
print(et-st, 's')

