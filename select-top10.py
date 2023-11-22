# 1. 直接抽取前10条句法解析树

fr = open("../retrieve-results-siscp-qqppos/sort_batch+mse-lr_3e-5-epoch19-parse.txt", 'r', encoding='utf-8')

all_parses = []
for line in fr.readlines():
    all_parses.append(line.strip())

fr.close()
print(len(all_parses))

num = 100
save_parses = []
for i in range(0, len(all_parses), 1000):
    save_parses.extend(all_parses[i: i+num])

print(len(save_parses))

fw = open("../retrieve-results-siscp-qqppos/SISCP-roberta-epoch19-top100-parse.txt", 'w', encoding='utf-8')

for j in save_parses:
    fw.write(j + '\n')
fw.close()


# 2. 在pred前加高斯噪声打乱顺序后，选前10条；