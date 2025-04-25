import re
import os
import jieba

stopwords_table_path = os.getcwd() + "/llm-corpus/data/四川大学机器智能实验室停用词库.txt"

# 读取停词表，并使用set来存储
with open(stopwords_table_path, 'r', encoding='utf-8') as file:
    stopwords_set = set(line.strip() for line in file)

NewsCatalog = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

dir_path = os.getcwd() + "/llm-corpus/data/THUCNews"

def list_all_files(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file != '.DS_Store':  # 添加此行来跳过.DS_Store文件
                result.append(os.path.join(root, file))
    return result

for category in NewsCatalog:
    categorya_dir_path = dir_path + "/" + category
    if not os.path.exists(categorya_dir_path):
        os.makedirs(categorya_dir_path)
        
    combine = open(dir_path + '/' + '{}.txt'.format(category), 'w', encoding='utf-8')
    sentence = []
    idx = 0
    print("处理类型：{}".format(category))
    for file_path in list_all_files(categorya_dir_path):
        if idx % 10000 == 0:
            print("  已处理{}：{}".format(category,idx))
        file = open(file_path, 'r', encoding='utf-8')
        txt = file.read().replace('\n　　',' ')      # 一篇文章为一排
        file.close()
        # 提取中文
        txt = ''.join(re.findall('[\u4e00-\u9fa5| |]', txt))
        # 分词
        txt = ' '.join(jieba.cut(txt, cut_all=False)).replace('   ',' ')
        # 删除停用词
        for word in txt.split(' '):
            if word in stopwords_set:
                txt = txt.replace(word+' ','')
        sentence.append(txt+'\n')
        idx += 1 
    combine.write(''.join(sentence))
    print("总和：{}".format(idx))
print('文本处理完毕')
