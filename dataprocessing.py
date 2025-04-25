import re
import opencc
import jieba
import os

zhwiki_dir_path = os.getcwd() + "/llm-corpus/data/zhwiki-pages-articles"
stopword_dir_path = zhwiki_dir_path + '/stopword_txt'
seg_dir_path = zhwiki_dir_path + '/parse_txt'
stopwords_table_path = os.getcwd() + "/llm-corpus/data/四川大学机器智能实验室停用词库.txt"

cc = opencc.OpenCC('t2s')


# 读取停词表，并使用set来存储
with open(stopwords_table_path, 'r', encoding='utf-8') as file:
    stopwords_set = set(line.strip() for line in file)

    
def checkDir():
    # 检查你的目录是否存在，如果不存在，创建它
    if not os.path.exists(zhwiki_dir_path):
        os.makedirs(zhwiki_dir_path)
    if not os.path.exists(stopword_dir_path):
        os.makedirs(stopword_dir_path)
    if not os.path.exists(seg_dir_path):
        os.makedirs(seg_dir_path)

def simplified_Chinese(txt):
    txt_sim = []
    for sentence in txt.split('\n'):
        txt_sim.append(cc.convert(sentence) + '\n')
        #print("第{}句话转化成简体成功".format(i))
    txt = ''.join(txt_sim)
    return txt_sim


def remove_stopwords(idx):
    in_path = seg_dir_path
    out_path = stopword_dir_path
    
    with open(f'{in_path}/zhwikiSegDone_{idx}.txt', 'r', encoding='utf-8') as file_in,\
         open(f'{out_path}/zhwikiStopWord_{idx}.txt', 'w', encoding='utf-8') as file_out:
        
        # 读取所有行，并将每一行分割成单词
        lines = file_in.readlines()
        sentence_list = [line.split(' ') for line in lines]
        
        # 对每一行的每一个单词，如果它不在停词表中，就保留
        result = []
        for words in sentence_list:
            result.append(' '.join(word for word in words if word not in stopwords_set))

        # 将结果写入文件
        file_out.write('\n'.join(result))



def seg_done(idx, txt):
    # 以下为分词部分
    out_path = seg_dir_path
    file = open(out_path + '/zhwikiSegDone_{}.txt'.format(idx),'w',encoding='utf-8')
    for t in txt:
        file.write(' '.join(jieba.cut(t, cut_all=False)).replace(' \n ', '\n'))
    file.close()


def parse_txt():
    in_path = zhwiki_dir_path
    for i in range(0, 47):      # 理论上应该是从0至47
        file = open(in_path+'/zhwiki_{}.txt'.format(i),'r',encoding='utf-8')
        txt = file.read()
        file.close()
        # 1. 提取汉字
        txt = ''.join(re.findall('[\u4e00-\u9fa5|\n]',txt))      # 只保留汉字,如果其后有空格则保留
        print('第' + str(i) + '个txt文件提取汉字成功')
        # 2. 简化汉字
        txt = simplified_Chinese(txt)
        print('第' + str(i) + '个txt文件繁体汉字转化简体汉字成功')
        # 3. 汉字分词
        seg_done(i, txt)
        print('第' + str(i) + '个txt文件分词成功')
        # 4. 去除停用词
        remove_stopwords(i)
        print('第' + str(i) + '个txt文件去除停用词成功')

checkDir()
parse_txt()