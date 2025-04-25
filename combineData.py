import os

NewsCatalog = ['体育','娱乐','家居','彩票','房产','教育','时尚','时政','星座','游戏','社会','科技','股票','财经']

path = os.getcwd() + "/llm-corpus/data"
wiki_path = os.getcwd() + "/llm-corpus/data/zhwiki-pages-articles/stopword_txt"
THUCNews_path = os.getcwd() + "/llm-corpus/data/THUCNews"

Data = open(path + '/data.txt', 'a', encoding='utf-8')

for i in range(47):      # 合并中文wiki百科文件
    file = open(wiki_path + '/zhwikiStopWord_{}.txt'.format(i), 'r', encoding='utf-8')
    txt = file.read().strip('\n').strip(' ')
    Data.write(txt + '\n')
    file.close()

print('中文wiki百科文件合并完成')

for item in NewsCatalog:        # 合并THU数据集
    file = open(THUCNews_path + '/{}.txt'.format(item), 'r', encoding='utf-8')
    txt = file.read().strip('\n').strip(' ')
    Data.write(txt + '\n')
    file.close()

print('THU数据集合并完成')