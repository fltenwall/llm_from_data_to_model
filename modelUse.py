import os
import jieba
from gensim.models import word2vec

model = word2vec.Word2Vec.load(os.getcwd() + 'word2vec.model')

# 加载模型
def load_word2vec_model(w2v_path):
    model = word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语最相似的词
def calculate_most_similar(self, word):
    similar_words = self.wv.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


# 计算两个词相似度
def calculate_words_similar(self, word1, word2):
    print(self.wv.similarity(word1, word2))


# 找出不合群的词
def find_word_dismatch(self, list):
    print(self.wv.doesnt_match(list))
    
def sentence_to_vector(sentence, model):
    words = jieba.cut(sentence, cut_all=False)# 分词
    word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]  # 获取句子中每个词的词向量
    if len(word_vectors) == 0:
        return None  # 当句子中的词都不在词向量模型的词汇表中时返回None
    sentence_vector = sum(word_vectors) / len(word_vectors)  # 平均词向量作为句子向量
    return sentence_vector


print(model.vector_size)
# print(model.accuracy)
print(model.total_train_time)
# print(model.wv)
print(model.wv.most_similar('猫'))
print(model.wv.most_similar('吉林大学'))
print(sentence_to_vector('你有没有听说过侠客行的故事', model))