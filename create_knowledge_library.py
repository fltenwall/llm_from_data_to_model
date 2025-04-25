import os
import time
from typing import List, Union, Dict

# from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import jieba
import numpy as np
from loguru import logger
from numpy import ndarray
from tqdm import tqdm

stopwords_table_path = os.getcwd() + "/llm-corpus/data/四川大学机器智能实验室停用词库.txt"


def load_stopwords(file_path):
    # 读取停词表，并使用set来存储
    with open(stopwords_table_path, 'r', encoding='utf-8') as file:
        stopwords_set = set(line.strip() for line in file)
        return stopwords_set


class Word2VecManager:
    """Pre-trained word2vec embedding"""
    def __init__(self, model_name_or_path: str = './word2vec.model',
                 w2v_kwargs: Dict = None,
                 stopwords: List[str] = None):
        """
        Init word2vec model

        Args:
            model_name_or_path: word2vec file path
            w2v_kwargs: dict, params pass to the ``load_word2vec_format()`` function of ``gensim.models.KeyedVectors`` -
                https://radimrehurek.com/gensim/models/keyedvectors.html#module-gensim.models.keyedvectors
            stopwords: list, stopwords
        """
        from gensim.models import KeyedVectors  # noqa

        self.w2v_kwargs = w2v_kwargs if w2v_kwargs is not None else {}

        t0 = time.time()
        # w2v.init_sims(replace=True)
        logger.debug('Load w2v from {}, spend {:.2f} sec'.format(model_name_or_path, time.time() - t0))
        self.stopwords = stopwords if stopwords else load_stopwords(default_stopwords_file)
        self.model = Word2Vec.load(model_name_or_path)
        self.w2v = self.model.wv
        self.jieba = jieba
        self.model_name_or_path = model_name_or_path

    def __str__(self):
        return f"<Word2Vec, word count: {len(self.w2v.key_to_index)}, emb size: {self.w2v.vector_size}, " \
               f"stopwords count: {len(self.stopwords)}>"

    def encode(self, sentences: Union[List[str], str], show_progress_bar: bool = False) -> ndarray:
        """
        Encode sentences to vectors
        """
        if self.w2v is None:
            raise ValueError('No model for embed sentence')

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        for sentence in tqdm(sentences, desc='Word2Vec Embeddings', disable=not show_progress_bar):
            emb = []
            count = 0
            for word in sentence:
                # 过滤停用词
                if word in self.stopwords:
                    continue
                # 调用词向量
                if word in self.w2v.key_to_index:
                    emb.append(self.w2v.get_vector(word, norm=True))
                    count += 1
                else:
                    if len(word) == 1:
                        continue
                    # 再切分
                    ws = self.jieba.lcut(word, cut_all=True, HMM=True)
                    for w in ws:
                        if w in self.w2v.key_to_index:
                            emb.append(self.w2v.get_vector(w, norm=True))
                            count += 1
            tensor_x = np.array(emb).sum(axis=0)  # 纵轴相加
            if count > 0:
                avg_tensor_x = np.divide(tensor_x, count)
            else:
                avg_tensor_x = np.zeros(self.w2v.vector_size, dtype=float)
            all_embeddings.append(avg_tensor_x)
        all_embeddings = np.array(all_embeddings, dtype=float)
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

stop_word_set = load_stopwords(stopwords_table_path)              
# model = Word2VecManager(stopwords=stop_word_set)
# model.encode("你好我是中国人")

def split_sentences(text):
    sent_delimiters = ['。', '？', '！', '?', '!', '.']
    for delimiter in sent_delimiters:
        text = text.replace(delimiter, '\n')
    sentences = text.split('\n')
    sentences = [sent for sent in sentences if sent.strip()]
    return sentences

text = '欢迎使用结巴中文分词！请问有什么可以帮助您的吗？'
split_sentences(text)

# model = word2vec.Word2Vec.load(os.getcwd() + 'word2vec.model')
model = Word2VecManager(stopwords=stop_word_set)

# 加载模型
def load_word2vec_model(w2v_path):
    model = Word2vec.Word2Vec.load(w2v_path)
    return model


# 计算词语最相似的词
def calculate_most_similar(self, word):
    similar_words = self.wv.most_similar(word)
    print(word)
    for term in similar_words:
        print(term[0], term[1])


# 计算两个词相似度
def calculate_words_similar(self, word1, word2):
    print(word1, word2)
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

# 以下是系统的知识库
def create_corpus(doc):
    sents = split_sentences(doc)
    corpus = []
    corpus_embeddings = []
    for sent in sents:
        vc = model.encode(sent)
        corpus.append(sent)
        corpus_embeddings.append(vc)
    return corpus, corpus_embeddings

def create_corpus_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        doc = file.read()
    return create_corpus(doc)

def semantic_search(query_embedding, corpus_embeddings, top_k=5):
    similarities = []
    for doc_embedding in corpus_embeddings:
        print(type(doc_embedding))
        similarities.append(calculate_words_similar(query_embedding, doc_embedding))
    # similarities = [calculate_words_similar(query_embedding, doc_embedding) for doc_embedding in corpus_embeddings]
    sorted_indexes = np.argsort(similarities)[::-1]
    top_k_indexes = sorted_indexes[:top_k]
    return top_k_indexes, np.array(similarities)[top_k_indexes]
    

# 以下是用户的问题
queries = ["《铃芽户缔》剧情有几条线？"]
corpus, corpus_embeddings = create_corpus_from_file("suzume.md")

for query in queries:
    query_embedding = model.encode(query)
    # 将问题通过模型从知识库匹配，取前3条
    hit_indexes, hit_similarities = semantic_search(query_embedding, corpus_embeddings, top_k=5)
    print("\n问题:", query, "\n最优前5条:")
    for hit_index, hit_similarity in zip(hit_indexes, hit_similarities):
        print(corpus[hit_index], "({:.2f})".format(hit_similarity))
