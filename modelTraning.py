from gensim.models import Word2Vec
from gensim.models import word2vec
from loguru import logger
import time
import os
from gensim.models.callbacks import CallbackAny2Vec

# 定义回调函数类，用于在每个epoch结束时记录训练信息
class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.time = time.time()

    def on_epoch_begin(self, model):
        logger.info("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        duration = time.time() - self.time
        logger.info("Epoch #{} end, loss: {}, duration: {}".format(self.epoch, loss, duration))
        model.save('word2vec_model_{}.model'.format(self.epoch))  # 保存模型，模型名中包含epoch数
        self.epoch += 1
        self.time = time.time()

# 打印日志
logger.add("word2vec_training.log")

path = os.getcwd() + "/llm-corpus/data/data.txt"

logger.info('Starting to load sentences from %s', path)
sentences = word2vec.LineSentence(path)
logger.info('Finished loading sentences')

# 提前定义callback
epoch_logger = EpochLogger()

# sg——word2vec两个模型的选择。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型
# hs——word2vec两个解法的选择，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling
# negative——即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间
# min_count——需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
# iter——随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
# alpha——在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025
# min_alpha——由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值
# 训练模型
logger.info('Starting to build Word2Vec model')
model = Word2Vec(sentences, vector_size=300, window=5, epochs=10, compute_loss=True, callbacks=[epoch_logger])
logger.info('Finished building Word2Vec model')
model_path = 'word2vec.model'
logger.info('Starting to save model to %s', model_path)
model.save(model_path)
logger.info('Finished saving model')
