import unittest

from gensim import corpora      #
from wordcloud import WordCloud
import gensim
import seaborn as sns
import pandas as pd
import stopwords as stopwords
import pandas as pd

# Matplot
import matplotlib.pyplot as plt


# Scikit-learn
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, LdaModel
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
# Keras
# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import collections

# Word2vec

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

if __name__ == '__main__':
    hair = pd.read_csv("hair_dryer.tsv", sep="\t", )        #
    microwave = pd.read_csv("microwave.tsv", sep="\t")   #
    pac = pd.read_csv("pacifier.tsv", sep="\t")  #
    #转换时间类型以及给予标记
    hair['review_date'] = pd.to_datetime(hair['review_date'],format="%m/%d/%Y")  #也就是得到了date的时间
    pac['review_date'] = pd.to_datetime(pac['review_date'],format="%m/%d/%Y")   #得到pac的时间
    microwave['review_date'] = pd.to_datetime(microwave['review_date'],format="%m/%d/%Y")
    stop_words = stopwords.words("english")
    wnl = nltk.WordNetLemmatizer()
    stemmer = SnowballStemmer("english")  #也就是
    tokens = []
    #预处理函数  进行停顿词删除与词根提取
    def preprocess(text):  # 训练可以加入别的东西，自己选择
        tokens = []
        words = text.split(' ')  # 也就是这个东西如果不是在stop_words中的话，进行一部分操作
        for token in words:  # for token in words,也就是如果不存在一个东西可以使用，
            token = token.lower()
            if token not in stop_words:
                tokens.append(stemmer.stem(token))
        return " ".join(tokens)
        # 得到了一堆列表的列表
    hair['review_body'] = hair['review_body'].apply(lambda x:preprocess(x))
    #已经生成了一些句子
    documents = [_test.split() for _test in hair.review_body]
    #词向量模型
    import wordcloud
    # 创建词云对象，赋值给w，现在w就表示了一个词云对象

    word_string=''
    for i in [_test for _test in hair.review_body]:
        word_string+=str(i)
    w = wordcloud.WordCloud(width=1000,height=700,background_color='white',font_path='msyh.ttc',max_words=50,collocations=True)
    w.generate(word_string)
    # 将生成的词云保存为output2-poem.png图片文件，保存到当前文件夹中
    w.to_file('temp.png')
    model = Word2Vec(documents,
                     size=200,  # 词向量维度
                     min_count=5,  # 词频阈值
                     window=5)  # 窗口大小   进行训练

    #构建字典
    dictionary = corpora.Dictionary(documents)      #dictionary
    dictionary.filter_tokens(None)
    id2word = dictionary.id2token
    dictionary.filter_extremes(no_below=2, no_above=0.1)
    #进行词向量分析
    corpus = [dictionary.doc2bow(text) for text in documents]
    num_topics = 20
    chunksize = 2000
    passes = 20
    iterations = 400
    #准备LDA模型
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           eval_every=None,workers=4)       #
    #按照相关度提取主题
    top_topics = lda_model.top_topics(corpus)  # , num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    from pprint import pprint
    pprint(top_topics)
    import pyLDAvis.gensim
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    # 需要的三个参数都可以从硬盘读取的，前面已经存储下来了

    pyLDAvis.show(vis)

































# dict1 = dict()
# for i in documents:
#     for j in i:
#         if j not in dict1:
#             dict1[j]=1
#         else:
#             dict1[j]+=1
# new_sys1 = sorted(dict1.items(), key=lambda d: d[1], reverse=False)
# print(new_sys1)
# w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
#                                             window=W2V_WINDOW,
#                                             min_count=W2V_MIN_COUNT,
#                                             workers=8)
#
# w2v_model.build_vocab(documents)
# words = w2v_model.wv
# print(words)
# vocab_size = len(words)
# print("Vocab size", vocab_size)
