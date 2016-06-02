# -*- coding:utf-8 -*-  
import re
import jieba
from data import stopwords
import algorithm_collection
import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

__author__ = "JOHNKYON"


def text_init(raw):
    """
    去除文本中的停用词，分词
    :param raw:
    :return 分词后的list,list中的元素为2级list,2级list中的元素为词。此时1级list中元素已经是未数字化的词向量:
    """

    # 去除空格符
    raw_without_space = re.sub(' *', '', raw)
    # 将不同的专业分开作list元素
    # 专业号的正则表达式
    major_re = re.compile(u"\d{6}\D")

    '''# 用于测试匹配数量不一致问题
    out_test = codecs.open("data/re_test.txt", 'wb', encoding='utf8')
    re_find = major_re.findall(raw_without_space)
    counter = 0
    for ele in re_find:
        out_test.write(str(counter) + '\t' + ele + '\n')
        counter += 1
    out_test.close()'''

    # 对整个字符串进行切片,分割符为此正则表达
    raw_splited = major_re.split(raw_without_space)[1:]

    # 分词
    # 载入自定义词典
    jieba.load_userdict("data/jieba_dict.txt")
    raw_cut = map(lambda x: jieba.cut(x, cut_all=False), raw_splited)
    # 去除停用词
    raw_without_sw = map(lambda x: filter(lambda y: y not in stopwords, x), raw_cut)
    return raw_without_sw


def tSNE_init(raw, topics, labels):
    """
    针对t-SNE算法进行数据初始化，使数据格式符合scikit-learn的t-SNE模块输入
    暂定使用主题向量对计算初始值
    :param raw:
    :return:
    """

    # TODO:按照第二级目录进行标注颜色用于区分

    # output_file = codecs.open('temp/temp.txt', 'wb', 'utf8')

    # 文本本身处理，去除空白符，去除停用词
    dic_corpus = text_digitalize(raw)

    dictionary = dic_corpus[0]
    corpus = dic_corpus[1]

    # 用tfidf训练
    corpus_tfidf = algorithm_collection.build_tfidf(corpus)

    # 训练lsi模型
    lsi_model = algorithm_collection.build_lsi(corpus_tfidf, dictionary, topics)

    corpus_lsi = lsi_model[corpus_tfidf]

    mtr = [[y[1] for y in x] for x in corpus_lsi]

    # for ele in mtr:
    #     for x in ele:
    #         output_file.write(str(x))
    #         output_file.write('\t')
    #     output_file.write('\n')
    #
    # output_file.close()

    mtr = np.array(mtr)

    label = np.transpose(np.array(['#' + str(hex(np.square(long(str(x / 1000)[1:])) * 90))[2:-1] for x in labels]))
    # label = np.transpose(np.array(['#' + str(hex(np.sqrt(long(str(x / 100000)[1:])) * 2948576))[2:-1] for x in raw]))
    print label

    return mtr, label


def tSNE_init_test(raw):
    X = np.vstack([raw.data[raw.target == i]
                   for i in range(10)])
    y = np.hstack([raw.target[raw.target == i]
                   for i in range(10)])
    return X, y


def class_init_tf_idf(raw):
    """
    用于将原始文本转换为tf-idf表示的矩阵
    :param raw:
    :return:
    """
    dic_corpus = text_digitalize(raw)
    dictionary = dic_corpus[0]
    corpus = dic_corpus[1]

    dimension = max(dictionary.token2id.values())
    print len(corpus)

    # transformer = TfidfTransformer()

    # 用tfidf训练
    corpus_tfidf = algorithm_collection.build_tfidf(corpus)
    # for ele in corpus_tfidf:
    #     print ele

    print max(dictionary.token2id.values())

    # 建立并初始化tfidf矩阵
    arr = tfidf_matrix(dictionary, corpus, dic_corpus)

    return arr


def text_digitalize(raw):
    """
    将文本初始化并数字化
    :param raw:
    :return:
    """
    raw_without_space = map(lambda x: [re.sub('\s*', '', x[0])], raw)

    raw_without_space = [x[0] for x in raw_without_space]

    # temp = codecs.open("temp/corpus.txt", 'wb', encoding='utf8')

    jieba.load_userdict("data/jieba_dict.txt")
    raw_cut = [jieba.cut(x, cut_all=False) for x in raw_without_space]

    # for ele in raw_cut:
    #     print ele[0][0]

    # for ele in raw_cut:
    #     for a in ele:
    #         temp.write(a+'\t')
    #     temp.write('\n')
    #
    # temp.close()

    raw_without_sw = map(lambda x: [filter(lambda y: y not in stopwords, x)], raw_cut)

    # for ele in raw_doc:
    #     print ele[0][0]

    raw_doc = [x[0] for x in raw_without_sw]
    dic_corpus = algorithm_collection.digitalize(raw_doc)
    return dic_corpus


def tfidf_matrix(dictionary, corpus, corpus_tfidf):
    """
    用于将tfidf信息转化为矩阵
    :param corpus_tfidf:
    :return:
    """
    arr = np.zeros([len(corpus), max(dictionary.token2id.values())], dtype='float64')

    counter = 0
    for line in corpus_tfidf:
        for ele in line:
            arr[counter][ele[0] - 1] = ele[1]
        counter += 1

    return arr