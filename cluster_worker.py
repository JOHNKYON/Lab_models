# -*- coding: utf-8 -*-

import src
import codecs
import conf

# 读入文件
input_file = codecs.open("data/source_without_header.txt", 'rb', encoding='utf8')
raw = input_file.read()

# 分词
conf.jieba_conf.init()
pg_conf = conf.pg_config
# raw = src.pg.pg_select()

text_list = src.init.text_init(raw)

# 生成文档向量
dic_corpus = src.algorithm_collection.digitalize(text_list)
dictionary = dic_corpus[0]
corpus = dic_corpus[1]

# 用tfidf训练
corpus_tfidf = src.algorithm_collection.build_tfidf(corpus)

# 训练lsi模型
lsi = src.algorithm_collection.build_lsi(corpus_tfidf, dictionary)

# 获取每个major的bucket_id及匹配度
bucket = src.algorithm_collection.topic_cluster(lsi, corpus_tfidf)

src.pg.pg_insert(bucket, pg_conf)

# 获取索引
index = src.algorithm_collection.lsi_index(lsi, corpus)

# 用索引和语料库生成关联度矩阵
sim_matrix = src.algorithm_collection.sim_matrix(index)

# 聚类结果查询
src.querry.form_bucket(sim_matrix)

# 储存关联度矩阵
output_file = codecs.open("data/sim_matrix.txt", 'wb', encoding='utf8')

for ele in sim_matrix:
    for m in ele:
        output_file.write('('+str(m[0])+',\t'+str(m[1])+')\t')
    output_file.write('\n')

output_file.close()
input_file.close()
