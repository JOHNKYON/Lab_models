# -*- coding:utf-8 -*-  
import src
import conf
from sklearn.datasets import load_digits

__author__ = "JOHNKYON"

# 分词
conf.jieba_conf.init()


pg_conf = conf.pg_config

raw = src.pg.pg_select(pg_conf)
# raw = load_digits()
# raw.data.shape

print 'pg finished'

print 'split finished'

raw = [[x[0]+x[1], x[2]] for x in raw]

labels = [x[1] for x in raw]

mtr, label = src.init.tSNE_init(raw, 23, labels)
# mtr, label = src.init.tSNE_init_test(raw)

print 'init finished'

result = src.t_SNE.plot_build(mtr)

print 'build finished'

src.image_build.imamge_2D(result, label, 23)


