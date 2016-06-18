# -*- coding:utf-8 -*-  
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
import numpy as np
import conf
import src

__author__ = "JOHNKYON"


# 数据初始化
conf.jieba_conf.init()
print "init finished"

pg_conf = conf.pg_config

sql = """SELECT * FROM clean_person WHERE label = False LIMIT 3000"""

raw = src.pg.pg_select(pg_conf, sql)

sql = """SELECT * FROM clean_person WHERE label = True LIMIT 3000"""

raw += src.pg.pg_select(pg_conf, sql)

print "pg finished"

x = [[x[1], x[3], x[5], x[6], x[7], x[8]] for x in raw]

x = src.init.neural_init(x)

y = [a[9] for a in raw]

x = np.array(x)

# 1,0为True 0,1为False
y = np.array(map(src.init.label_init, y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

dim = len(x[0])

# 初始化完成
# TODO: 需要修改初始化过程使得输入源符合神经网络的输入方式。输出为二维

model = Sequential()

model.add(Dense(output_dim=64, input_dim=dim))
model.add(Activation("tanh"))
model.add(Dense(output_dim=2))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=5, show_accuracy=True, validation_split=0.2, shuffle=True)

print "fit finished"

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=30, show_accuracy=True)

print loss_and_metrics

