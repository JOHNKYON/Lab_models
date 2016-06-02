# -*- coding:utf-8 -*-
import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import conf
import data
import src

__author__ = 'JOHNKYON'

conf.jieba_conf.init()
print "init finished"

pg_conf = conf.pg_config

raw = src.pg.pg_select(pg_conf)

print "pg finished"

raw = [[x[0]+x[1], x[2]] for x in raw]

x = [[a[0]] for a in raw]

x = src.init.class_init_tf_idf(x)


y = [[(a[1]-10400000)/1000] for a in raw]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print "model train start"

clf = neighbors.KNeighborsClassifier(weights='distance', algorithm='auto')

print "mode train finished"

print "test start"
clf.fit(x_train, y_train)

print "test finished"

y_test = [x[0] for x in y_test]

precision = src.judgement.judgement(np.array(y_test), clf.predict(x_test))

print precision

# precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
#
# answer = clf.predict_proba(x)[:, 1]
# print(classification_report(y, answer, target_names=y))
