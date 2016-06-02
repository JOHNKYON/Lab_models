# -*- coding:utf-8 -*-  
import codecs
import json

__author__ = "JOHNKYON"

input_file = codecs.open("data/major.json", 'rb', encoding='utf8')
output_file = codecs.open("data/major_dict.json", 'wb', encoding='utf8')

counter = -1

dic = dict()

for line in input_file.readlines():
    if counter == -1:
        counter += 1
        continue
    if counter == 505:
        dic[counter] = line[3:-2]
        counter += 1
    else:
        dic[counter] = line[3:-3]
        counter += 1

j_dict = json.dumps(dic, ensure_ascii=False)
output_file.write(j_dict)

input_file.close()
output_file.close()

