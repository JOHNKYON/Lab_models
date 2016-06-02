# -*- coding: utf-8 -*-
__author__ = 'zhiquan'

import re
import codecs

# input_file = codecs.open("data/source.txt", 'rb', encoding='utf8')

# 暂存无页眉页脚原始文本
# output_file = codecs.open("data/source_without_header.txt", 'wb', encoding='utf8')

# 去除页眉页脚
'''page_header_re = re.compile(u"普通高等.*专业介绍")
for line in input_file.readlines():
    if page_header_re.search(line):
        continue
    output_file.write(line)'''

# 按照学科大类分类存储
'''input_file = codecs.open("data/source_without_header.txt", 'rb', encoding='utf8')
counter = 0
output_file = codecs.open("data/genre/"+str(counter)+".txt", 'wb', encoding='utf8')
genre_re = re.compile(u"(\d\d)\s*学科门类")
for line in input_file.readlines():
    if genre_re.match(line):
        output_file.close()
        counter += 1
        if counter == 11:
            counter += 1
        output_file = codecs.open("data/genre/"+str(counter)+".txt", 'wb', encoding='utf8')
    output_file.write(line)'''

# 各学科小类别存储
index = range(1, 14)
counter2 = 0    # 用于存储小类编号
# 小类别的正则表达式
class_re = re.compile(u"\d\d\d\d *.*类")
# 各专业的正则表达式
major_re = re.compile(u"\d\d\d\d\d\d")
# 初始化状态
output_file = codecs.open("data/genre/class/1_0.txt", 'wb', encoding='utf8')

# 遍历每个大类
for counter in index:
    counter2 = 0
    # counter3用于去除小类文件头的大类信息
    # 重置counter3
    counter3 = 0
    if counter == 11:
        continue
    input_file = codecs.open("data/genre/"+str(counter)+".txt", 'rb', encoding='utf8')
    for line in input_file.readlines():
        if counter3 == 0:
            counter3 += 1
            continue
        if class_re.match(line):
            output_file.close()
            counter2 += 1
            output_file = codecs.open("data/genre/class/"+str(counter)+'_'+str(counter2)+'.txt', 'wb', encoding='utf8')
            # 此处不执行是为了不记录小类名称信息，方便下一步分词处理
            continue
        # 每两个专业之间空两行
        if major_re.match(line):
            output_file.write('\n\n')
        output_file.write(line)
    input_file.close()


'''output_file = codecs.open("data/data_for_lsi.txt", 'wb', encoding='utf8')
# 为了方便处理而仅去除了大类信息的
for counter in index:
    # counter3用于去除小类文件头的大类信息
    # 重置counter3
    counter3 = 0
    if counter == 11:
        continue
    input_file = codecs.open("data/genre/"+str(counter)+".txt", 'rb', encoding='utf8')
    for line in input_file.readlines():
        if counter3 == 0:
            counter3 += 1
            continue
        if class_re.match(line):
            continue
        output_file.write(line)
    input_file.close()'''
