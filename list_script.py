# -*- coding: utf-8 -*-

import codecs
import re
import json

input_file = codecs.open("data/source_without_header.txt", 'rb', encoding='utf8')
output_file = codecs.open("data/major_dict.txt", 'wb', encoding='utf8')

major_re = re.compile(u"\d{6}")
counter = 0
l = dict()


for line in input_file.readlines():
    if major_re.match(line):
        l[str(counter)] = line
        # output_file.write(str(counter) + "\t" + line)
        counter += 1

j_list = json.dumps(l, ensure_ascii=False)
output_file.write(j_list)

input_file.close()
output_file.close()
