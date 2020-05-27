#!/bin/bash
set -e 

data_dir="F:\徐佳明量子物理\matlab_data"

# 整理数据
bash search.sh $data_dir png > data.lst # linux

cat data.lst | awk -F "/" '{printf("%s %s\n",$0,$(NF-1))}' > data.label
cat data.label | awk 'BEGIN{i=-1;labels[null]=""};{if(!($NF in labels)){labels[$NF]++;i++};printf("%s %s\n",$1,i);}' > data.label.num

# 划分 训练/开发/测试 集合
cat data.label.num | shuf -n 200 > data.dev
cat data.label.num | shuf -n 200 > data.test
cat data.label.num | grep -v -f data.dev -f data.test > data.train

# 训练
