# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import jieba 
jieba.initialize()
from collections import Counter
from decimal import *
import cPickle as pickle

label_t = 'label_t.txt' #data which is labeled as 'theft'
label_other = 'label_other.txt' #data which is labeled as 'other'
vocab_file = 'vocab.txt'
metadata_file = 'metadata.pkl' #缓存文件

def gen_metadata():
	vocab_fp = open(vocab_file, 'wb') #open vocab file
	metadata_fp = open(metadata_file, 'wb') #open cache file
	vocab_li = []

	words_counter = {} #统计语料中总共有多少个词
	label_words_count = Counter() #统计词频-->词：频率

	label = 'theft'
	with open(label_t, 'rb') as fp:
		cut_words_li = list(jieba.cut(fp.read())) #a list.['的', '地方', '地方', '的', '方', '地方', '似懂非懂']
		words_counter[label] = len(cut_words_li)
		label_words_counter = Counter(cut_words_li)    #Counter({u'\u7684': 2, u'\u8bf4\u6cd5': 1, u'\u65f6\u4ee3': 1, u'\u4e86': 1, u'\u8428\u83f2': 1, u'\u5730\u65b9': 1, u'\u53d1\u751f': 1, u'\u7f57\u65af': 1, u'\u7406\u53d1\u5e97': 1})
		label_words_count[label] = label_words_counter #Counter({u'\u7684': 2, u'\u8bf4\u6cd5': 1, u'\u65f6\u4ee3': 1, u'\u4e86': 1, u'\u8428\u83f2': 1, u'\u5730\u65b9': 1, u'\u53d1\u751f': 1, u'\u7f57\u65af': 1, u'\u7406\u53d1\u5e97': 1})
		vocab_li.extend(list(set(cut_words_li))) # a list of uniqe words. [u'\u8bf4\u6cd5', u'\u65f6\u4ee3', u'\u7684', u'\u4e86', u'\u8428\u83f2', u'\u5730\u65b9', u'\u53d1\u751f', u'\u7f57\u65af', u'\u7406\u53d1\u5e97']

	label = 'non_theft'
	words_counter[label] = 0
	label_words_count[label] = Counter()
	with open(label_other, 'rb') as fp:
		for line in fp.readlines():
			content = ' '.join(line.split()[1:]) #a string
			cut_words_li = list(jieba.cut(content)) #get a list.each element is a word
			words_counter[label] = words_counter[label] + len(cut_words_li)
			label_words_counter = Counter(cut_words_li)	#Counter({u'\u7684': 2, u'\u8bf4\u6cd5': 1, u'\u65f6\u4ee3': 1, u'\u4e86': 1, u'\u8428\u83f2': 1, u'\u5730\u65b9': 1, u'\u53d1\u751f': 1, u'\u7f57\u65af': 1, u'\u7406\u53d1\u5e97': 1})
			label_words_count[label] = label_words_count[label] + label_words_counter
			vocab_li.extend(list(set(cut_words_li))) #add to vocab

	vocab_li = list(set(vocab_li)) #得到词汇表的列表

	w2idx = {} #a dict for word to index
	idx2w = {} #a dict for index to word
	i = 0
	#保持词汇表到文件中，word to index,index to word to a dict
	for w in vocab_li:
		vocab_fp.write(w+'\n')
		w2idx[w] = i
		idx2w[i] = w
		i += 1
	vocab_fp.close()
	
	metadata = {
		'w2idx':w2idx,
		'idx2w':idx2w,
		'vocab_li':vocab_li,
		'words_counter':words_counter,
		'label_words_count':label_words_count ##统计词频-->词：频率
		}

	pickle.dump(metadata, metadata_fp)
	metadata_fp.close()

def load_metadata():
	metadata = pickle.load(open(metadata_file, 'rb'))
	return metadata

def train():
	gen_metadata()

#预测
def eval_step(metadata, labels, prepro_label, content):
	words_counter = metadata['words_counter'] #统计语料中总共有多少个词
	label_words_count = metadata['label_words_count'] ##统计词频-->词：频率
	vocab_li = metadata['vocab_li'] #得到词汇表的列表
	len_vocab = len(vocab_li)
	
	max_pro = 0.0
	pred_label = ''
	cut_words_li = list(jieba.cut(content)) ##a list.['的', '地方', '地方', '的', '方', '地方', '似懂非懂']
	for label in labels:
		print("label:",label)
		tmp_pro = prepro_label[label]
		for w in cut_words_li:
			if w not in vocab_li:
				continue
			tmp_pro = tmp_pro * (label_words_count[label][w]+1)/(words_counter[label]+len_vocab) #为什么加len_vocab
		if tmp_pro > max_pro:
			max_pro = tmp_pro
			pred_label = label
	return pred_label

def evaluate():
	test_file = 'test.txt' #label:text
	labels =['theft', 'non_theft'] # ['theft', 'non_theft']
	prepro_label =  {'theft':0.5, 'non_theft':0.5} #{'theft':0.5, 'non_theft':0.5} #假设他们的类先验一样
	metadata = load_metadata()
	
	precisian = 0.0
	count_right = 0
	count_all = 0	

	with open(test_file, 'rb') as fp:
		for line in fp.readlines():
			line = line.strip()
			true_label = line.split()[0]
			content = ' '.join(line.split()[1:])
			pred_label = eval_step(metadata, labels, prepro_label, content)
			print ('True label:{},    Pred label:{}'.format(true_label, pred_label))
			if true_label == pred_label:
				count_right += 1
			count_all += 1
	
	precisian = Decimal(count_right)/Decimal(count_all)
	print ('Precisian   :', precisian)

			
#######################################################################
if __name__ == '__main__':
	train()
	evaluate()
	
