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

train_valid_data = 'fitme_list_of_dict.pik' #data which is labeled as 'theft'
vocab_file = 'vocab.txt'
metadata_file = 'metadata.pkl' #缓存文件
evaluate_number=100 #how many data to evaluate.

def gen_metadata():
    vocab_li = []
    words_counter = {} #统计语料中总共有多少个词
    label_words_count = Counter() #统计词频-->词：频率

    with open(train_valid_data, 'rb') as fp:
        train_list, _, _ = pickle.load(fp)
        total_training_data=len(train_list)
        print("total_training_data:",total_training_data) #1,701,546
        dict_label_strings,prepro_label,labels = process_dialogue_list(train_list)
        countt=0
        for label,strings in dict_label_strings.items():
            cut_words_li = list(jieba.cut(strings)) #1.segmentation of string. a list.['的', '地方', '地方', '的', '方', '地方', '似懂非懂']
            words_counter[label] = len(cut_words_li)#2.count total words in this label
            label_words_count[label] = Counter(cut_words_li)#3.count frequency of words in this label #Counter({u'\u7684': 2, u'\u8bf4\u6cd5': 1, u'\u65f6\u4ee3': 1, u'\u4e86': 1, u'\u8428\u83f2': 1, u'\u5730\u65b9': 1, u'\u53d1\u751f': 1, u'\u7f57\u65af': 1, u'\u7406\u53d1\u5e97': 1})
            vocab_li.extend(list(set(cut_words_li))) #4.append to vocabulary. a list of uniqe words. [u'\u8bf4\u6cd5', u'\u65f6\u4ee3', u'\u7684', u'\u4e86', u'\u8428\u83f2', u'\u5730\u65b9', u'\u53d1\u751f', u'\u7f57\u65af', u'\u7406\u53d1\u5e97']

            print(countt, "->progress:", str((float(countt) / float(len(labels))) * 100) + "%")
            countt=countt+1
    vocab_li = list(set(vocab_li)) #5.get vocabulary list
    w2idx = {} #a dict for word to index
    idx2w = {} #a dict for index to word

    i = 0
    #保持词汇表到文件中，word to index,index to word to a dict
    vocab_fp = open(vocab_file, 'wb') #open vocab file
    for w in vocab_li:
        vocab_fp.write(w+'\n')
        w2idx[w] = i
        idx2w[i] = w
        i += 1
    vocab_fp.close()

    #print
    #print("words_counter:",words_counter)
    #key0=label_words_count.keys()[0]
    #kk=0
    #for word,freq in label_words_count[key0].items():
        #if kk>20:
         #   break
        #kk=kk+1
        #print(key0,word,freq)

    metadata = {
		'w2idx':w2idx,
		'idx2w':idx2w,
		'vocab_li':vocab_li,
		'words_counter':words_counter,
		'label_words_count':label_words_count, ##统计词频-->词：频率
        'prepro_label':prepro_label,
        'labels':labels
		}

    metadata_fp = open(metadata_file, 'wb') #open cache file
    pickle.dump(metadata, metadata_fp)
    metadata_fp.close()

#process dialogue list, put same label to a file
state_token='$$'
def process_dialogue_list(dialogue_list):#('total_training_data:', 1,701,546)
    print("process_dialogue_list.started")
    dict_label_lists={}
    prepro_label = {}
    #dialogue_list=dialogue_list[0:1000] #TODO
    total_training_data=len(dialogue_list) #1,701,546
    for i,dialogue in enumerate(dialogue_list):
        message=dialogue['message']
        label = dialogue['response'].strip()
        if i%100000==0:
            print(i,"label:",label)
        label = label.split("|")[0]
        state_token_flag=label.find(state_token) #remove anything after the special token.
        if state_token_flag>0:
            label=label[0:state_token_flag]
        #1.assign key-value(list) to dict_label_lists
        sub_list=dict_label_lists.get(label,None)
        if sub_list is None:
            dict_label_lists[label]=[message]
        else:
            sub_list.append(message)
            dict_label_lists[label] =sub_list

        #2.count frequency for each label
        freq_ = prepro_label.get(label, None)
        if freq_ == None:
            prepro_label[label] = 1
        else:
            prepro_label[label] = freq_ + 1

    dict_label_strings={}
    print("length of dict_label_lists:",len(dict_label_lists))
    for label,sub_list in dict_label_lists.items():
        dict_label_strings[label]=" ".join(sub_list)

    prepro_label = {x: float(prepro_label[x]) / total_training_data for x in prepro_label.keys()}

    labels=list(prepro_label.keys())
    #('length of dict_label_strings:', 131378, ';length of prepro_label:', 131378, ';length of labels:', 131378)
    print("length of dict_label_strings:",len(dict_label_strings),";length of prepro_label:",len(prepro_label),";length of labels:",len(labels))

    print("process_dialogue_list.ended")
    return dict_label_strings,prepro_label,labels


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
    print("eval_step.1.segment content,completed. length of content:",len(cut_words_li))
    print("eval_step.2.length of labels:",len(labels))
    for label in labels:
        #print("eval_step.2.label:",label)
        tmp_pro = prepro_label[label]
        for w in cut_words_li:
            #print("eval_step.3.word:",w)
            if w not in vocab_li:
                continue
            tmp_pro = tmp_pro * (label_words_count[label][w]+1)/(words_counter[label]+len_vocab) #为什么加len_vocab
        if tmp_pro > max_pro:
            max_pro = tmp_pro
            pred_label = label
    return pred_label

def evaluate():
    print("evaluate.started...")
    with open(train_valid_data, 'rb') as fp:
        _, valid_list, _ = pickle.load(fp)
        print("1.load valid data,finished.")
    valid_list=valid_list[0:evaluate_number]
    metadata = load_metadata()
    print("2.load meta data,finished.")

    precisian = 0.0
    count_right = 0
    count_all = 0

    len_valid_list=len(valid_list)
    for i,dialogue in enumerate(valid_list):
        print(str(float(i) / float(len_valid_list) * 100) + "%")

        content=dialogue['message']
        print("content:",content)

        true_label = dialogue['response'].strip()
        true_label = true_label.split("|")[0]
        state_token_flag=true_label.find(state_token) #remove anything after the special token.
        if state_token_flag>0:
            true_label=true_label[0:state_token_flag]

        prepro_label=metadata['prepro_label']
        labels=metadata['labels']
        pred_label = eval_step(metadata, labels, prepro_label, content)
        print ('True label:{},    Pred label:{}'.format(true_label, pred_label))
        print("accuracy update to now:",Decimal(count_right)/Decimal(count_all))
        if true_label == pred_label:
            count_right += 1
        count_all += 1


    precisian = Decimal(count_right)/Decimal(count_all)
    print ('Precisian   :', precisian)


#######################################################################
if __name__ == '__main__':
    #train()
    evaluate()

