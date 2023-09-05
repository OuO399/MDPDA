import copy
import pickle
import numpy as np
import sklearn
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix,roc_auc_score
from dbn import UnsupervisedDBN,AbstractSupervisedDBN,NumPyAbstractSupervisedDBN,SupervisedDBNClassification
import pandas as pd
import os
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler as  ROS
import threading
from multiprocessing import Process
import time
import random
from save_for_glove import save_train_set_tokens_in_txt_by_version
from get_token_embeding import TokenEmbedding

import keras
from keras.layers import Input,Embedding,LSTM,Dense,Activation,Multiply, Masking
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from keras.backend import clear_session
import tensorflow as tf
from sklearn import linear_model, manifold
from sklearn.utils import compute_class_weight
import math



def crop_or_fill(list1,list2,cut_length,fix_length=0):
    max_length = 0
    new_list = list1+list2
    for i in new_list:
        if(len(i)>max_length):
            max_length = len(i)
        # if(len(i) == 7063):
        #     print(key)
    print(max_length)

    # 将数据全都裁剪或填充到固定长度
    fix_length = int(max_length*cut_length)
    print("fix_length={}".format(fix_length))
    for i in range(len(list1)):
        if len(list1[i]) > fix_length:
            list1[i] = list1[i][:fix_length]
        elif len(list1[i]) < fix_length:
            length = len(list1[i])
            for _ in range(fix_length-length):
                list1[i].append([0 for j in range(40)])
    for i in range(len(list2)):
        if len(list2[i]) > fix_length:
            list2[i] = list2[i][:fix_length]
        elif len(list2[i]) < fix_length:
            length = len(list2[i])
            for _ in range(fix_length-length):
                list2[i].append([0 for j in range(40)])

    return list1,list2


def get_token_and_labels_before_embedding(project_name,version):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,version))
    # print(df[df['bugs']==1])
    label_dict = {}
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df.iloc[i]["bugs"] == 0:
            label_dict[df.iloc[i]["name"]+".java"] = 0
        else:
            label_dict[df.iloc[i]["name"]+".java"] = 1
    # print(label_dict)
    with open("../numtokens_with_GloVe/{}_{}_without_mutation_before_embedding.pkl".format(project_name,version),"rb") as f:
        num_tokens_dict = pickle.load(f)

    return num_tokens_dict, label_dict

def pre_process(project_name,version,cut_length):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,version))
    # print(df[df['bugs']==1])
    label_dict = {}
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df.iloc[i]["bugs"] == 0:
            label_dict[df.iloc[i]["name"]+".java"] = 0
        else:
            label_dict[df.iloc[i]["name"]+".java"] = 1
    # print(label_dict)
    with open("../numtokens_with_GloVe/{}_{}_without_mutation_before_embedding.pkl".format(project_name,version),"rb") as f:
        num_tokens_dict = pickle.load(f)
        max_length = 0
        for key,i in num_tokens_dict.items():
            if(len(i)>max_length):
                max_length = len(i)
            # if(len(i) == 7063):
            #     print(key)
        print(max_length)


        # # 将所有数据填充到最长数据的长度
        # for i in num_tokens_dict.keys():
        #     for _ in range(max_length-len(num_tokens_dict[i])):
        #         num_tokens_dict[i].append(0)

        # 将数据全都裁剪或填充到固定长度
        fix_length = int(max_length*cut_length)
        for i in num_tokens_dict.keys():
            if(len(num_tokens_dict[i])>fix_length):
                num_tokens_dict[i] = num_tokens_dict[i][:fix_length]
            else:
                length = len(num_tokens_dict[i])
                for _ in range(fix_length-length):
                    num_tokens_dict[i].append(0)

        

        # for i in file_names:
        #     try:
        #         labels.append(label_dict[i])
        #         DBN_data.append(np.array(num_tokens_dict[i]))
        #     except KeyError:
        #         continue

        # print(np.array((DBN_data[1])).shape)
        # print(labels)
        return num_tokens_dict,label_dict,str(int(cut_length*100))+"%"

def train_test_split(project,version,data_dict,labels_dict,test_size,random_state):
    file_num = len(labels_dict)
    if (os.path.exists("../train_test_index/{}_{}_{}_{}.pkl".format(project,version,test_size,random_state))):
        with open("../train_test_index/{}_{}_{}_{}.pkl".format(project,version,test_size,random_state),"rb")as f:
            test_index_list = pickle.load(f)
    else:
        # 随机选取指定大小的下标列表，下标对应的文件作为测试集
        # test_index = set()
        test_file_num = int(file_num * test_size)
        test_index_list = random.sample(range(0,file_num),test_file_num)
        # while len(test_index) < test_file_num:
        #     random_num = random.randint(0, file_num - 1)
        #     test_index.add(random_num)
        #     print(len(test_index))
        # test_index_list = list(test_index)
        with open("../train_test_index/{}_{}_{}_{}.pkl".format(project, version, test_size, random_state), "wb") as f:
            pickle.dump(test_index_list,f)
    train_X = {}
    train_Y = {}
    test_X = {}
    test_Y = {}
    for i in test_index_list:
        file_names = list(labels_dict.keys())
        try:
            test_X[file_names[i]] = data_dict[file_names[i]]
            test_Y[file_names[i]] = labels_dict[file_names[i]]
        except KeyError:
            continue
    for i in range(file_num):
        file_names = list(labels_dict.keys())
        if i not in test_index_list:
            try:
                train_X[file_names[i]] = data_dict[file_names[i]]
                train_Y[file_names[i]] = labels_dict[file_names[i]]
            except KeyError:
                continue
    print("train_X length:{}".format(len(train_X)))
    print("train_Y length:{}".format(len(train_Y)))
    print("test_X length:{}".format(len(test_X)))
    print("test_Y length:{}".format(len(test_Y)))
    return train_X,test_X,train_Y,test_Y


def save_without_mutation_file(project_name,train_version,test_version,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    save_train_set_tokens_in_txt_by_version(project_name,train_version,test_version,train_X_dict)


def save_manual_mutation_file_used_in_oversample(project_name,train_version,test_version,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    train_X_dict = copy.deepcopy(train_X_dict)
    train_Y_dict = copy.deepcopy(train_Y_dict)
    with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project_name,train_version),"rb") as f:
        num_tokens_dict_with_manual_mutation = pickle.load(f)

    train_tokens_dict_with_manual_mutation = {}
    for i in num_tokens_dict_with_manual_mutation.keys():
        str1 = i.split("_")[0]
        if str1 in train_X_dict.keys():
            train_tokens_dict_with_manual_mutation[i] = num_tokens_dict_with_manual_mutation[i]
    print("num_tokens_dict_with_manual_mutation_length = {}".format(len(num_tokens_dict_with_manual_mutation)))
    print("train_tokens_dict_with_manual_mutation_length = {}".format(len(train_tokens_dict_with_manual_mutation)))
    num_tokens = train_X_dict
    train_Y_list = []
    for i in train_Y_dict.values():
        train_Y_list.append(i)
    train_Y = np.array(train_Y_list)
    num = np.sum(train_Y==0)-np.sum(train_Y==1)

    random.seed(3618)
    
    if (len(train_tokens_dict_with_manual_mutation)<num) :
        numlist = [i for i in range(len(train_tokens_dict_with_manual_mutation))]
    else:
        numlist = random.sample(range(0,len(train_tokens_dict_with_manual_mutation)-1),num)
    add_tokens = {}
    for i in numlist:
        key = list(train_tokens_dict_with_manual_mutation.keys())[i]
        token_list = train_tokens_dict_with_manual_mutation[key]
        num_tokens[key] = token_list
        add_tokens[key] = token_list
        train_Y_dict[key] = 1
    save_train_set_tokens_in_txt_by_version(project_name,train_version,test_version,add_tokens)
    with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file.pkl".format(project_name,train_version),"wb") as f:
        pickle.dump(num_tokens,f)
    return train_Y_dict



#划分测试集与训练集，并将训练集token序列存入txt中，供glove生成词向量表
def save_for_glove(projects,cut_length):  
    train_X_dict = {}
    train_Y_dict = {}
    test_X_dict = {}
    test_Y_dict = {}
    for project in projects_with_version.keys():
        # 获取每个项目版本号
        project_path = "../PROMISE/promise_data/{}".format(project)
        versions = projects_with_version[project]
        # for root,dirnames,filenames in os.walk(project_path):
        #     for i in filenames:
        #         versions.append(i[0:-4])
        # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
        for i in range(len(versions)-1):
            train_version = versions[i]
            test_version = versions[i+1]
            train_X_dict_before_embedding,train_Y_dict_before_embedding = get_token_and_labels_before_embedding(project,train_version)
            test_X_dict_before_embedding,test_Y_dict_before_embedding = get_token_and_labels_before_embedding(project,test_version)
            # data_dict_before_embedding,labels_dict = get_token_and_labels_before_embedding(project,version)
            # train_X,test_X,train_Y,test_Y = train_test_split(project,version,data_dict_before_embedding,labels_dict,test_size=0.3,random_state = 10)
            save_without_mutation_file(project,train_version,test_version,train_X_dict_before_embedding,train_Y_dict_before_embedding,test_X_dict_before_embedding,test_Y_dict_before_embedding)
            # train_Y_with_mutation = save_mutation_file_used_in_oversample(project,version,train_X,train_Y,test_X,test_Y)
            train_Y_with_manual_mutation = save_manual_mutation_file_used_in_oversample(project,train_version,test_version,train_X_dict_before_embedding,train_Y_dict_before_embedding,test_X_dict_before_embedding,test_Y_dict_before_embedding)

            with open("./train_data_with_glove/{}_{}_train_X_{}.pkl".format(project,train_version,cut_length),"wb") as f :
                pickle.dump(train_X_dict_before_embedding,f)

            with open("./train_data_with_glove/{}_{}_without_mutation_train_Y_{}.pkl".format(project,train_version,cut_length),"wb") as f :
                pickle.dump(train_Y_dict_before_embedding,f)

            # with open("./train_data_with_glove/{}_{}_with_mutation_train_Y_{}.pkl".format(project,version,cut_length),"wb") as f :
            #     pickle.dump(train_Y_with_mutation,f)

            with open("./train_data_with_glove/{}_{}_with_manual_mutation_train_Y_{}.pkl".format(project,train_version,cut_length),"wb") as f :
                pickle.dump(train_Y_with_manual_mutation,f)

            with open("./test_data_with_glove/{}_{}_test_X_{}.pkl".format(project,test_version,cut_length),"wb") as f :
                pickle.dump(test_X_dict_before_embedding,f)

            with open("./test_data_with_glove/{}_{}_test_Y_{}.pkl".format(project,test_version,cut_length),"wb") as f :
                pickle.dump(test_Y_dict_before_embedding,f)

    #         train_X_dict['{}_{}'.format(project,version)] = train_Xs
    #         train_Y_dict['{}_{}_without_mutation'.format(project,version)] = train_Y
    #         train_Y_dict['{}_{}_with_mutation'.format(project,version)] = train_Y_with_mutation
    #         train_Y_dict['{}_{}_with_manual_mutation'.format(project,version)] = train_Y_with_manual_mutation
    #         test_X_dict['{}_{}'.format(project,version)] = test_X
    #         test_Y_dict['{}_{}'.format(project,version)] = test_Y
    # return train_X_dict,test_X_dict,train_Y_dict,test_Y_dict

def handle_batch(projects_with_version,cut_length):
    Processes = []  
    save_for_glove(projects_with_version,cut_length)
    os.system('bash ../GloVe/demo.sh')


if __name__ == '__main__':
    # pre_process("lucene",'2.4')
    # projects = ['ant','ivy','jEdit']
    # projects_with_version = {'ant':['1.6','1.7'],"jEdit":['4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],"ivy":['1.4','2.0']}
    # projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
    #                             "ivy":['1.4','2.0'],"poi":['2.0','2.5'],"xalan":['2.4','2.5'],"xerces":['1.2','1.3'],"log4j":["1.0","1.1"]}
    # projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
    #                             "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
    projects_with_version = {"ivy":['1.4','2.0']}
    # projects_with_version = {"synapse":['1.0','1.1','1.2'],'xalan':['2.4','2.5']}
    # projects = ['ivy']
    # projects = ['ant']
    save_for_glove(projects_with_version,1.0)
    os.system('bash ../GloVe/demo.sh')
    # handle_batch(projects_with_version,0.1)
    # handle_batch("ant")