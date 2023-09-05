import pickle
import numpy as np
import sklearn
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from dbn import UnsupervisedDBN,AbstractSupervisedDBN,NumPyAbstractSupervisedDBN,SupervisedDBNClassification
import pandas as pd
import os
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler as  ROS
from imblearn.over_sampling import SMOTE,KMeansSMOTE
import threading
from multiprocessing import Process,Pool,cpu_count
import time
import random

def change_length(num_tokens_dict,fix_length):

    for i in num_tokens_dict.keys():
        if(len(num_tokens_dict[i])>fix_length):
            num_tokens_dict[i] = num_tokens_dict[i][:fix_length]
        else:
            length = len(num_tokens_dict[i])
            for _ in range(fix_length-length):
                num_tokens_dict[i].append(0)
    return num_tokens_dict

        
def pre_process(project_name,train_version,test_version,cut_length):
    df_train = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,train_version))
    df_test = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name,test_version))
    # print(df[df['bugs']==1])
    train_label_dict = {}
    test_label_dict = {}
    for i in range(len(df_train)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df_train.iloc[i]["bugs"] == 0:
            train_label_dict[df_train.iloc[i]["name"]+".java"] = 0
        else:
            train_label_dict[df_train.iloc[i]["name"]+".java"] = 1

    for i in range(len(df_test)):
        if df_test.iloc[i]["bugs"] == 0:
            test_label_dict[df_test.iloc[i]["name"]+".java"] = 0
        else:
            test_label_dict[df_test.iloc[i]["name"]+".java"] = 1

    # print(label_dict)
    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,train_version),"rb") as f:
        train_num_tokens_dict = pickle.load(f)
        train_max_length = 0
        for key,i in train_num_tokens_dict.items():
            if(len(i)>train_max_length):
                train_max_length = len(i)
            # if(len(i) == 7063):
            #     print(key)
        # print(max_length)

    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,test_version),"rb") as f:
        test_num_tokens_dict = pickle.load(f)
        test_max_length = 0
        for key,i in test_num_tokens_dict.items():
            if(len(i)>test_max_length):
                test_max_length = len(i)


        # # 将所有数据填充到最长数据的长度
        # for i in num_tokens_dict.keys():
        #     for _ in range(max_length-len(num_tokens_dict[i])):
        #         num_tokens_dict[i].append(0)

        # 将数据全都裁剪或填充到固定长度
        fix_length = max(train_max_length,test_max_length)
        fix_length = int(fix_length*cut_length)

        train_num_tokens_dict = change_length(train_num_tokens_dict,fix_length)
        test_num_tokens_dict = change_length(test_num_tokens_dict,fix_length)

        

        # for i in file_names:
        #     try:
        #         labels.append(label_dict[i])
        #         DBN_data.append(np.array(num_tokens_dict[i]))
        #     except KeyError:
        #         continue

        # print(np.array((DBN_data[1])).shape)
        # print(labels)
        return train_num_tokens_dict,test_num_tokens_dict,train_label_dict,test_label_dict,str(int(cut_length*100))+"%",fix_length

def without_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("without_oversample start")
    start_time = time.time()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)
    print(project+train_version)
    print(len(train_X))
    print(np.sum(train_Y == 1)/len(train_Y))
    print("-------------------------------------------")
    print(project+test_version)
    print(len(test_X))
    print(np.sum(test_Y == 1)/len(test_Y))
    print("============================================")

    # dbn_train(project,train_version,test_version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("without_oversample end")

def mutation_oversample(project_name,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("mutation_oversample start")
    with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name, version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
    train_tokens_dict_with_mutation = {}
    for i in num_tokens_dict_with_mutation.keys():
        str1 = i.split("_")[0]
        if str1 in train_X_dict.keys():
            train_tokens_dict_with_mutation[i] = num_tokens_dict_with_mutation[i]
    print("num_tokens_dict_with_mutation_length = {}".format(len(num_tokens_dict_with_mutation)))
    print("train_tokens_dict_with_mutation_length = {}".format(len(train_tokens_dict_with_mutation)))
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_X_dict.keys()
    for i in train_file_names:
        train_X_list.append(train_X_dict[i])
        train_Y_list.append(train_Y_dict[i])
    train_Y = np.array(train_Y_list)
    test_file_names = test_X_dict.keys()
    for i in test_file_names:
        test_X_list.append(test_X_dict[i])
        test_Y_list.append(test_Y_dict[i])
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
    fix_length = len(train_X_list[0])
    print("fix_length={}".format(fix_length))
    num = np.sum(train_Y==0)-np.sum(train_Y==1)
    if (len(train_tokens_dict_with_mutation)<num) :
        numlist = [i for i in range(len(train_tokens_dict_with_mutation))]
    else:
        numlist = random.sample(range(0,len(train_tokens_dict_with_mutation)-1),num)
        # while len(numArray)<num:
        #     random_num = random.randint(0,len(num_tokens_dict_with_mutation)-1)
        #     # 确定变异文件为训练集中存在的文件变异而来的
        #     # 工具生成的变异文件
        #     file_name = list(num_tokens_dict_with_mutation.keys())[random_num].split("_")[0]
        #     if(file_name in train_X_dict.keys()):
        #         numArray.add(random_num)
        # numlist = list(numArray)
    for i in numlist:
        key = list(train_tokens_dict_with_mutation.keys())[i]
        num_tokens = train_tokens_dict_with_mutation[key]
        if(len(num_tokens)>fix_length):
            num_tokens = num_tokens[:fix_length]
        else:
            length = len(num_tokens)
            for _ in range(fix_length-length):
                num_tokens.append(0)
        train_X_list.append(num_tokens)
        train_Y = np.append(train_Y,1)
    train_X = np.array(train_X_list)
    end_time = time.time()
    with open("./cost_time.txt", "a+") as f:
        f.write("{}_{}_{}_without_mutation_cost:{} \n".format(project, train_version,test_version,end_time-start_time))
    # dbn_train(project_name,train_version,test_version,"with_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("mutation_oversample end")


def manual_mutation_oversample(project_name,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("manual_mutation_oversample start")
    start_time = time.time()
    with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name, train_version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
        # file_names = num_tokens_dict_with_mutation.keys()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    # 从变异文件的token序列中随机选取和原始数据相同大小的样本数
    fix_length = len(train_X_list[0])
    print("fix_length={}".format(fix_length))
    num = np.sum(train_Y==0)-np.sum(train_Y==1)
    print("num:{}".format(num))
    print("token_list_length:{}".format(len(num_tokens_dict_with_mutation)))

    random.seed(3618) 

    if (len(num_tokens_dict_with_mutation)<=num) :
        numlist = [i for i in range(len(num_tokens_dict_with_mutation))]
    else:
        numlist = random.sample(range(0,len(num_tokens_dict_with_mutation)-1),num)
        # while len(numArray)<num:
        #     random_num = random.randint(0,len(num_tokens_dict_with_mutation)-1)
        #     # 确定变异文件为训练集中存在的文件变异而来的
        #     # 手工生成的变异文件
        #     if(list(num_tokens_dict_with_mutation.keys())[random_num] in train_X_dict.keys()):
        #         numArray.add(random_num)
        # numlist = list(numArray)
    for i in numlist:
        key = list(num_tokens_dict_with_mutation.keys())[i]
        num_tokens = num_tokens_dict_with_mutation[key]
        if(len(num_tokens)>fix_length):
            num_tokens = num_tokens[:fix_length]
        else:
            length = len(num_tokens)
            for _ in range(fix_length-length):
                num_tokens.append(0)
        train_X_list.append(num_tokens)
        train_Y = np.append(train_Y,1)
    train_X = np.array(train_X_list)
    end_time = time.time()
    # with open("./cost_time.txt", "a+") as f:
    #     f.write("{}_{}_{}_with_mutation_cost:{} \n".format(project_name, train_version,test_version,end_time-start_time))
    dbn_train(project_name,train_version,test_version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("manual_mutation_oversample end")

def ROS_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("ROS_oversample start")
    start_time = time.time()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    # print("ROS:{}__  {}".format(len(train_X),len(train_X_dict)))
    ros = ROS(random_state=66)
    train_X_resample,train_Y_resample = ros.fit_resample(train_X,train_Y)
    end_time = time.time()
    # with open("./cost_time.txt", "a+") as f:
    #     f.write("{}_{}_{}_with_ROS_cost:{} \n".format(project, train_version,test_version,end_time-start_time))
    dbn_train(project,train_version,test_version,"with_ROS",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    print("ROS_oversample end")

def SMOTE_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("SMOTE_oversample start")
    start_time = time.time()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    smote = SMOTE(random_state=3618)
    train_X_resample,train_Y_resample = smote.fit_resample(train_X,train_Y)
    end_time = time.time()
    # with open("./cost_time.txt", "a+") as f:
    #     f.write("{}_{}_{}_with_smote_cost:{} \n".format(project, train_version,test_version,end_time-start_time))
    dbn_train(project,train_version,test_version,"with_SMOTE",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    print("SMOTE_oversample end")

def KMeansSMOTE_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict):
    print("KMeansSMOTE_oversample start")
    start_time = time.time()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    try:
        print(np.sum(train_Y==0))
        print(np.sum(train_Y==1))
        print(train_X.shape)
        print(train_Y.shape)
        ksmote = KMeansSMOTE()

        train_X_resample,train_Y_resample = ksmote.fit_resample(train_X,train_Y)
        print(np.sum(train_Y_resample==0))
        print(np.sum(train_Y_resample==1))
    except RuntimeError:
        return 0 
    print(train_X.shape)
    end_time = time.time()
    with open("./cost_time.txt", "a+") as f:
        f.write("{}_{}_{}_with_KMeansSMOTE_cost:{} \n".format(project, train_version,test_version,end_time-start_time))
    # dbn_train(project,train_version,test_version,"with_KMeansSMOTE",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    print("KMeansSMOTE_oversample end")
    return 1



def mix_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,ros_ratio,smote_ratio,mutation_ratio):
    print("mix_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(train_X_dict[i])
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(test_X_dict[i])
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    defect_num = np.sum(train_Y == 1)
    nondefect_num = np.sum(train_Y == 0)
    D_value = (nondefect_num-defect_num)
    print("train_X:{}".format(train_X.shape))


    
    # ros = ROS(sampling_strategy={0:nondefect_num,1:num_for_ros},random_state=66)
    # train_X_resample_ros,train_Y_resample_ros = ros.fit_resample(train_X,train_Y)
    # print("train_X_resample_ros:{}".format(len(train_X_resample_ros)))

    # smote部分
    num_for_smote = int(D_value*smote_ratio)+defect_num
    smote = SMOTE(sampling_strategy={0:nondefect_num,1:num_for_smote},random_state=3618)
    train_X_resample_smote,train_Y_resample_smote = smote.fit_resample(train_X,train_Y)
    print("train_X_resample_smote:{}".format(train_X_resample_smote.shape))
    print(num_for_smote)
    print("train_X_resample_smote:{}".format(np.sum(train_Y_resample_smote==1)))
    print("train_X:{}".format(train_X.shape))

    #变异部分
    fix_length = len(train_X_list[0])
    num_for_mutation = int(D_value*mutation_ratio)
    train_X_resample_mutation = []
    train_Y_resample_mutation = []
    with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project, train_version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
    if (len(num_tokens_dict_with_mutation)<=num_for_mutation) :
        numlist = [i for i in range(len(num_tokens_dict_with_mutation))]
    else:
        numlist = random.sample(range(0,len(num_tokens_dict_with_mutation)-1),num_for_mutation)
    for i in numlist:
        key = list(num_tokens_dict_with_mutation.keys())[i]
        num_tokens = num_tokens_dict_with_mutation[key]
        if(len(num_tokens)>fix_length):
            num_tokens = num_tokens[:fix_length]
        else:
            length = len(num_tokens)
            for _ in range(fix_length-length):
                num_tokens.append(0)
        train_X_resample_mutation.append(num_tokens)
        train_Y_resample_mutation.append(1)
    # train_Y_resample_mutation = train_Y
    # train_X_resample_mutation = np.append(train_X,add_list,axis=0)
    # train_X = np.array(train_X_list)
    # print("train_X_resample_mutation:{}".format(len(train_X_resample_mutation)))

    # ros部分(手动实现)
    random.seed(66)
    num_for_ros = int(D_value*ros_ratio)
    train_X_length = len(train_X)
    index_list = []
    train_X_resample_ros=[]
    train_Y_resample_ros=[]
    for i in range(num_for_ros):
        index = random.randint(0,train_X_length-1)
        index_list.append(index)
    for i in index_list:
        train_X_resample_ros.append(train_X[i])
        train_Y_resample_ros.append(train_Y[i])

    # 整合三种增强方式
    train_X_resample_list = []
    train_Y_resample_list = []
    # train_X_resample = np.append(train_X_resample_smote,train_X_resample_ros,axis=0)
    # train_Y_resample = np.append(train_Y_resample_smote,train_Y_resample_ros)
    # train_X_resample = np.append(train_X_resample,train_X_resample_mutation,axis=0)
    # train_Y_resample = np.append(train_Y_resample,train_Y_resample_mutation)
    train_X_resample = np.append(train_X_resample_smote,train_X_resample_mutation,axis=0)
    train_Y_resample = np.append(train_Y_resample_smote,train_Y_resample_mutation)

    print("train_X_resample:{}".format(train_X_resample.shape))
    dbn_train(project,train_version,test_version,"with_mix",cut_length,train_X_resample,train_Y_resample,test_X,test_Y,ros_ratio,smote_ratio,mutation_ratio)
    print("mix_oversample end")



def dbn_train(project,train_version,test_version,data_type,cut_length,train_X,train_Y,test_X,test_Y,ros_ratio=0,smote_ratio=0,mutation_ratio=0):
    # 构建dbn模型进行训练
    max_num = 0
    with open("../vocabdict/{}.pkl".format(project),'rb') as f:
        vocabdict = pickle.load(f)
        max_num = len(vocabdict)
    train_X = train_X / (max_num)
    test_X = test_X / max_num
    within_dbn = UnsupervisedDBN(hidden_layers_structure=[512,256,100],
                                    batch_size=4096,
                                    learning_rate_rbm=0.06,
                                    n_epochs_rbm=100,
                                    activation_function='sigmoid',
                                    verbose=False)

    print(train_X.shape)
    print(test_X.shape)
    within_dbn_input = train_X

    print('{}_{}_{}_{}_{} within DBN train started'.format(project,train_version,test_version,data_type,cut_length))
    within_dbn.fit(within_dbn_input)

    print('{}_{}_{}_{}_{} within DBN train ended'.format(project,train_version,test_version,data_type,cut_length))
    within_train_dbn_output = within_dbn.transform(within_dbn_input)
    print(within_train_dbn_output.shape)
    if data_type == "with_mix":
        np.save('./train_data_by_version/{}_{}_{}_train_X_{}_{}_{}_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length,ros_ratio,smote_ratio,mutation_ratio), within_train_dbn_output)

        np.save('./train_data_by_version/{}_{}_{}_train_Y_{}_{}_{}_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length,ros_ratio,smote_ratio,mutation_ratio), np.array(train_Y))
    else:
        np.save('./train_data_by_version/{}_{}_{}_train_X_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length), within_train_dbn_output)

        np.save('./train_data_by_version/{}_{}_{}_train_Y_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length), np.array(train_Y))

    within_test_dbn_output = within_dbn.transform(test_X)
    np.save('./test_data_by_version/{}_{}_{}_test_X_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length), within_test_dbn_output)

    np.save('./test_data_by_version/{}_{}_{}_test_Y_{}_{}.npy'.format(project,train_version,test_version,data_type,cut_length), np.array(test_Y))

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




def handle_batch(projects,cut_length):
    Processes = []  
    pool= Pool(processes=(int(cpu_count()/2)))
    for project in projects.keys():
        # 获取每个项目版本号
        project_path = "../PROMISE/promise_data/{}".format(project)
        versions = projects[project]
        # for root,dirnames,filenames in os.walk(project_path):
        #     for i in filenames:
        #         versions.append(i[0:-4])
            # versions.sort()
        # 根据项目名和版本号获取最终的数据，并使用sklearn进行训练测试集划分
        for i in range(len(versions)-1):
            train_version = versions[i]
            test_version = versions[i+1]
            train_X,test_X,train_Y,test_Y,cut_length_string,train_fix_length = pre_process(project,train_version,test_version,cut_length)
            # test_X,test_Y,cut_length_string,test_fix_length = pre_process(project,test_version,cut_length,train_fix_length)
            # print(DBN_data.shape)
            # train_X,test_X,train_Y,test_Y = train_test_split(project,version,DBN_data_dict,labels_dict,test_size=0.3,random_state = 10)
            
            # without_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)
            # mutation_oversample(project,version,cut_length_string,train_X,train_Y,test_X,test_Y)
            # ROS_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)
            # SMOTE_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)
            # is_finish = 0
            # while is_finish == 0:
            #     is_finish = KMeansSMOTE_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)
            manual_mutation_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)
            # mix_oversample(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y,0,0.25,0.75)


            # # 逐个项目跑
            # p1 = Process(target=without_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y))
            # p2 = Process(target=SMOTE_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y))
            # p3 = Process(target=ROS_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y))
            # p4 = Process(target=manual_mutation_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y))
            # start_time = time.time()
            # p1.start()
            # p2.start()
            # p3.start()
            # p4.start()
            # p1.join()
            # p2.join()
            # p3.join()
            # p4.join()
            # end_time = time.time()
            # print("time:{}".format(end_time-start_time))

    #         # 所有项目一起跑
    #         Processes.append(Process(target=without_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)))
    #         Processes.append(Process(target=ROS_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)))
    #         Processes.append(Process(target=SMOTE_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)))
    #         Processes.append(Process(target=manual_mutation_oversample,args=(project,train_version,test_version,cut_length_string,train_X,train_Y,test_X,test_Y)))
    # start_time = time.time()
    # for process in Processes:
    #     process.start()
    # for process in Processes:
    #     process.join()
    # end_time = time.time()
    # print("time:{}".format(end_time-start_time))


# pre_process("lucene",'2.4')
# projects = ['jEdit','ant','ivy']
# projects_with_version = {'ant':['1.6','1.7'],"jEdit":['4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],"ivy":['1.4','2.0']}
# projects_with_version = {"synapse":['1.0','1.1'],"camel":['1.4','1.6'],
#                             "ivy":['1.4','2.0']}
# projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
#                             "ivy":['1.4','2.0'],"poi":['2.0','2.5'],"xalan":['2.4','2.5'],"xerces":['1.2','1.3'],"log4j":['1.0','1.1']}
# projects_with_version = {"xerces":['1.2','1.3'],"log4j":['1.0','1.1']
# projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
#                             "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
projects_with_version = {"ant":['1.5','1.6'],"jEdit":['4.1','4.2'],"camel":['1.4','1.6'],"xalan":['2.4','2.5']}
# projects_with_version = {"synapse":['1.0','1.1']}
# projects_with_version = {"jEdit":['4.0','4.1','4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
#                             "ivy":['1.4','2.0'],"poi":['2.0','2.5'],"xalan":['2.4','2.5'],"xerces":['1.2','1.3'],"log4j":['1.0','1.1']}
# projects = ['ivy']
# projects = ['ant']
handle_batch(projects_with_version,0.5)
# handle_batch("ant")