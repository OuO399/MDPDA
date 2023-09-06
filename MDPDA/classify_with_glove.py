from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix,roc_auc_score

from tensorflow.keras.layers import Input,Embedding,LSTM,Dense,Activation,Multiply, Masking,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Model,Sequential
# from tensorflow.keras.optimizers import adam_v2
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.backend import clear_session
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import numpy as np
from sklearn import linear_model, manifold
from sklearn.utils import compute_class_weight
from multiprocessing import Process
import time
import math
import random
import os
import pickle
from save_for_glove import save_train_set_tokens_in_txt
from get_token_embeding import TokenEmbedding
from imblearn.over_sampling import SMOTE,KMeansSMOTE


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# set GPU memory
# if('tensorflow' == K.backend()):
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
    # sess =tf.compat.v1.Session(config=config)

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
                list1[i].append([0.0 for j in range(40)])
    for i in range(len(list2)):
        if len(list2[i]) > fix_length:
            list2[i] = list2[i][:fix_length]
        elif len(list2[i]) < fix_length:
            length = len(list2[i])
            for _ in range(fix_length-length):
                list2[i].append([0.0 for j in range(40)])

    return list1,list2



def without_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
    print(project)
    # if project == 'jEdit':
    #     print(111)
    #     return 0
    print("without_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(em.get_token_embedding(train_X_dict[i]))
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue


    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(em.get_token_embedding(test_X_dict[i]))
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue

    train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    # if project == "ant" and train_version =="1.5":
    #     return 0
    lstm_classify(project,train_version,test_version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # CNN_classify(project,train_version,test_version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # RF_classify(project,version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # LR_classify(project,version,"without_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("without_oversample end")

# def mutation_oversample(project_name,version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
#     print("mutation_oversample start")
#     with open("../numtokens_with_GloVe/{}_{}_with_mutation_file.pkl".format(project_name, version), "rb") as f:
#         num_tokens_dict_with_mutation = pickle.load(f)
#         # file_names = num_tokens_dict_with_mutation.keys()
#     train_X_list = []
#     train_Y_list = []
#     test_X_list = []
#     test_Y_list = []
#     train_file_names = num_tokens_dict_with_mutation.keys()
#     for i in train_file_names:
#         train_X_list.append(em.get_token_embedding(num_tokens_dict_with_mutation[i]))
#         train_Y_list.append(train_Y_dict[i])
#     test_file_names = test_X_dict.keys()
#     for i in test_file_names:
#         test_X_list.append(em.get_token_embedding(test_X_dict[i]))
#         test_Y_list.append(test_Y_dict[i])

#     train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)
    
#     train_Y = np.array(train_Y_list)
#     train_X = np.array(train_X_list)
#     test_X = np.array(test_X_list)
#     test_Y = np.array(test_Y_list)
#     lstm_classify(project_name,version,"with_mutation",cut_length,train_X,train_Y,test_X,test_Y)
#     # RF_classify(project,version,"with_mutation",cut_length,train_X,train_Y,test_X,test_Y)
#     # LR_classify(project,version,"with_mutation",cut_length,train_X,train_Y,test_X,test_Y)
#     print("mutation_oversample end")


def manual_mutation_oversample(project_name,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
    print("manual_mutation_oversample start")
    with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file.pkl".format(project_name, train_version), "rb") as f:
        num_tokens_dict_with_mutation = pickle.load(f)
        # file_names = num_tokens_dict_with_mutation.keys()
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(em.get_token_embedding(num_tokens_dict_with_mutation[i]))
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue

    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(em.get_token_embedding(test_X_dict[i]))
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue

    train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)
    lstm_classify(project_name,train_version,test_version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # CNN_classify(project_name,train_version,test_version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # RF_classify(project,version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    # LR_classify(project,version,"with_manual_mutation",cut_length,train_X,train_Y,test_X,test_Y)
    print("manual_mutation_oversample end")

def ROS_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
    print("ROS_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(em.get_token_embedding(train_X_dict[i]))
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue

    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(em.get_token_embedding(test_X_dict[i]))
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue

    train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)

    train_Y = np.array(train_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)

    #手动实现ROS
    random.seed(36)
    num = np.sum(train_Y==0)-np.sum(train_Y==1)
    train_X_length = len(train_X_list)
    index_list = []
    for i in range(num):
        index = random.randint(0,train_X_length-1)
        index_list.append(index)
    for i in index_list:
        train_X_list.append(train_X_list[i])
        train_Y = np.append(train_Y,train_Y[i])
    train_X = np.array(train_X_list)
    lstm_classify(project,train_version,test_version,"with_ROS",cut_length,train_X,train_Y,test_X,test_Y)
    # CNN_classify(project, train_version, test_version, "with_ROS", cut_length, train_X, train_Y,
    #              test_X, test_Y)
    # RF_classify(project,version,"with_ROS",cut_length,train_X,train_Y,test_X,test_Y)
    # LR_classify(project,version,"with_ROS",cut_length,train_X,train_Y,test_X,test_Y)

    # print("ROS:{}__  {}".format(len(train_X),len(train_X_dict)))
    # ros = ROS(random_state=66)
    # train_X_resample,train_Y_resample = ros.fit_resample(train_X,train_Y)
    # lstm_classify(project,version,"with_ROS",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    # print("ROS_oversample end")

def SMOTE_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
    print("SMOTE_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(em.get_token_embedding(train_X_dict[i]))
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    # print(len(train_X_list))
    # for i in train_X_list:
    #     if len(i) != 40:
    #         print(len(i))
    # train_X = np.array(train_X_list)
    # train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(em.get_token_embedding(test_X_dict[i]))
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue

    train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)
    print(train_X.shape)
    x = train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2])
    print(x.shape)
    smote = SMOTE(random_state=3618)
    train_X_resample,train_Y_resample = smote.fit_resample(x,train_Y)
    train_X_resample = train_X_resample.reshape(train_X_resample.shape[0],train_X.shape[1],train_X.shape[2])
    print(train_X_resample.shape)
    lstm_classify(project,train_version,test_version,"with_SMOTE",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    # CNN_classify(project, train_version, test_version, "with_SMOTE", cut_length, train_X_resample, train_Y_resample,
    #              test_X, test_Y)
    print("SMOTE_oversample end")

def KMeansSMOTE_oversample(project,train_version,test_version,cut_length,train_X_dict,train_Y_dict,test_X_dict,test_Y_dict,em):
    print(project+"KMeansSMOTE_oversample start")
    train_X_list = []
    train_Y_list = []
    test_X_list = []
    test_Y_list = []
    train_file_names = train_Y_dict.keys()
    for i in train_file_names:
        try:
            train_X_list.append(em.get_token_embedding(train_X_dict[i]))
            train_Y_list.append(train_Y_dict[i])
        except KeyError:
            continue
    # print(len(train_X_list))
    # for i in train_X_list:
    #     if len(i) != 40:
    #         print(len(i))
    # train_X = np.array(train_X_list)
    # train_Y = np.array(train_Y_list)
    test_file_names = test_Y_dict.keys()
    for i in test_file_names:
        try:
            test_X_list.append(em.get_token_embedding(test_X_dict[i]))
            test_Y_list.append(test_Y_dict[i])
        except KeyError:
            continue

    train_X_list,test_X_list = crop_or_fill(train_X_list,test_X_list,cut_length)
    train_X = np.array(train_X_list)
    train_Y = np.array(train_Y_list)
    test_X = np.array(test_X_list)
    test_Y = np.array(test_Y_list)
    print(train_X.shape)
    x = train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2])
    
    # x = np.random.randn(281,100000)
    # flag = 1
    # while flag:
    #     print (1)
    try:
        # print(np.sum(train_Y==0))
        # print(np.sum(train_Y==1))
        # print(x.shape)
        # print(train_Y.shape)
        ksmote = KMeansSMOTE()
        train_X_resample,train_Y_resample = ksmote.fit_resample(x,train_Y)
        # print(np.sum(train_Y_resample==0))
        # print(np.sum(train_Y_resample==1))
        flag = 0
    except ValueError as e:
        print("ValueError")
        return 1
    except RuntimeError as r:
        print("RuntimeError")
        return 1
    train_X_resample = train_X_resample.reshape(train_X_resample.shape[0],train_X.shape[1],train_X.shape[2])
    print(train_X_resample.shape)
    lstm_classify(project,train_version,test_version,"with_KMeansSMOTE",cut_length,train_X_resample,train_Y_resample,test_X,test_Y)
    # CNN_classify(project, train_version, test_version, "with_KMeansSMOTE", cut_length, train_X_resample, train_Y_resample,
    #              test_X, test_Y)
    print("KMeansSMOTE_oversample end")
    return 0


# 直接使用lstm进行分类
def lstm_classify(project,train_version,test_version,data_type,cut_length,train_X,train_Y,test_X,test_Y):
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    for i in range(10):
        traditional_input = Input(shape=(train_X.shape[1],40),name='input')
        # print(traditional_input.shape)
        # mask_zeros = Masking(mask_value=0.0)(traditional_input)
        # print(mask_zeros.shape)
        traditional_lstm_out = LSTM(128,name='promise_lstm')(traditional_input)#(mask_zeros)
        # print(traditional_lstm_out.shape)
        traditional_gate = Dense(128,activation='sigmoid',name='traditional_gate')(traditional_lstm_out)
        # print(traditional_gate.shape)
        traditional_gated_res = Multiply(name='traditional_gated_res')([traditional_gate,traditional_lstm_out])
        # print(traditional_gated_res.shape)
        main_output = Dense(1,activation='sigmoid',name='main_output')(traditional_gated_res)


        model = Model(inputs=[traditional_input], outputs=[main_output])
        early_stopping = EarlyStopping(monitor="f1", mode="max", patience=40, verbose=1, restore_best_weights=True)
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy',f1])

        val_data = ({'input': test_X},
                    {'main_output': test_Y})
        model.fit(x={'input': train_X},
                    y={'main_output': train_Y},
                    # batch_size=128,
                    batch_size=512,
                    epochs=100,
                #   class_weight=lstm_weight,
                    # validation_split=0.2,
                    # callbacks=[early_stopping],
                    )

        predict_y = model.predict(x={'input': test_X})
        np.save('./Deeper_LSTM/{}_{}_{}_{}.npy'.format(project, train_version,test_version, data_type), predict_y)
        predict_y_LSTM = np.round(predict_y)
        confusion_matrix_LSTM = confusion_matrix(test_Y,predict_y_LSTM)
        TP = confusion_matrix_LSTM[1][1]
        FP = confusion_matrix_LSTM[0][1]
        TN = confusion_matrix_LSTM[0][0]
        FN = confusion_matrix_LSTM[1][0]
        MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc_list.append(MCC)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_LSTM))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_LSTM))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_LSTM))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_LSTM))
        auc_list.append(roc_auc_score(test_Y, predict_y_LSTM))
        with open('./Deeper_LSTM/res_with_glove_by_version_{}_1206_synapse_KmeansSmote_b512_e100_median.txt'.format(cut_length), 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                data_type,
                cut_length,
                i,
                precision_score(y_true=test_Y, y_pred=predict_y_LSTM),
                recall_score(y_true=test_Y, y_pred=predict_y_LSTM),
                f1_score(y_true=test_Y, y_pred=predict_y_LSTM),
                accuracy_score(y_true=test_Y, y_pred=predict_y_LSTM),
                roc_auc_score(test_Y, predict_y_LSTM),
                MCC
            ))
    with open('./Deeper_LSTM/res_with_glove_by_version_{}_1206_synapse_KmeansSmote_b512_e100_median.txt'.format(cut_length), 'a+', encoding='utf-8') as f:
        f.write('\n')        
    print(np.array(f1_list).mean())
    print(np.array(auc_list).mean())
    with open('./Deeper_LSTM/res_with_glove_by_version_{}_1206_synapse_KmeansSmote_b512_e100.txt'.format(cut_length), 'a+', encoding='utf-8') as f:
        f.write('{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            data_type,
            cut_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write("\n")

#使用CNN分类
def CNN_classify(project,train_version,test_version,data_type,cut_length,train_X,train_Y,test_X,test_Y):
    CNN_sum_list = [0 for i in range(6)]
    p_list = []
    r_list = []
    f1_list = []
    acc_list = []
    auc_list = []
    mcc_list = []
    print(train_X.shape)
    print(len(train_Y))
    print(test_X.shape)
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2],1)
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],test_X.shape[2],1)
    print(train_X.shape)
    print(test_X.shape)
    # print(train_Y)
    for i in range(30):

        traditional_input = Input(shape=(train_X.shape[1], train_X.shape[2], 1),name="input")
        traditional_cnn1_output = Conv2D(16,kernel_size=5,strides=1,activation='relu',name="traditional_cnn1_output")(traditional_input)
        traditional_maxpool1_output = MaxPooling2D(pool_size=2,strides=2,name="traditional_maxpool1_output")(traditional_cnn1_output)
        traditional_cnn2_output = Conv2D(32,kernel_size=5,strides=1,activation='relu',name="traditional_cnn2_output")(traditional_maxpool1_output)
        traditional_maxpool2_output = MaxPooling2D(pool_size=2,strides=2,name="traditional_maxpool2_output")(traditional_cnn2_output)
        traditional_flatten_output = Flatten(name="traditional_flatten_output")(traditional_maxpool2_output)
        traditional_dense_output = Dense(128,activation="sigmoid",name="traditional_dense_output")(traditional_flatten_output)
        main_output = Dense(1,activation="sigmoid",name="main_output")(traditional_dense_output)

        model = Model(inputs=[traditional_input],outputs=[main_output])
        early_stopping = EarlyStopping(monitor="f1", mode="max", patience=40, verbose=1, restore_best_weights=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1])
        model.fit(x={'input': train_X},
                    y={'main_output': train_Y},
                    batch_size=128,
                    epochs=250,
                #   class_weight=lstm_weight,
                    # validation_split=0.2,
                    callbacks=[early_stopping])

        
        predict_y = model.predict(x={'input': test_X})
        np.save('./Deeper_CNN/{}_{}_{}_{}.npy'.format(project, train_version, test_version, data_type), predict_y)
        predict_y_CNN = np.round(predict_y)
        confusion_matrix_CNN = confusion_matrix(test_Y, predict_y_CNN)
        TP = confusion_matrix_CNN[1][1]
        FP = confusion_matrix_CNN[0][1]
        TN = confusion_matrix_CNN[0][0]
        FN = confusion_matrix_CNN[1][0]
        MCC_CNN = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc_list.append(MCC_CNN)
        p_list.append(precision_score(y_true=test_Y, y_pred=predict_y_CNN))
        r_list.append(recall_score(y_true=test_Y, y_pred=predict_y_CNN))
        f1_list.append(f1_score(y_true=test_Y, y_pred=predict_y_CNN))
        acc_list.append(accuracy_score(y_true=test_Y, y_pred=predict_y_CNN))
        auc_list.append(roc_auc_score(test_Y, predict_y_CNN))
        # CNN_sum_list[0] += precision_score(y_true=test_Y, y_pred=predict_y_CNN)
        # CNN_sum_list[1] += recall_score(y_true=test_Y, y_pred=predict_y_CNN)
        # CNN_sum_list[2] += f1_score(y_true=test_Y, y_pred=predict_y_CNN)
        # CNN_sum_list[3] += accuracy_score(y_true=test_Y, y_pred=predict_y_CNN)
        # CNN_sum_list[4] += roc_auc_score(test_Y,predict_y_CNN)
        # CNN_sum_list[5] += MCC_CNN
        with open('./Deeper_CNN/res_with_glove_by_version_{}_median_value_1108.txt'.format(cut_length), 'a+', encoding='utf-8') as f:
            f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
                project,
                train_version,test_version,
                str(i),
                data_type,
                cut_length,
                precision_score(y_true=test_Y, y_pred=predict_y_CNN),
                recall_score(y_true=test_Y, y_pred=predict_y_CNN),
                f1_score(y_true=test_Y, y_pred=predict_y_CNN),
                accuracy_score(y_true=test_Y, y_pred=predict_y_CNN),
                roc_auc_score(test_Y,predict_y_CNN),
                MCC_CNN
            ))
    # CNN_avg = np.array(CNN_sum_list) / 30
    with open('./Deeper_CNN/res_with_glove_by_version_{}_1108.txt'.format(cut_length), 'a+', encoding='utf-8') as f:
        f.write('\n')
        f.write('{}_{}_{}_{}_{}_{} P:{}  R:{}  F1:{}  A:{}  AUC:{}   MCC:{}\n'.format(
            project,
            train_version,test_version,
            "avg",
            data_type,
            cut_length,
            np.array(p_list).mean(),
            np.array(r_list).mean(),
            np.array(f1_list).mean(),
            np.array(acc_list).mean(),
            np.array(auc_list).mean(),
            np.array(mcc_list).mean()
        ))
        f.write('\n')

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def handle_batch(projects_with_version,cut_length):
    Processes = []
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
            em = TokenEmbedding(project,train_version,test_version)
            # project_version= project+"_"+version


            with open("./train_data_with_glove/{}_{}_train_X_{}.pkl".format(project,train_version,cut_length),"rb") as f :
                train_X = pickle.load(f)

            with open("./train_data_with_glove/{}_{}_without_mutation_train_Y_{}.pkl".format(project,train_version,cut_length),"rb") as f :
                train_Y_without_mutation = pickle.load(f)

            # with open("./train_data_with_glove/{}_{}_with_mutation_train_Y_{}.pkl".format(project,version,cut_length),"rb") as f :
            #     train_Y_with_mutation = pickle.load(f)

            with open("./train_data_with_glove/{}_{}_with_manual_mutation_train_Y_{}.pkl".format(project,train_version,cut_length),"rb") as f :
                train_Y_with_manual_mutation = pickle.load(f)

            with open("./test_data_with_glove/{}_{}_test_X_{}.pkl".format(project,test_version,cut_length),"rb") as f :
                test_X = pickle.load(f)

            with open("./test_data_with_glove/{}_{}_test_Y_{}.pkl".format(project,test_version,cut_length),"rb") as f :
                test_Y = pickle.load(f)

            print(len(train_X))
            print(len(train_Y_without_mutation))
            print(len(train_Y_with_manual_mutation))

            without_oversample(project,train_version,test_version,cut_length,train_X,train_Y_without_mutation,test_X,test_Y,em)
            ROS_oversample(project,train_version,test_version,cut_length,train_X,train_Y_without_mutation,test_X,test_Y,em)
            manual_mutation_oversample(project,train_version,test_version,cut_length,train_X,train_Y_with_manual_mutation,test_X,test_Y,em)
            SMOTE_oversample(project,train_version,test_version,cut_length,train_X,train_Y_without_mutation,test_X,test_Y,em)
            flag = 1
            while flag == 1:
                flag = KMeansSMOTE_oversample(project,train_version,test_version,cut_length,train_X,train_Y_without_mutation,test_X,test_Y,em) 


if __name__ == '__main__':
    projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
                                "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}

    handle_batch(projects_with_version,1.0) 
    # handle_batch("ant")