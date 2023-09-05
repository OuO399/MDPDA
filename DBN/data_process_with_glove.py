import javalang
from javalang.ast import Node
import os
import pickle
from get_token_list import get_token_list,get_mutation_token_list
from multiprocessing import Process
import time
import pandas as pd
import re
from get_token_embeding import TokenEmbedding

def get_token(node):
    token = ''
    # print(isinstance(node, Node))
    # print(type(node))
    if isinstance(node, str):
        token = node
        print("node 是 str类型：",token)
    elif isinstance(node, set):
        # print("node 是 set类型：", list(node))
        token = 'Modifier'

    if isinstance(node, Node):
        token = node.__class__.__name__
        print("node 是 Node类型：",token)
    # print(node.__class__.__name__,str(node))
    # print(node.__class__.__name__, node)
    return token


def get_child(root):
    # print(root)
    if isinstance(root, Node):
        children = root.children
        # print(children)
    elif isinstance(root, set):
        children = list(root)
        # print(children)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))

def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)


# 根据token序列，针对每个项目构建词表，将token序列数值化
def tokenized_ast(project_path,project_name,version):
    if os.path.exists("../numtokens_with_GloVe/{}_{}_without_mutation_before_embedding.pkl".format(project_name,version)):
        with open("../numtokens_with_GloVe/{}_{}_without_mutation_before_embedding.pkl".format(project_name,version),"rb") as f:
            num_tokens_dict = pickle.load(f)
    else:
        num_tokens_dict = {}
    start = len(project_path)+1
    count =0
    # em = TokenEmbedding(project_name,version)
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            file_path = os.path.join(path,file)
            alltokens = get_token_list(file_path)
            if alltokens == None:
                continue
            # alltokens_embedding = em.get_token_embedding(alltokens)

            if file_path[start:].replace('/','.') in num_tokens_dict.keys():
                continue
            num_tokens_dict[file_path[start:].replace('/','.')] = alltokens
            count+=1


    print(len(num_tokens_dict.keys()))
    max_length = 0
    for i in num_tokens_dict.values():
        if len(i)>max_length:
            max_length = len(i)
    token_length_list = []
    count1 = 0
    for i,j in num_tokens_dict.items():
        if len(num_tokens_dict[i]) >max_length*0.3:
            count1 += 1
        token_length_list.append(len(num_tokens_dict[i]))
    with open("./0.3_length.txt".format(project_name,version),"a+", encoding='utf-8') as f:
        f.write('{}_{}_0.3     maxlength:{}   nums:{}  ratio:{}   length_list:{}\n'.format(
            project_name,version,max_length,count1,(count1/len(num_tokens_dict)),token_length_list
        ))
        f.write("\n")
    print("token_length_list: {}".format(token_length_list)) 

    print("count1 = {}".format(count1))
    print("count = {}".format(count))
    # print(vocabdict)
    # print(count)
    with open("../numtokens_with_GloVe/{}_{}_without_mutation_before_embedding.pkl".format(project_name,version),"wb") as f:
        pickle.dump(num_tokens_dict,f)


def get_non_bug_files(project_name,version):
    df = pd.read_csv("../PROMISE/promise_data/{}/{}.csv".format(project_name, version))
    non_bug_files = []
    for i in range(len(df)):
        # print(type(df.iloc[i]["name"]))
        # print(df.iloc[i]["bugs"])
        if df.iloc[i]["bugs"] == 0:
            non_bug_files.append(df.iloc[i]["name"])
    # print(non_bug_files)
    return non_bug_files

# 手动针对每个没有缺陷的文件进行变异
def tokenized_manual_mutation_ast(project_path,project_name,version):
    if os.path.exists("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project_name,version)):
        with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project_name,version),"rb") as f:
            num_tokens_dict = pickle.load(f)
    else:
        num_tokens_dict = {}
    # with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project_name,version),"rb") as f:
    #     num_tokens_dict = pickle.load(f)

    non_bug_files = get_non_bug_files(project_name, version)
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            file_path = os.path.join(path,file)
            list1 = re.split("[/\\\]",file_path)
            file_name = ""
            for i in list1[3:]:
                file_name += i
                file_name += "."
            file_name = file_name[:-1]
            if file_name[:-5] not in non_bug_files:
                continue
            alltokens,count = get_mutation_token_list(file_path,3)
            if (count == 0):
                flag = 0
                for i in range(10):
                    alltokens,count = get_mutation_token_list(file_path,3)
                    if count != 0:
                        flag = 1
                        break
                if flag ==0:
                    continue
            current_file_token_dict = {}
            for times in range(5):
                new_file_name = file_name+"_"+str(times)
                ast_num_tokens_list = []
                if(ast_num_tokens_list not in current_file_token_dict.values()):
                    current_file_token_dict[new_file_name] = ast_num_tokens_list
                alltokens,count = get_mutation_token_list(file_path,3)
            for i in current_file_token_dict.keys():
                num_tokens_dict[i] = current_file_token_dict[i]
            # alltokens,count = get_mutation_token_list(file_path,1)
            # if (count == 0):
            #     continue
            # for times in range(5):
            #     new_file_name = file_name+"_"+str(times)
            #     if new_file_name in num_tokens_dict.keys():
            #         continue
            #     num_tokens_dict[new_file_name] = alltokens
            #     alltokens,count = get_mutation_token_list(file_path,times+2)
            #     if count < times+2:
            #         break
    # em = Embedding(project_name,version)
    # num_tokens_dict_after_embedding = {}
    # for i in num_tokens_dict.keys():
    #     alltokens_embedding = em.get_token_embedding(num_tokens_dict[i])
    #     num_tokens_dict_after_embedding[num_tokens_dict[i]] = alltokens_embedding

    # print(len(num_tokens_dict_after_embedding.keys()))
    # print(vocabdict)    
    with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project_name, version), "wb") as f:
        pickle.dump(num_tokens_dict, f)


# # 根据token序列，针对每个项目构建词表，将token序列数值化
# def tokenized_mutation_file_ast(project_path,project_name,version):
#     if os.path.exists("../numtokens_with_GloVe/{}_{}_with_mutation_file_before_embedding.pkl".format(project_name,version)):
#         with open("../numtokens_with_GloVe/{}_{}_with_mutation_file_before_embedding.pkl".format(project_name,version),"rb") as f:
#             num_tokens_dict = pickle.load(f)
#     else:
#         num_tokens_dict = {}
#     em = TokenEmbedding(project_name,version)
#     non_bug_files = get_non_bug_files(project_name,version)
#     for path,file_dirs,files in os.walk(project_path):
#         source_files =[f for f in files if is_code(f)]
#         for file in source_files:
#             file_path = os.path.join(path,file)
#             if "original" in file_path:
#                 continue
#             list1 = file_path.split("/")
#             if list1[3] in non_bug_files:
#                 try:
#                     alltokens = get_token_list(file_path)
#                     # alltokens_embedding = em.get_token_embedding(alltokens)
#                 except FileNotFoundError:
#                     continue
#                 file_key_name = list1[3]+".java_"+list1[-2] #list[-2]为变异算子
#                 if file_key_name in num_tokens_dict.keys():
#                     continue
#                 num_tokens_dict[file_key_name] = alltokens
#     print(len(num_tokens_dict.keys()))
#     # print(vocabdict)
#     with open("../numtokens_with_GloVe/{}_{}_with_mutation_file_before_embedding.pkl".format(project_name,version),"wb") as f:
#         pickle.dump(num_tokens_dict,f)
        


def is_code(filename):
    return filename.endswith('.java')


# def get_all_file_num(path):
#     count = 0
#     list1 = []
#     for dirpath,dirs,files in os.walk(path):
#         source_files = [f for f in files if is_code(f)]
#         for file in source_files:
#             list1.append(file)
#     print(len(list1))

def process_for_each_project(project,version):
    project_path = '../dataset_source/{}_{}'.format(project,version)
    project_mutation_file_path = '../mutation_file/{}_{}'.format(project, version)

    start_time = time.time()
    print("{}_{}_without_mutation start".format(project,version))
    tokenized_ast(project_path,project,version)
    print("{}_{}_without_mutation end".format(project,version))
    # print("{}_{}_with_mutation start".format(project,version))
    # tokenized_mutation_file_ast(project_mutation_file_path,project,version)
    # print("{}_{}_with_mutation end".format(project,version))
    print("{}_{}_with_manual_mutation start".format(project,version))
    # tokenized_manual_mutation_ast(project_path,project,version)
    print("{}_{}_with_manual_mutation end".format(project,version))
    end_time = time.time()
    print("{}_{}_time:{}".format(project,version,end_time-start_time))


if __name__ == '__main__':
    # projects = ["ant"]
    # projects_with_version = {'ant':['1.6','1.7'],'jEdit':['4.3'],'ivy':['2.0']}
    # projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2','4.3'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
    #                             "ivy":['1.4','2.0'],"poi":['2.0','2.5'],"xalan":['2.4','2.5'],"xerces":['1.2','1.3'],"log4j":["1.0","1.1"]}
    projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
                                "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
    # projects_with_version = {'jEdit':['4.1']}
    # projects_with_version = {'jEdit':['4.3']}
    Processes = []
    for project in projects_with_version.keys():
        for version in projects_with_version[project]:
            process_for_each_project(project,version)
    #         p = Process(target=process_for_each_project, args=(project,version))
    #         Processes.append(p)
    # start_time = time.time()
    # for i in Processes:
    #     i.start()
    # for i in Processes:
    #     i.join()
    # end_time = time.time()
    # print("time:{}".format(end_time-start_time))

# if __name__ == '__main__':
#     # projects = ["ant"]
#     projects_with_version = {'ant':['1.6','1.7'],'jEdit':['4.3'],'ivy':['2.0']}
#     # projects_with_version = {'jEdit':['4.3']}
#     for project in projects_with_version.keys():
#         for version in projects_with_version[project]:
#             project_path = '../dataset_source/{}_{}'.format(project,version)
#             project_mutation_file_path = '../mutation_file/{}_{}'.format(project, version)
#             # print(project_path)
#             # get_all_file_num(project_mutation_file_path)
#             start_time = time.time()
#             print("{}_{}_without_mutation start".format(project,version))
#             tokenized_ast(project_path,project,version)
#             print("{}_{}_without_mutation end".format(project,version))
#             print("{}_{}_with_mutation start".format(project,version))
#             tokenized_mutation_file_ast(project_mutation_file_path,project,version)
#             print("{}_{}_with_mutation end".format(project,version))
#             print("{}_{}_with_manual_mutation start".format(project,version))
#             tokenized_manual_mutation_ast(project_path,project,version)
#             print("{}_{}_with_manual_mutation end".format(project,version))
#             end_time = time.time()
#             print("{}_{}_time:{}".format(project,version,end_time-start_time))
            # p1 = Process(target=tokenized_ast,args=(project_path,project,version))
            # p2 = Process(target=tokenized_mutation_file_ast,args=(project_mutation_file_path,project,version))
            # p3 = Process(target=tokenized_manual_mutation_ast,args=(project_path,project,version))

            # start_time = time.time()
            # p1.start()
            # p2.start()
            # p3.start()
            # p1.join()
            # p2.join()
            # p3.join()
            # end_time = time.time()
            # print("time:{}".format(end_time-start_time))

