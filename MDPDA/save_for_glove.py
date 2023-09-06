import javalang
from javalang.ast import Node
import os
import pickle
from get_token_list import get_token_list,get_mutation_token_list
from multiprocessing import Process
import time
import pandas as pd
import re

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
def save_tokenized_ast(project_path,project_name,version):
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            file_path = os.path.join(path,file)
            alltokens = get_token_list(file_path)
            save_ast_token_in_txt(project_name,version,alltokens,"a")



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
def save_tokenized_manual_mutation_ast(project_path,project_name,version):
    non_bug_files = get_non_bug_files(project_name, version)
    manual_mutation_dict = {}
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
            alltokens,count = get_mutation_token_list(file_path,1)
            if (count == 0):
                continue
            for times in range(5):
                new_file_name = file_name+"_"+str(times)
                manual_mutation_dict[new_file_name] = alltokens
                save_ast_token_in_txt(project_name,version,alltokens,"a")
                alltokens,count = get_mutation_token_list(file_path,times+2)
    save_manual_mutation_tokens_in_file(project_name,version,manual_mutation_dict)



# 根据token序列，针对每个项目构建词表，将token序列数值化
def save_tokenized_mutation_file_ast(project_path,project_name,version):
    non_bug_files = get_non_bug_files(project_name,version)
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            file_path = os.path.join(path,file)
            if "original" in file_path:
                continue
            list1 = file_path.split("/")
            if list1[3] in non_bug_files:
                try:
                    alltokens = get_token_list(file_path)
                    save_ast_token_in_txt(project_name,version,alltokens,"a")
                except FileNotFoundError:
                    continue





def save_ast_token_in_txt(project,version,alltokens,write_type):
    with open("../vocal_vec/{}_{}.txt".format(project,version),write_type) as f:
        file_str = ''
        for i in alltokens:
            try:
                file_str = file_str + i + " "
            except TypeError:
                # print (alltokens)
                print(2)
        file_str += "\n"
        f.write(file_str)

def save_manual_mutation_tokens_in_file(project,version,manual_mutation_dict):
    with open("../numtokens_with_GloVe/{}_{}_with_manual_mutation_file_before_embedding.pkl".format(project,version),"wb") as f:
        pickle.dump(manual_mutation_dict,f)
        

def save_train_set_tokens_in_txt(project,version,token_dict):
    with open("../vocal_vec/{}_{}.txt".format(project,version),"a") as f:
        for i in token_dict.keys():
            file_str = ''
            for j in token_dict[i]:
                file_str = file_str + j + " "
            file_str += "\n"
            f.write(file_str)

def save_train_set_tokens_in_txt_by_version(project,train_version,test_version,token_dict):
    with open("../vocal_vec/{}_{}_{}.txt".format(project,train_version,test_version),"a") as f:
        for i in token_dict.keys():
            file_str = ''
            for j in token_dict[i]:
                file_str = file_str + j + " "
            file_str += "\n"
            f.write(file_str)

 
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
    save_tokenized_ast(project_path,project,version)
    print("{}_{}_without_mutation end".format(project,version))
    print("{}_{}_with_mutation start".format(project,version))
    save_tokenized_mutation_file_ast(project_mutation_file_path,project,version)
    print("{}_{}_with_mutation end".format(project,version))
    print("{}_{}_with_manual_mutation start".format(project,version))
    save_tokenized_manual_mutation_ast(project_path,project,version)
    print("{}_{}_with_manual_mutation end".format(project,version))
    end_time = time.time()
    print("{}_{}_time:{}".format(project,version,end_time-start_time))


if __name__ == '__main__':
    # projects = ["ant"]
    projects_with_version = {'ant':['1.6','1.7'],'jEdit':['4.3'],'ivy':['2.0']}
    # projects_with_version = {'jEdit':['4.3']}
    Processes = []
    for project in projects_with_version.keys():
        for version in projects_with_version[project]:
            p = Process(target=process_for_each_project, args=(project,version))
            Processes.append(p)
    start_time = time.time()
    for i in Processes:
        i.start()
    for i in Processes:
        i.join()
    end_time = time.time()
    print("time:{}".format(end_time-start_time))


