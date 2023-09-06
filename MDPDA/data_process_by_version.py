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
def tokenized_ast(project_path,project_name,version):
    if os.path.exists("../vocabdict/{}.pkl".format(project_name)):
        with open("../vocabdict/{}.pkl".format(project_name),'rb') as f:
            vocabdict = pickle.load(f)
    else:
        vocabdict = {}
    print(len(vocabdict))
    if os.path.exists("../numtokens/{}_{}_without_mutation.pkl".format(project_name,version)):
        with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,version),"rb") as f:
            num_tokens_dict = pickle.load(f)
    else:
        num_tokens_dict = {}
    # vocabdict = {}
    # num_tokens_dict = {}
    index = len(vocabdict.keys())+1
    start = len(project_path)+1
    count =0
    for path,file_dirs,files in os.walk(project_path):
        source_files =[f for f in files if is_code(f)]
        for file in source_files:
            file_path = os.path.join(path,file)
            alltokens = get_token_list(file_path)
            if alltokens == None:
                continue
            # programfile = open(file_path)
            # programtext = programfile.read()
            # ast = javalang.parse.parse(programtext)
            # alltokens = []
            # get_sequence(ast, alltokens)
            # print(alltokens)
            alltokens_without_repeat = list(set(alltokens))

            for i in alltokens_without_repeat:
                if (i not in vocabdict.keys()):
                    vocabdict[i] = index
                    index += 1
            ast_num_tokens_list = []
            for i in alltokens:
                ast_num_tokens_list.append(vocabdict[i])
            # print(file_path[start:].replace('/','.'))
            if file_path[start:].replace('/','.') in num_tokens_dict.keys():
                continue
            num_tokens_dict[file_path[start:].replace('/','.')] = ast_num_tokens_list
            count+=1


    print("num_tokens_dict_length = {}".format(len(num_tokens_dict.keys())))
    # print(vocabdict)
    token_length_list = []
    count1 = 0
    for i,j in num_tokens_dict.items():
        if len(num_tokens_dict[i]) >3000:
            count1 += 1
        token_length_list.append(len(num_tokens_dict[i]))
    # print("token_length_list: {}".format(token_length_list))
    # print("count1 = {}".format(count1))
    # print("count = {}".format(count))
    with open("../vocabdict/{}.pkl".format(project_name),'wb') as f:
        pickle.dump(vocabdict,f)
    with open("../numtokens/{}_{}_without_mutation.pkl".format(project_name,version),"wb") as f:
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
    with open("../vocabdict/{}.pkl".format(project_name),'rb') as f:
        vocabdict = pickle.load(f)
    if os.path.exists("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name,version)):
        with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name,version),"rb") as f:
            num_tokens_dict = pickle.load(f)
    else:
        num_tokens_dict = {}
    index = len(vocabdict.keys()) + 1
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
            alltokens,count = get_mutation_token_list(file_path,7)
            if (count == 0):
                flag = 0
                for i in range(10):
                    alltokens,count = get_mutation_token_list(file_path,7)
                    if count != 0:
                        flag = 1
                        break
                if flag ==0:
                    continue
            current_file_token_dict = {}
            for times in range(5):
                new_file_name = file_name+"_"+str(times)
                alltokens_without_repeat = list(set(alltokens))
                for j in alltokens_without_repeat:
                    if (j not in vocabdict.keys()):
                        vocabdict[j] = index
                        index += 1
                ast_num_tokens_list = []
                for j in alltokens:
                    ast_num_tokens_list.append(vocabdict[j])
                if(ast_num_tokens_list not in current_file_token_dict.values()):
                    current_file_token_dict[new_file_name] = ast_num_tokens_list
                alltokens,count = get_mutation_token_list(file_path,7)
                # if count < times+2:
                #     break
            for i in current_file_token_dict.keys():
                num_tokens_dict[i] = current_file_token_dict[i]
    print(len(num_tokens_dict.keys()))
    # print(vocabdict)    
    with open("../vocabdict/{}.pkl".format(project_name),'wb') as f:
        pickle.dump(vocabdict,f)
    with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name, version), "wb") as f:
        pickle.dump(num_tokens_dict, f)



def is_code(filename):
    return filename.endswith('.java')



def process_for_each_project(project,version):
    project_path = '../dataset_source/{}_{}'.format(project,version)
    project_mutation_file_path = '../mutation_file/{}_{}'.format(project, version)

    start_time = time.time()
    print("{}_{}_without_mutation start".format(project,version))
    start_without_time = time.time()
    tokenized_ast(project_path,project,version)
    end_without_time = time.time()
    without_time = end_without_time-start_without_time
    print("{}_{}_without_mutation end".format(project,version))
    print("{}_{}_with_manual_mutation start".format(project,version))
    start_with_time = time.time()
    tokenized_manual_mutation_ast(project_path,project,version)
    end_with_time = time.time()
    with_time = end_with_time-start_with_time
    print("{}_{}_with_manual_mutation end".format(project,version))
    end_time = time.time()
    print("{}_{}_time:{}".format(project,version,end_time-start_time))
    # with open("./cost_time.txt", "a+") as f:
    #     f.write("{}_{}_without_mutation_cost:{} \n".format(project, version,without_time))
    #     f.write("{}_{}_with_mutation_cost:{} \n".format(project, version,with_time))

if __name__ == '__main__':
    projects_with_version = {'ant':['1.5','1.6','1.7'],"jEdit":['4.0','4.1','4.2'],"synapse":['1.0','1.1','1.2'],"camel":['1.4','1.6'],
                                "ivy":['1.4','2.0'],"xalan":['2.4','2.5']}
    Processes = []
    for project in projects_with_version.keys():
        for version in projects_with_version[project]:
            process_for_each_project(project,version)