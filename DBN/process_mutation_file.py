import os
import javalang
from javalang.ast import Node
import os
import pickle
from get_token_list import get_token_list,get_mutation_token_list
import pandas as pd
import re



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
    with open("../vocabdict/{}_{}.pkl".format(project_name,version),'rb') as f:
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
            alltokens,count = get_mutation_token_list(file_path,1)
            if (count == 0):
                break
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
                save_ast_token_in_txt(alltokens)
                alltokens,count = get_mutation_token_list(file_path,times+2)
            for i in current_file_token_dict.keys():
                num_tokens_dict[i] = current_file_token_dict[i]
    print(len(num_tokens_dict.keys()))
    # print(vocabdict)    
    with open("../vocabdict/{}_{}.pkl".format(project_name,version),'wb') as f:
        pickle.dump(vocabdict,f)
    with open("../numtokens/{}_{}_with_manual_mutation_file.pkl".format(project_name, version), "wb") as f:
        pickle.dump(num_tokens_dict, f)


# 根据token序列，针对每个项目构建词表，将token序列数值化
def tokenized_mutation_file_ast(project_path,project_name,version):
    with open("../vocabdict/{}_{}.pkl".format(project_name,version),'rb') as f:
        vocabdict = pickle.load(f)
    # with open("../numtokens/{}_{}.pkl".format(project_name,version),"rb") as f:
    #     num_tokens_dict = pickle.load(f)
    if os.path.exists("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name,version)):
        with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name,version),"rb") as f:
            num_tokens_dict = pickle.load(f)
    else:
        num_tokens_dict = {}
    # vocabdict = {}
    # num_tokens_dict = {}
    index = len(vocabdict.keys())+1
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
                    save_ast_token_in_txt(alltokens)
                except FileNotFoundError:
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
                # print(file_path[start:].replace('\\','.'))
                file_key_name = list1[3]+".java_"+list1[-2] #list[-2]为变异算子
                num_tokens_dict[file_key_name] = ast_num_tokens_list
    print(len(num_tokens_dict.keys()))
    # print(vocabdict)
    with open("../vocabdict/{}_{}.pkl".format(project_name,version),'wb') as f:
        pickle.dump(vocabdict,f)
    with open("../numtokens/{}_{}_with_mutation_file.pkl".format(project_name,version),"wb") as f:
        pickle.dump(num_tokens_dict,f)



def is_code(filename):
    return filename.endswith('.java')

# if __name__ == '__main__':
#     projects = ["ant"]
#     for project in projects:
#         version = "1.7"
#         project_path = './mutation_file/{}_{}'.format(project,version)
#         # print(project_path)
#         tokenized_mutation_file_ast(project_path,project,version)

