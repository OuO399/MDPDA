import javalang
from manual_mutation import mutation
import random

# 将ast token化，token序列包含节点类型和一元二元运算符的信息。
def get_token_list(file_path):
    alltokens = []
    # programfile = open(r'./CachingTokenFilter.java')
    # print(file_path)
    programfile = open(file_path,encoding='gbk')
    # tree = javalang.parse.parse(programfile.read())

    try:
        tree = javalang.parse.parse(programfile.read())
    except:
        # print(file_path)
        return None


    for path, node in tree:
        if path==[] or path==():
            continue
        # print(type(node))
        alltokens.append(node.__class__.__name__)
        # print(node.__class__.__name__)

        if "operator" in node.attrs:
            # print("operator:{}".format(node.operator))
            alltokens.append(node.operator)
        #前缀运算符
        if 'prefix_operators' in node.attrs and node.prefix_operators !=None:
            for i in node.prefix_operators:
                # print("前缀运算符:{}".format(i))
                alltokens.append(i)
        #后缀运算符
        if 'postfix_operators' in node.attrs and node.prefix_operators !=None:
            for i in node.postfix_operators:
                # print("后缀运算符:{}".format(i))
                alltokens.append(i)
            # print(node.operator)
        if 'expressionl' in node.attrs and node.type != '=':
            # print("+=运算符:{}".format(node.type))
            alltokens.append(node.type)
        # print("--------------------------------")
        # print(dir(node))  # 获取node中的属性
        # print(node.attrs)
        # for attr in node.attrs:  # 获取所有属性值
        #     print(getattr(node,attr))
        # print("--------------------------------")
    # print(alltokens)
    return alltokens

def get_mutation_token_list(file_path,num):
    alltokens = []
    programfile = open(file_path, encoding='gbk')
    # tree = javalang.parse.parse(programfile.read())

    try:
        tree = javalang.parse.parse(programfile.read())
    except:
        # print(file_path)
        return None,0

    count = 0
    for path, node in tree:
        if path == [] or path == ():
            continue
        alltokens.append(node.__class__.__name__)

        if "operator" in node.attrs:
            # count += 1
            # if count == num:
            #     str1 = mutation(node.operator,"operator")
            #     if str1 == node.operator:
            #         count -=1
            # else:
            #     str1 = node.operator
            # print("operator:{}".format(node.operator))
            if random.randint(0,1) and count <= num:
                str1 = mutation(node.operator,"operator")
                if str1 != node.operator:
                    count +=1
            else:
                str1 = node.operator
            alltokens.append(str1)
            
        # 前缀运算符
        if 'prefix_operators' in node.attrs and node.prefix_operators != None:
            for i in node.prefix_operators:
                # count += 1
                # if count == num:
                #     str1 = mutation(i,'prefix_operators')
                #     if str1 == None:
                #         continue 
                # else:
                #     str1 = i
                if random.randint(0,1) and count <= num:
                    # print("前缀运算符:{}".format(i))
                    str1 = mutation(i,'prefix_operators')
                    count += 1
                    if str1 == None:
                        continue 
                else:
                    str1 = i
                alltokens.append(str1)
        # 后缀运算符
        if 'postfix_operators' in node.attrs and node.prefix_operators != None:
            for i in node.postfix_operators:
                # count += 1
                # if count == num:
                #     str1 = mutation(i,'postfix_operators')
                #     if str1 == None:
                #         continue 
                # else:
                #     str1 = i
                if random.randint(0,1) and count <= num:
                    # print("后缀运算符:{}".format(i))
                    str1 = mutation(i, 'postfix_operators')
                    count += 1
                    if str1 == None:
                        continue
                else:
                    str1 = i
                alltokens.append(str1)
            # print(node.operator)
        if 'expressionl' in node.attrs and node.type != '=':
            # print("+=运算符:{}".format(node.type))
            # count += 1
            # if count == num:
            #     str1 = mutation(node.type,'expressionl')
            # else:
            #     str1 = node.type
            if random.randint(0,1) and count <= num:
                str1 = mutation(node.type,'expressionl')
                count += 1
            else:
                str1 = node.type

            alltokens.append(str1)
            

        # print("--------------------------------")
        # print(dir(node))  # 获取node中的属性
        # print(node.attrs)
        # for attr in node.attrs:  # 获取所有属性值
        #     print(getattr(node,attr))
        # print("--------------------------------")
    # print(alltokens)
    return alltokens,count

# a1 = get_token_list("./no_mutation.java")
# a2 = get_mutation_token_list("./no_mutation.java")
# print(a1 == a2)
