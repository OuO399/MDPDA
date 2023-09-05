import random

ARITHMETIC_OPERATORS = ["+","-","*","/","%"]  #AORB
RELATIONAL_OPERATORS = [">","<",">=","<=","==","!="]  #ROR
SHORT_CUT_ASSIGNMENT_OPERATORS = ["+=","-=","*=","/=","%="]  # ASRS
SHORT_CUT_ARITHMETIC_OPERATORS = ["++","--"]  #AORS,AODS
BINARY_CONDITIONAL_OPERATORS = ["||","&&"]  #COR
UNARY_CONDITIONAL_OPERATORS = ["!"]  #COI,COD
BINARY_LOGICAL_OPERATORS = ["|","&","^"]  #LOR
UNARY_LOGICAL_OPERATORS = ["~"]  #LOI,LOD
SHIFT_OPERATORS = ["<<",">>",">>>"]  # SOR

# 当为前置运算符时，只可能出现AORS,AODU,AODS,COD,LOD
# 当为运算符时，只可能出现AORB,ROR,COR,SOR,LOR,ASRS
# 当出现后置运算符时，只可能出现 AORS,AODS
# 插入类型的变异都未实现。 COI,LOI,AOIU,AOIS

def AORB(str1):
    while True:
        index = random.randint(0, 4)
        if(str1 != ARITHMETIC_OPERATORS[index]):
            return ARITHMETIC_OPERATORS[index]

def ROR(str1):
    while True:
        index = random.randint(0, 5)
        if(str1 != RELATIONAL_OPERATORS[index]):
            return RELATIONAL_OPERATORS[index]

def ASRS(str1):
    while True:
        index = random.randint(0, 4)
        if(str1 != SHORT_CUT_ASSIGNMENT_OPERATORS[index]):
            return SHORT_CUT_ASSIGNMENT_OPERATORS[index]

def COR(str1):
    while True:
        index = random.randint(0, 1)
        if(str1 != BINARY_CONDITIONAL_OPERATORS[index]):
            return BINARY_CONDITIONAL_OPERATORS[index]

def LOR(str1):
    while True:
        index = random.randint(0, 2)
        if(str1 != BINARY_LOGICAL_OPERATORS[index]):
            return BINARY_LOGICAL_OPERATORS[index]

def SOR(str1):
    while True:
        index = random.randint(0, 1)
        if(str1 != SHIFT_OPERATORS[index]):
            return SHIFT_OPERATORS[index]

def AORS(str1):
    while True:
        index = random.randint(0, 1)
        if(str1 != SHORT_CUT_ARITHMETIC_OPERATORS[index]):
            return SHORT_CUT_ARITHMETIC_OPERATORS[index]
def AODS(str1):
    return None

def COD(str1):
    return None

def LOD(str1):
    return None

def AODU(str1):
    return None

def mutation(str1,mutation_type):
    if(mutation_type == "operator"):
        if str1 in ARITHMETIC_OPERATORS:
            return AORB(str1)
        if str1 in RELATIONAL_OPERATORS:
            return ROR(str1)
        if str1 in BINARY_CONDITIONAL_OPERATORS:
            return COR(str1)
        if str1 in BINARY_LOGICAL_OPERATORS:
            return LOR(str1)
        if str1 in SHIFT_OPERATORS:
            return SOR(str1)
        # print(str1)
        return str1
    elif(mutation_type == "prefix_operators"):
        if str1 in SHORT_CUT_ARITHMETIC_OPERATORS:
            able_mutation = [AORS,AODS]
            index = random.randint(0, 1)
            return able_mutation[index](str1)
        if str1 == UNARY_CONDITIONAL_OPERATORS:
            return COD(str1)
        if str1 == "-":
            return AODU(str1)
        if str1 in UNARY_LOGICAL_OPERATORS:
            return LOD(str1)
    elif(mutation_type == "postfix_operators"):
        able_mutation = [AORS,AODS]
        index = random.randint(0, 1)
        return able_mutation[index](str1)
    elif(mutation_type == "expressionl"):
        return ASRS(str1)


