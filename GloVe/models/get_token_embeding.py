

class Embedding(object):
    def __init__(self,project_name,version):
        self.project = project_name
        self.version = version
        self.vocal_dict = {}
        get_vocal_dict()


    def get_vocal_dict():
        with open("./project/40/{}_{}".format(self.project,self.version),"r") as f:
            project_list = f.readlines()
        for i in range(0,len(project_list)):
            str1 = project_list[i]
            file_list = str1.split()
            self.vocal_dict[file_list[0]] = [eval(j) for j in file_list[1:]]
    
    def get_token_embedding(alltokens):
        file_embedding = []
        for i in alltokens:
            file_embedding.append(self.vocal_dict[i])
        return file_embedding