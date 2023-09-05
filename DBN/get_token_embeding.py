

class TokenEmbedding(object):
    def __init__(self,project_name,train_version,test_version):
        self.project = project_name
        self.train_version = train_version
        self.test_version = test_version
        self.vocal_dict = {}
        self.get_vocal_dict()


    def get_vocal_dict(self):
        with open("../GloVe/models/project/40/{}_{}_{}.txt".format(self.project,self.train_version,self.test_version),"r") as f:
            project_list = f.readlines()
        for i in range(0,len(project_list)):
            str1 = project_list[i]
            file_list = str1.split()
            self.vocal_dict[file_list[0]] = [eval(j) for j in file_list[1:]]
    
    def get_token_embedding(self,alltokens):
        file_embedding = []
        for i in alltokens:
            try:
                file_embedding.append(self.vocal_dict[i])
            except KeyError:
                file_embedding.append(self.vocal_dict["<unk>"])
        return file_embedding