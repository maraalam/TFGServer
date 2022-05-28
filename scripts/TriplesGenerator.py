import pandas as pd
import json

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json

class TriplesGenerator:

    
    def __init__(self, path = None):
        if(path==None):
            self.data_df = self.__generateData()
        else:
            self.data_df = self.__loadFile(path)
         

    #GETTERS
    def getData(self):
        return self.data_df


    def getTriples(self):
        return self.triples


    #OTHER METHODS
    def saveFile(self, path = "data/example.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=4)


        
    def genDataFrame(self, data : list):

        #Extract info from datalist
        source = [d['data'][0] for d in data]
        target = [d['data'][2] for d in data]
        relation = [d['data'][1] for d in data]
        stage = [d['tags']['stage'] for d in data]
        themes = [d['tags']['themes'] for d in data]


        # create dataframe from lists
        data_df = pd.DataFrame({'source':source, 'relation':relation, 'target':target, 'stage':stage, 'themes':themes})

        # create a directed-graph from a dataframe
        #graph = nx.from_pandas_edgelist(graph_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

        #graph.show()
        return data_df


    def to_json(self):
        buf = self.data_df.to_json(orient="records")
        return json.loads(buf)


    #PRIVATE METHODS
    def __loadFile(self, path):
        return pd.read_json(path)
        
    def __generateData(self): 
        data = self.__genData()
        return self.genDataFrame(data)

    def __genData(self):
        data = []
        #tags => 
        #basic, family, parents, friends, school, studies, love, university, work, marriage, children, living place,
        #     hobbies,  feelings

        #stages of life => early childhood (18 months to 6 years), late childhood (6 years to 13 years),
        #     adolescence (13 years to 20 years), early adulthood (20 years to 30 years),
        #     middle adulthood (30 years to 65 years), late adulthood (+64 years)

        
        data = []

        data.append(self.__dataConstructor("Elisa", "birth place","Lugo, Galicia", "timeless",['basic']))
        data.append(self.__dataConstructor("Elisa", "birth date","18-05-1937", "timeless",['basic']))
        data.append(self.__dataConstructor("Elisa", "age","82", "timeless",['basic']))
        data.append(self.__dataConstructor("Elisa", "father's name", "Juan", "timeless", ["basic"]))
        data.append(self.__dataConstructor("Elisa", "mother's name","Eva", "timeless", ["basic"]))

        #family parents
        data.append(self.__dataConstructor("Juan", "occupation", "labrador", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Juan", "wife", "Eva", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Juan", "birth place", "Santander", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Juan", "birth date", "25-11-1911", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Eva", "birth place", "Barcelona", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Eva", "birth date", "12-01-1912", "timeless", ["family","parents"]))
        data.append(self.__dataConstructor("Eva", "occupation", "seamstress", "timeless", ["family","parents"]))
        
        #family siblings
        data.append(self.__dataConstructor("Elisa", "part of", "large family", "timeless", ["family","siblings"]))
        data.append(self.__dataConstructor("Elisa", "number of sisters", "2", "timeless", ["family","siblings"]))
        data.append(self.__dataConstructor("Elisa", "number of brothers", "6", "timeless", ["family","siblings"]))  
        data.append(self.__dataConstructor("Elisa", "the youngest of", "her siblings", "timeless", ["family","siblings"])) 

        #late childhood
        data.append(self.__dataConstructor("Elisa", "educated at", "La Salle's school", "late chilhood", ["school"]))        
        data.append(self.__dataConstructor("La Salle's school", "was", "all-girl catholic", "late chilhood", ["school"]))
        data.append(self.__dataConstructor("Elisa", "favorite subjects", "history and drawing", "late chilhood", ["school"]))

        
        data.append(self.__dataConstructor("Elisa", "best friend", "Veronica", "late chilhood", ["friends"]))
        data.append(self.__dataConstructor("Elisa", "best friend", "Luis", "late chilhood", ["friends"]))
        data.append(self.__dataConstructor("Elisa", "favorite games", "hide-and-seek", "late chilhood", ["friends"]))
        data.append(self.__dataConstructor("Elisa", "go swimming with", "friends", "late chilhood", ["friends"]))
               


        #adolescence
        data.append(self.__dataConstructor("Elisa", "went to", "new high school", "adolescence", ["school"]))
        data.append(self.__dataConstructor("New high school", "name", "Casas School", "adolescence", ["school"]))
        data.append(self.__dataConstructor("Elisa", "met", "new friends", "adolescence", ["school"]))
        data.append(self.__dataConstructor("Elisa", "worked at", "workshop sewing", "adolescence", ["work"]))
        data.append(self.__dataConstructor("Elisa", "sewed", "pretty dresses", "adolescence", ["work"]))
        data.append(self.__dataConstructor("Elisa", "liked", "her work", "adolescence", ["work"]))
        
        data.append(self.__dataConstructor("Elisa", "help her mother with", "housework", "adolescence", ["family"]))
        data.append(self.__dataConstructor("Elisa", "prepare", "bread every day", "adolescence", ["family"]))
        data.append(self.__dataConstructor("Elisa", "bring bread", "to customers", "adolescence", ["family"]))

        #early adulthood
        data.append(self.__dataConstructor("Elisa", "educated at", "Santiago's university", "early adulthood", ["university"]))
        data.append(self.__dataConstructor("Elisa", "career", "education", "early adulthood", ["university"]))
        data.append(self.__dataConstructor("Elisa", "meet new friends", "at university", "early adulthood", ["university"]))
        data.append(self.__dataConstructor("Elisa", "best friend", "Maria", "early adulthood", ["university"]))
        data.append(self.__dataConstructor("Elisa", "best friend", "Laura", "early adulthood", ["university"]))

        data.append(self.__dataConstructor("Elisa", "start new job", "at 20 years old", "early adulthood", ["work"]))
        data.append(self.__dataConstructor("Elisa", "worked in", "textile factory", "early adulthood", ["work"]))
        data.append(self.__dataConstructor("textile factory", "location", "Burgos", "early adulthood", ["work"]))

        data.append(self.__dataConstructor("Elisa", "meet", "Sergio", "early adulthood", ["love"]))
        data.append(self.__dataConstructor("Sergio", "ask Elisa", "to dance", "early adulthood", ["love"]))
        data.append(self.__dataConstructor("Elisa", "fall in love with", "Sergio", "early adulthood", ["love"]))
        data.append(self.__dataConstructor("Elisa", "started dating", "Sergio", "early adulthood", ["love"]))

        #late adulthood
        data.append(self.__dataConstructor("Elisa", "state of life", "adulthood", "late adulthood", ["work"]))
        data.append(self.__dataConstructor("Elisa", "bought", "place in Santander", "late adulthood", ["work"]))
        data.append(self.__dataConstructor("Elisa", "set up a", "sewing shop", "late adulthood", ["work"]))
        data.append(self.__dataConstructor("Elisa", "worked", "hard", "late adulthood", ["work"]))

        data.append(self.__dataConstructor("Elisa", "marry", "Sergio", "late adulthood", ["family","love"]))
        data.append(self.__dataConstructor("Elisa", "wedding place", "beautiful church", "late adulthood", ["family","love"]))
        data.append(self.__dataConstructor("Elisa and Sergio", "had", "2 children", "late adulthood", ["family","love"]))
        data.append(self.__dataConstructor("Elisa and Sergio", "daughter", "Silvia", "late adulthood", ["family","love"]))
        data.append(self.__dataConstructor("Elisa  and Sergio", "son", "Julio", "late adulthood", ["family","love"]))

        return data


    def __dataConstructor(self, subject : str, attribute : str, objec : str, stageOfLife : str, themes : list):
        return {"data":[subject, attribute, objec], "tags":{"stage":stageOfLife,"themes":themes}}     

if __name__ == "__main__":
    triples_generator = TriplesGenerator()
    triples_generator.saveFile('data/elisaStoryLife.json')