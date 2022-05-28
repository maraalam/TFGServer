
from .Triple import Triple

class TripleList(list):
    
    def __init__(self, data):
        tripleList = self.__generateTriples(data)
        self = super().__init__(tripleList)
        return self


    #OTHER PUBLIC METHODS
    def triplesByTags(self):
        triplesByTags = {}
    
        for triple in self:
            stage = triple.getStage()

            if stage not in triplesByTags:
                triplesByTags[stage] = {}
                triplesByTags[stage]['tag'] = {}         

            predecesor = triplesByTags[stage]
            for theme in triple.getThemes()[:-1]:
                p = predecesor['tag']
                if theme not in p:
                    p[theme] = {}

                if 'tag' not in p[theme]:
                    p[theme]['tag'] = {}

                predecesor = p[theme]
            
            theme = triple.getThemes()[-1]

            if theme not in predecesor['tag']:
                predecesor['tag'][theme] = {}
            
            if 'input' not in predecesor['tag'][theme]:
                predecesor['tag'][theme]['input'] = []

            predecesor['tag'][theme]['input'].append(triple.getInput())

        return triplesByTags

    def filterbyNode(self, node):
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveNode(node)):
                filter_triples.append(triple)

        return filter_triples

    def filterbyStage(self, stage):
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveStage(stage)):
                filter_triples.append(triple)

        return filter_triples

    def filterbyTheme(self, theme):
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveTheme(theme)):
                filter_triples.append(triple)

        return filter_triples
      
    def getTagThemes(self):
        aux = set()
        for triple in self:
            for i in triple.getThemes():
                aux.add(i)
        return aux

    def getTagTrees(self):
        triplesByTags = {}
    
        for triple in self:
            stage = triple.getStage()

            if stage not in triplesByTags:
                triplesByTags[stage] = {}      

            predecesor = triplesByTags[stage]
            for theme in triple.getThemes()[:-1]:
                p = predecesor
                if theme not in p:
                    p[theme] = {}


                predecesor = p[theme]
            
            theme = triple.getThemes()[-1]

            if theme not in predecesor:
                predecesor[theme] = {}
            

        return triplesByTags

    def getTagStages(self):
        aux = list()
        for triple in self:
            if triple.getStage() not in aux:
                aux.append(triple.getStage())
        return aux

        
    #LIST METHODS
    def __add__(self, other):
        return TripleList(list.__add__(self,other))

    def __getslice__(self,i,j):
        return TripleList(list.__getslice__(self, i, j))

    #PRIVATE METHODS
    def __generateTriples(self, data):
        triples = []
        for item in data:
            try:
                triple = Triple(item['source'], item['relation'], item['target'], item['stage'], item['themes'])
            
                triples.append(triple)
            except:
                print("ERROR: " + str(item))

        return triples
        
    
        