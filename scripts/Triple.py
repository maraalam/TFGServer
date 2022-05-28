class Triple(dict):
    """Semantic Triple"""
    def __init__(self, subject : str, attribute : str, object : str, stage: str, themes: list):
        triple = {'input' : subject + " | " + attribute + " | " + object, 'tags' : {'stage' : stage, 'themes' : themes}}
        self = super().__init__(triple)

    def getTags(self):
        return self['tags']
    
    def getStage(self):
        return self['tags']['stage']

    def getThemes(self):
        return self['tags']['themes']

    def haveTheme(self,theme):
        return theme in self['tags']['themes']

    def haveStage(self, stage):
        return self['tags']['stage']==stage
    
    def haveNode(self, node):
        return self['input'].split("|")[0].strip() == node.strip() or self['input'].split("|")[2].strip() == node.strip()

    def getInput(self):
        return self['input']

        