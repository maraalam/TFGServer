
class Triples2Sentence:

    def __init__(self, triples):
        self.triples = triples
        self.text = self.__genSentence()

    def __genSentence(self):
        text = ""
        for triple in self.triples:
            text += triple+ " && "
                
        text = text[0:-3]

        return text

    def getText(self):
        return "WebNLG:" + self.text + "</s>"
