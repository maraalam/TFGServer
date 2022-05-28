from glob import glob
import re
import pandas as pd
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from os import remove,mkdir

from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV



class WebNLGDataset:

    def __init__(self, url = None):
        """
            Inicialización del conjunto de datos.
            Incluye tareas de preprocesamiento y limpieza de los datos de WebNLG.

            Devuelve los datos del conjunto de forma de diccionarios de pares -nombre del conjunto (train, test, dev)-DataFrame con los datos-.
        """

        self.URL_DATABASE = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en"
        self.dataset = {'train' : [], 'test': {'test':[],'train':[]},'dev':[]}

        self.genDataset(url)
        
    
    def genDataset(self, url = None):

        data_url = self.checkUrl(url = url)

        self.extractAllData(data_url)

        data = self.preprocessAllData()

        for i in range(len(data)):
            data[i] = self.parseData(data[i])
            data[i] = self.randomShuffle(data[i]).drop_duplicates(subset=None, 
                                keep='first', 
                                inplace=False, 
                                ignore_index=False)

        mkdir('data/cleaned')
        self.save_csv(data[0],'data/cleaned/webNLG2020_train.csv')
        self.save_csv(data[1],'data/cleaned/webNLG2020_dev.csv')
        self.save_csv(data[2],'data/cleaned/webNLG2020_test.csv')
        self.save_csv(data[3],'data/cleaned/webNLG2020_testtrain.csv')

        self.dataset['train'] = data[0]
        self.dataset['dev'] = data[1]
        self.dataset['test']['test'] = data[2]
        self.dataset['test']['train'] = data[3]

        remove('data/webnlg.zip')


    def preprocessAllData(self):
        remove('data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/test/rdf-to-text-generation-test-data-without-refs-en.xml')
        remove('data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/test/semantic-parsing-test-data-with-refs-en.xml')
        
        sourceURL = "data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/"

        mkdir('data/parsed')
        self.preprocessData(sourceURL+"train/**/*.xml","data/parsed/webNLG2020_train.csv", typefile='train')
        self.preprocessData(sourceURL+"dev/**/*.xml","data/parsed/webNLG2020_dev.csv", typefile='train')

        self.preprocessData(sourceURL+"test/*.xml","data/parsed/webNLG2020_test.csv", typefile='test')
        self.preprocessData(sourceURL+"train/**/*.xml","data/parsed/webNLG2020_testtrain.csv", typefile='test')

        return [self.load_csv('data/parsed/webNLG2020_train.csv'),self.load_csv('data/parsed/webNLG2020_dev.csv'),
                self.load_csv('data/parsed/webNLG2020_test.csv'),self.load_csv('data/parsed/webNLG2020_testtrain.csv')]
    

    def preprocessData(self, sourceUrl, targetUrl = 'webNLG2020.csv', typefile ="train"):
        files = glob(sourceUrl, recursive=True)
        data_list=[]

        for file in files:
            tree = ET.parse(file)
            root = tree.getroot()
            for entries in root: #entries
                for entry in entries: #entry
                
                    structure_master=[]
                    unstructured= []

                    if(typefile=="train"):
                        m = entry.findall("modifiedtripleset")
                    else:
                        m = entry.findall("modifiedtripleset")
                        
                    for modifiedtripleset in m: 
                        triples= (' && ').join([triple.text for triple in modifiedtripleset])
                        structure_master.append(triples)

                    
                    for lex in entry.findall("lex"): 
                        unstructured.append(lex.text)

                    triples_num = int(entry.attrib.get("size"))

                    if(typefile=="train"):
                        for text in unstructured:
                            for triple in structure_master:
                                data_list.append([triple,text])
                    else:
                        for structure in structure_master:
                            data_list.append([structure,unstructured])

        mdata_dct={"input_text":[], "target_text":[]}

        for item in data_list:
            mdata_dct['input_text'].append(item[0])
            mdata_dct['target_text'].append(item[1])

        df=pd.DataFrame(mdata_dct)

        df.to_csv(targetUrl)


    def load_csv(self, sourceUrl):
        return pd.read_csv(sourceUrl, index_col=[0])


    def parseInstance(self,example):
        # remove @en
        example = re.sub('@en','', example)

        # change _ to ' '
        #example = re.sub('[_]',' ', example)
        #example = re.sub('"\"','" ', example)
        #example = re.sub('"\"','" ', example)
        example = re.sub("associatedBand/associatedMusicalArtist",'associatedBand',example)
        # remove urls
        # example = re.sub("\<http.*[^\>]\>", '',example)

        #  split relations according to uppercase tokens 
        # # triplets = re.split("&&", example)
        # # for triple in triplets:
        # #   entity = re.split("\|", triple)[1]
        # #   entity2 = entity[1].upper() + entity[2:]
        # #   uppercase = re.findall(r'[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)', entity2)
        # #   if(len(uppercase)>1):
        # #     uppercase = ' '.join(uppercase)
        # #     example = re.sub("{}".format(entity), " {} ".format(uppercase.lower()), example) 

        example = re.sub(r"xsd:[^\s]*\s", "",example)
        example = re.sub(r"xsd:[^\s]*$", "",example)

        example = re.sub('\^','', example)
        return example

    def parseTarget(self,example):
        example = re.sub(r"\.", " .",example)
        example = re.sub(r"F .C .", "F.C.", example)
        example = example.replace(',',' ,')
        example = example.replace('(','( ')
        example = example.replace(')',' )')
        return example


    def parseData(self,df):
        df['input_text'] = df['input_text'].map(self.parseInstance)
        df['target_text'] = df['target_text'].map(self.parseTarget)
        return df


    def randomShuffle(self, df, random_state = 13):
        return df.sample(frac = 1, random_state = random_state)


    def save_csv(self, df, targetUrl):
        df.to_csv(targetUrl)

    
    def checkUrl(self, url : str):
        if url is None:
            url = self.URL_DATABASE
        return url
   

    def extractAllData(self,url):
        urllib.request.urlretrieve(url, 'data/webnlg.zip')

        with zipfile.ZipFile('data/webnlg.zip', 'r') as zip_ref:
            zip_ref.extractall('data/webnlg')



if __name__ == "__main__":
    dataset = WebNLGDataset()