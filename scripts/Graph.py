

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import ipycytoscape

class Graph:
    def __init__(self, data : pd.DataFrame):
        self.graph = self.__getGraph(data)


    def getGraph(self):
        return self.graph

    def show(self):
        pos = nx.random_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color = 'r', node_size = 100, alpha = 1)
        ax = plt.gca()
        for e in self.graph.edges:
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                        ),
                                        ),
                        )
        plt.axis('off')
        plt.show()

    def __getGraph(self,data):

        #edges = [{'relation': d['relation'],'stage':d['stage'],'themes':d['themes']} for ind,d in data.iterrows()]
        
        #create dataframe from lists
        #data_df = pd.DataFrame({'source': data['source'], 'edge':edges, 'target':data['target']})

        # create a directed-graph from a dataframe
        graph = nx.from_pandas_edgelist(data, "source", "target",  edge_attr=['relation', 'stage', 'themes'], create_using=nx.MultiDiGraph())
        return graph
