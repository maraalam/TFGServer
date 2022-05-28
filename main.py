import math
import dash
from dash import dcc
from dash import html,State
import numpy as np
import dash_daq as daq

import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from colour import Color
from textwrap import dedent as d
import json




from scripts.Model import Model
from scripts.TripleList import TripleList
from scripts.Triples2Sentence import Triples2Sentence
from scripts.TriplesClustering import TriplesClustering
from scripts.TriplesGenerator import TriplesGenerator

import dash_bootstrap_components as dbc
from dash.dependencies import ClientsideFunction, Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,external_scripts=["https://cdnjs.cloudflare.com/ajax/libs/dragula/3.7.2/dragula.min.js"])
app.title = "Generación de historias de vida"

INITIAL_TAGS = []
FILEURL = "data/elisaStoryLife.json"
SELECTED_TAGS = []  
ORDER_TAGS = []

TRIPLES = TriplesGenerator(FILEURL)

model = Model() 

model.load_model("models/t5_1epoch_30000examples")


def printAllText(triplesClusters, do_sample = False, num_beams = 1, no_repeat_ngram_size = 1, min_length = 0, 
max_length = 500, top_k = 50, top_p = 0.92, temperature = 1.0, penalty = 1.0, num_beam_groups=1,num_return_sequences=1):
    final_text = ""
    
    for item in triplesClusters:
        if(item=='input'):
            cluster_set = set(triplesClusters[item]['cluster'])
            for c in cluster_set:             
                index = np.where(triplesClusters[item]['cluster']==c)[0]
                input =  []
                for i in index:
                    input.append(triplesClusters[item]['input'][i])
                
                triple2sen = Triples2Sentence(input)
                prompt = triple2sen.getText()
                inputs_id = model.encode(prompt=prompt)
                num_triples = len(prompt.split('&&'))
                outputs = model.generateText(encode_text = inputs_id, do_sample = do_sample, num_beams = num_beams, num_beam_groups=num_beam_groups,no_repeat_ngram_size = no_repeat_ngram_size,
     min_length = min_length*num_triples, max_length = max_length*num_triples*3, top_k = top_k, top_p = top_p, temperature = temperature, penalty = penalty, num_return_sequences = 1)

                for output in outputs:
                    text = model.decode(output)
                    text = text.replace('<pad>', '')
                    text = text.replace('</s>', '')
                    final_text += text

        else:
            final_text+=printAllText(triplesClusters= triplesClusters[item],do_sample = do_sample, num_beams = num_beams, no_repeat_ngram_size = no_repeat_ngram_size,
            min_length = min_length, max_length = max_length, top_k  = top_k, top_p = top_p, temperature  = temperature, penalty = penalty, num_beam_groups = num_beam_groups) + "\n"
    return final_text


def filteredTriplesByTags(triplesTree, themesSelection):
    if(len(themesSelection)==0):
        return []
    else:
        diccionario = {}
        for t in themesSelection:
            if 'tag' in triplesTree[t]:
                lista = filteredTriplesByTags(triplesTree[t]['tag'], themesSelection[t])
                if(lista==[]):
                    diccionario[t] = triplesTree[t]
                else:
                    diccionario[t] = lista
            else:        
                diccionario[t] = triplesTree[t]

        return diccionario



def clusteringByTags(themesSelection):

    triplesClustersList = {}
    for tag in themesSelection:

        if tag == 'input':
            clustering = TriplesClustering(themesSelection)
            clustering.genClusters()
            return {'input' : clustering.getTriples()}
               
        triplesClustersList[tag] = clusteringByTags(themesSelection[tag])
    
    return triplesClustersList


def listTags2Tree(list_tags):
    list_dict_tags = []
    for lt in list_tags:
        list_dict_tags.append(lt.split('_'))

    dict_tags ={}
    for ldt in list_dict_tags:
        
        pred = dict_tags
        for tag in ldt:
            if(tag not in pred): #Si no estaba la añado
                pred[tag] = {}
                
            pred=pred[tag]

    return dict_tags


def network_graph(fileurl, tags):
    if(len(tags)==0):
        figure = {
            "layout": go.Layout(title='Visualización interactiva', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure

    edges = TRIPLES.getData().copy()


    #######     FILTERING       #########

    nodesSet = set()
    #tags_selected = listTags2Tree(tags)

    stages = []
    for stage in tags:
        stages.append(stage.split('_')[0])


    for index,item in edges.iterrows():

        conservar = 0
        
        if(len(tags)==0):
            conservar = 1
        
        j=0    
        while(j<len(tags) and not conservar):
            theme = tags[j]
            count = sum(1 for s in tags if theme in s)
            if(count==1): # If only appears 1 time
                
                if(item['stage'] == theme.split('_')[0]): # If the stage is correct
                    if(theme in stages):
                        conservar = 1
                    else:    
                        # If there arent themes in tags selected
                        allthemes = 1
                        for i in range(0, len(theme.split('_'))-1):
                            if(len(item['themes']) <= i or item['themes'][i]!=theme.split('_')[i+1]):   
                                allthemes = 0
                            else:
                                
                                allthemes &= 1
                        conservar = allthemes
                        
            j = j+1
           
        if(not conservar):
            edges.drop(axis=0, index=index, inplace=True)
            
        else:            
            nodesSet.add(item['source'])
            nodesSet.add(item['target'])
        

    # to define the centric point of the networkx layout
    shells=[]
    shell1=[]
    shell1.append('Elisa')
    shells.append(shell1)
    shell2=[]
    for ele in nodesSet:
        if ele!='Elisa':
            shell2.append(ele)
    shells.append(shell2)
    
    G = nx.from_pandas_edgelist(edges, 'source', 'target', ['source','target','relation','stage','themes'], create_using=nx.MultiDiGraph())


    if len(shell2)<1:
        pos = nx.drawing.layout.shell_layout(G, shells)
    else:
        pos = nx.drawing.layout.spring_layout(G,k=2/(G.number_of_nodes()), seed=18)



    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])


    if len(shell2)==0:
        traceRecode = []  # contains edge_trace, node_trace, middle_node_trace

        node_trace = go.Scatter(x=tuple([1]), y=tuple([1]), text=tuple([str("Elisa")]), textposition="bottom center",
                                mode='markers+text',
                                marker={'size': 50, 'color': 'LightSkyBlue'})
        traceRecode.append(node_trace)

        node_trace1 = go.Scatter(x=tuple([1]), y=tuple([1]),
                                mode='markers',
                                marker={'size': 50, 'color': 'LightSkyBlue'},
                                opacity=0)
        traceRecode.append(node_trace1)

        figure = {
            "data": traceRecode,
            "layout": go.Layout(title='Visualización interactiva', showlegend=False,
                                margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                                xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                                height=600
                                )}
        return figure


    traceRecode = []  # contains edge_trace, node_trace, middle_node_trace
    ############################################################################################################################################################
    
    colors = list(Color('lightcoral').range_to(Color('darkred'), len(G.edges())))
    colors = ['rgb' + str(x.rgb) for x in colors]

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        trace = go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                           mode='lines',
                           marker=dict(color='darkgray'),
                           line_shape='spline',
                           opacity=1)
        traceRecode.append(trace)
        index = index + 1
    ###############################################################################################################################################################
    sizes = [d[1] for d in G.degree]
    maxis = max(sizes)
    sizes = [((math.log(s)*10)+20) for s in sizes]
    text_sizes = [int(((math.log(s)*10)+20)/5) for s in sizes]
    
    node_trace = go.Scatter(x=[], y=[], hovertext=[], text=[], mode='markers+text',textposition="middle center",
                            hoverinfo="text", 
                            marker={'size': sizes, 'color': 'lightcoral'},
                            textfont={
                                'size': text_sizes
                            })

    node_trace['text'] += tuple(G.nodes())
    index = 0
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        hovertext = "text: " + str(node)
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(node_trace)
    ################################################################################################################################################################
    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'},
                                    opacity=0)

    index = 0
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        hovertext = "From: " + str(G.edges[edge]['source']) + "<br>" + "To: " + str(
            G.edges[edge]['target']) + "<br>" + "Relation: " + str(
            G.edges[edge]['relation']) + "<br>" + "Stage: " + str(G.edges[edge]['stage'] +"<br>"+"Themes: " + str(G.edges[edge]['themes']))
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
        index = index + 1

    traceRecode.append(middle_hover_trace)
    #################################################################################################################################################################
    figure = {
        "data": traceRecode,
        "layout": go.Layout(title='Visualización Interactiva', showlegend=False, hovermode='closest',
                            margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                            height=600,
                            clickmode='event+select',
                            annotations=[
                                dict(
                                    ax=(G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0]) / 2,
                                    ay=(G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1]) / 2, axref='x', ayref='y',
                                    x=(G.nodes[edge[1]]['pos'][0] * 3 + G.nodes[edge[0]]['pos'][0]) / 4,
                                    y=(G.nodes[edge[1]]['pos'][1] * 3 + G.nodes[edge[0]]['pos'][1]) / 4, xref='x', yref='y',
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=4,
                                    arrowwidth=1,
                                    opacity=1
                                ) for edge in G.edges]
                            )}
    return figure


def recursivetags(name, tags, pad, before_id, class_name):

    if(len(tags)==0):
        button = dbc.Button(name, n_clicks=0, class_name='theme tag', id=before_id)
        return dbc.Card(children = button, style={'marginLeft':str(pad)+'em'}), [before_id]
    else:
        tree = []
        list_id = []
    
        tree.append(dbc.Button(name, n_clicks=0, 
                        class_name=class_name, id=before_id))
        list_id.append(before_id)

        for t in tags:
            tree_children, ids_used = recursivetags(t, tags[t], pad+1, before_id+'_'+t,'theme tag')

            list_id.extend(list(ids_used))
            tree.append(tree_children)

        drag = html.Div(className = "draggable",children=tree)
        return dbc.Card(children = drag, style={'marginLeft':str(pad)+'em'}), list_id


def tags_network(fileUrl):


    tg = TriplesGenerator(fileUrl)
    triples = TripleList(data = tg.to_json())
    stages = triples.getTagStages()
    content = []
    list_id = []
    
    for stage in stages:
        tags =  triples.getTagTrees()[stage]
           
        c, l = recursivetags(stage, tags, 1, stage,'theme stage')
        content.append(c)
        list_id.extend(list(l))

    return content, list_id


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div(
    id="ppal_container",
    children=[

    #----------- titulo -----------#
    html.Div([html.H1(app.title)],
            className="row", 
            id="titulo"),

    #----------- contenido -----------#
    html.Div(
        className="row",
        children=[

            #----------- columna de tags -----------#
            html.Div(
                className="two columns",
                children=[
                html.Div(
                        className="twelve columns ",
                        style={'display':'grid'},
                        children=[
                            html.H2("Lista de tags", className="subtitle"),
                            
                            html.Div(
                                id="tags_header",
                                children=[
                                    html.Button('Todas', id="allTags", className="button options_button", n_clicks=0, style={'float':'left'}),
                                    html.Button('Ninguna', id="noneTags", className="button options_button", n_clicks=0, style={'float':'right'}),                              
                                    
                                ]
                            ),                               
                            
                            html.Div(
                                id="drag_container", 
                                className="container", 
                                children = tags_network(fileUrl=FILEURL)[0]
                            ),                            
                        ]
                    )
                ]
            ),
            
            #----------- grafo de conocimiento y resultados de generación-----------#
            html.Div(
                className="eight columns",
                children=[
                    
                    dcc.Graph(id="grafo-conocimiento",figure=network_graph(FILEURL, INITIAL_TAGS)),
                
                    html.Div(
                        className="form-outline",
                        children=[
                            html.Div(
                                className="Header",
                                style={'style':'inline-block', 'width':'100%'},
                                children=[
                                    html.H2('Salida',style={'float':'left'}),
                                    html.Button('Generar', id="generator", n_clicks=0, style={'float':'right'}, className="button")                                
                                ]
                            ),
                            dcc.Textarea(
                                className="form-control",
                                id="outputGenerator",
                                placeholder="No hay salida",
                                rows='10'
                            )
                        ]
                    
                    ),
                    
                    html.Div(
                        className="filter",
                        children = [
                            dcc.Markdown('''
                                #### Aleatoriedad
                            '''),
                            dbc.Checklist(
                                options=[
                                    {"label": "(No | Si)", "value": 1},
                                ],
                                value=[0],
                                id="do_sample",
                                switch=True,
                            ),
                        ]
                    ),
                    html.Div(
                        className="filter experto",
                        children = [
                            dcc.Markdown('''
                                #### Modo Experto
                            '''),
                            dbc.Checklist(
                                options=[
                                    {"label": "(No | Si)", "value": 1},
                                ],
                                value=[0],
                                id="experto",
                                switch=True,
                            ),
                        ]
                    ),
                    html.Div(
                        className="filter",
                        children = [
                            dcc.Markdown('''
                                #### Longitud del texto
                            '''),
                            dcc.Slider(10, 90,
                                step=10,
                                id="max_len",
                                value=50,
                                marks={i: '{}'.format(j) for i,j in zip([10,50,90],['Corto','Medio','largo'])},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                     ),
                     html.Div(
                        className="filter hidden",
                        id="m_beams1",
                        children = [
                            dcc.Markdown('''
                                #### Entrada personalizada (si tiene contenido generará la salida a partir de estos datos)

                                Formato : sujeto1 | verbo1 | atributo1 && sujeto2 | verbo2 | atributo2 ...
                            '''),
                            dcc.Textarea(
                                className="form-control",
                                id="entradaPersonalizada",
                                placeholder="No hay entrada",
                                rows='10'
                            )
                        ]

                    ),
                    html.Div(
                        className="filter hidden",
                        id="m_beams2",
                        children = [
                            dcc.Markdown('''
                                #### Num beams
                            '''),
                            dcc.Slider(1, 50,
                                step=1,
                                id="num_beams",
                                value=18,
                                marks={i: '{}'.format(i) for i in range(1,50,5)},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]

                    ),
                    html.Div(
                        className="filter hidden",
                        id="m_beams3",
                        children = [
                            dcc.Markdown('''
                                #### Num Beam Groups
                            '''),
                            dcc.Slider(1, 50,
                                step=1,
                                id="num_beam_groups",
                                value=3,
                                marks={i: '{}'.format(i) for i in range(1,50,5)},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    ),   
                    html.Div(
                        className="filter hidden",
                        id="m_beams4",
                        children = [
                            dcc.Markdown('''
                                #### No repeat ngram size
                            '''),
                            dcc.Slider(1, 50,
                                step=1,
                                id="no_repeat_ngram_size",
                                value=7,
                                marks={i: '{}'.format(i) for i in range(1,50,5)},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    ),
                    html.Div(
                        className="filter hidden",
                        id="m_beams5",
                        children = [
                            dcc.Markdown('''
                                #### Top k
                            '''),
                            dcc.Slider(0, 100,
                                step=1,
                                id="top_k",
                                value=50,
                                marks={i: '{}'.format(i) for i in range(0,100,5)},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    ),
                    
                    html.Div(
                        className="filter hidden",
                        id="m_beams6",
                        children = [
                            dcc.Markdown('''
                                #### Top p
                            '''),
                            dcc.Slider(0, 2,
                                step=0.01,
                                id="top_p",
                                value=0.92,
                                marks={i/100: '{}'.format(i/100) for i in range(0,200,50)},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    ),
                    html.Div(
                        className="filter",
                        children = [
                            dcc.Markdown('''
                                #### Creatividad
                            '''),
                            dcc.Slider(0.0, 2.0,
                                step=0.01,
                                id="temperature",
                                value=1.0,
                                marks={i: '{}'.format(j) for i,j in zip([0,1,2],['Escasa','Media','Bastante'])},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    ),
                    html.Div(
                        className="filter",
                        children = [
                            dcc.Markdown('''
                                #### Redundancia
                            '''),
                            dcc.Slider(0.1, 1.9,
                                step=0.01,
                                id="penalty",
                                value=1.0,
                                marks={i: '{}'.format(j) for i,j in zip([0.1,1,1.9],['Escasa','Media','Bastante'])},
                                tooltip=dict(always_visible=True,placement="left")
                            )
                        ]
                    )
            
                ]
            ),

            #----------- columna de información -----------#
            html.Div(
                className="two columns",
                children=[
                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Información dinámica del nodo**

                            Información sobre el nodo al pasar el cursor del ratón por encima
                            """)),
                            html.Pre(id='hover-data', style=styles['pre'])
                        ],
                        style={'height': '400px','display':'none'}),

                    html.Div(
                        className='twelve columns',
                        children=[
                            dcc.Markdown(d("""
                            **Información del nodo seleccionado**

                            Información del nodo seleccionado del grafo.
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ],
                        style={'height': '400px','display':'none'})
                ],style={'display':'none'}
            ),
            html.Div("hola",id='hidden-div', style={'display':'none'}),
            html.Div("hola2",id='hidden-div_node', className="none", style={'display':'none'})
        ]
    )
    ,
    dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Aviso")),
                dbc.ModalBody("Dependiendo de la configuración elegida y la cantidad de datos a procesar el proceso puede tardar unos minutos. Espere."),
                dbc.ModalFooter(
                    dbc.Button(
                        "De acuerdo", id="close", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="modal",
            is_open=False,
        ),
])


#----------CALLBACK----------
@app.callback(
    [Output('grafo-conocimiento', 'figure'),
    [Output(i, 'class_name') for i in tags_network(FILEURL)[1]]],
    [Input(i, 'id') for i in tags_network(FILEURL)[1]],
    [Input(i, 'n_clicks') for i in tags_network(FILEURL)[1]]
)
def update_output(*args):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]    
    for changed in changed_id:
        
        changed = changed.split('.')[0]

        changed_clicks = args[args.index(changed) + int(len(args)/2)]

        if(changed_clicks%2==1): # No seleccionado
            if changed in SELECTED_TAGS:
                SELECTED_TAGS.remove(changed)

        else: # Seleccionado
            if changed not in SELECTED_TAGS:
                SELECTED_TAGS.append(changed)

    styles = []
    for i in range(0,int(len(args)/2)):
        theme_n_clicks = args[i+int(len(args)/2)]
        theme_name = args[i]
        if(theme_n_clicks%2==1): # No seleccionado
            if('_' in theme_name):
                styles.append('theme tag NotSelected')
            else:
                styles.append('theme stage NotSelected')

        else: # Seleccionado
            if('_' in theme_name):
                styles.append('theme tag')
            else:
                styles.append('theme stage')

      
    return network_graph(FILEURL, SELECTED_TAGS), styles

@app.callback(
    Output('hover-data', 'children'),
    Input('grafo-conocimiento', 'hoverData')
)
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)



@app.callback(
    Output('m_beams1', 'className'),
    Output('m_beams2', 'className'),
    Output('m_beams3','className'),
    Output('m_beams4','className'),
    Output('m_beams5','className'),
    Output('m_beams6','className'),
    Input('experto','value'),
)
def modo_experto(experto):
    experto=(experto[-1]==1)
    if(experto):
        return 'filter show','filter show','filter show','filter show','filter show','filter show'
    else:
        return 'filter hidden','filter hidden','filter hidden','filter hidden','filter hidden','filter hidden'



@app.callback(
    Output('outputGenerator','value'),
    Input('entradaPersonalizada','value'),
    Input('generator','n_clicks'),
    Input("hidden-div", "className"),
    Input('outputGenerator','value'),
    Input('hidden-div_node', 'className'),
    Input('do_sample','value'),      
    Input('max_len', 'value'),
    Input('top_p', 'value'),
    Input('top_k', 'value'),
    Input('temperature', 'value'),
    Input('penalty', 'value'),
    Input('num_beams', 'value'),
    Input('no_repeat_ngram_size', 'value'),
    Input('num_beam_groups','value'),
    Input('experto','value'),
)
def display_output_text(contenido,n_clicks,tags,text,node, 
do_sample, max_len,p,k,temp,penal, num_beams, no_repeat_ngram_size,num_beam_groups, experto):
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
    if 'generator.n_clicks' in changed_id:
                
        #Parametros
        do_sample=(do_sample[-1]==1)
        experto=(experto[-1]==1)


        if(not experto and do_sample):
            num_beams=1
            no_repeat_ngram_size=1
            num_beam_groups=1

        #longitud
        max_len = int(max_len/10)
        min_len=0
        if (max_len>5):
            min_len=max_len*2
            max_len=1000
            
        penal = abs(2-penal)

        if(len(SELECTED_TAGS)==0 and len(contenido)==0):
            return ''
        data = TripleList(TRIPLES.to_json())
        if (node):
            data = data.filterbyNode(node)
        triplesTree = data.triplesByTags()


        order_tags= []
        tags =tags.split(",")
        for tag in tags:
            if tag in SELECTED_TAGS:
                order_tags.append(tag)

        
        print("Generando...")

        if(experto and contenido!='' and contenido!=' '):
            triples_filtered = {'timeless': {'basic':  {'input': contenido.split("&&")}}}
        
        else:
            triples_filtered = filteredTriplesByTags(triplesTree, listTags2Tree(order_tags))

        triplesClusters = clusteringByTags(triples_filtered)
        text = printAllText(triplesClusters=triplesClusters, do_sample=do_sample,num_beams=num_beams, 
        no_repeat_ngram_size=no_repeat_ngram_size,top_k=k,top_p=p,temperature=temp,
        penalty=penal,max_length=max_len,min_length=min_len,num_beam_groups=num_beam_groups)
        
        print("Generado.")
        return text

    return 




@app.callback(
    [Output(i, 'n_clicks') for i in tags_network(FILEURL)[1]],
    Input('allTags','n_clicks'),
    Input('noneTags','n_clicks')
)
def allTasgNotSelected(n_clicks_all, n_clicks_none):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
    if 'noneTags.n_clicks' in changed_id:
        return list(np.ones(shape=(len(tags_network(FILEURL)[1]),)))
    else:
        return list(np.zeros(shape=(len(tags_network(FILEURL)[1]),)))



@app.callback(
    Output('click-data', 'children'),
    Output('hidden-div_node', 'className'),
    Input('grafo-conocimiento', 'clickData')
)
def display_click_data(clickData):
    data = json.dumps(clickData, indent=2)
    node = ""
    if(clickData):
        node = json.loads(data)['points'][0]['text']
    return json.dumps(clickData, indent=2), node



app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="make_draggable"),
    Output("drag_container", "data-drag"),
    [Input("drag_container", "id")],
)


app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="order"),
    Output("hidden-div", "className"),
    [Input(i, 'n_clicks') for i in tags_network(FILEURL)[1]]
)

@app.callback(
    Output("modal", "is_open"),
    [Input("generator", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
