from collections import defaultdict

import dgl
from dgl.data import DGLDataset
import torch
import os
import pickle as pkl
import numpy as np
import pandas as pd
from jinja2.nodes import Test
from pandas import DataFrame
from tqdm import tqdm
from dataset_utils import *
from sentence_transformers import SentenceTransformer,models
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

os.environ['ALL_PROXY'] = "socks5://127.0.0.1:1889" # comment it if you dont need use proxy to connect internet
class morgan_graph():
    def __init__(self,g:dgl.DGLGraph,names:list,attrs:list,edge_features:DataFrame): #attr: list of list, every inner list [[str,str],[...]]
        self.g = g
        self.context_g = None
        id2name = []
        nodes_len = g.num_nodes()
        attr_tuple_lists= [] #[nodeid1,attr1],[nodeid2,attr2], ...
        for i in range(nodes_len):
            id2name.append(names[i])
            for attr_s in attrs[i]:
                attr_tuple_lists.append((i,attr_s))
        u,v = g.edges()
        df_edges_feats = edge_features
        self.id2name = id2name
        self.attr_tuple_lists = attr_tuple_lists
        self.edge_features = edge_features
    def save_context(self,cg):
        self.context_g = cg
    def save_context_edge_feats(self,ef):
        self.context_edge_feats = ef

# dataset = MergedRefRecDataset()
# dataset

def explain_pca(pca):
    exp_var_pca = pca.explained_variance_ratio_
    #
    # Cumulative sum of eigenvalues; This will be used to create step plot
    # for visualizing the variance explained by each principal component.
    #
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    #
    # Create the visualization plot
    #
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    pass

class PyEcoreBaseDataset(DGLDataset):
    def __init__(self,encoder_name,pre_compute_emb=True,pca = 0,use_attr = True):
        self.encoder_name = encoder_name
        self.pre_compute_emb = pre_compute_emb
        self.pca=  pca
        self.use_attr = use_attr
        super().__init__(name='PyEcoreBase')


    def process(self):
        fname_node_df = 'data/ecore-parsed-df/nodes_ecore-parsed_all.pkl'
        fname_graph_df = 'data/ecore-parsed-df/graphs_ecore-parsed_all.pkl'
        fname_edge_df = 'data/ecore-parsed-df/edges_ecore-parsed_all.pkl'
        nodes = pkl.load(open(fname_node_df, 'rb'))
        properties = pkl.load(open(fname_graph_df, 'rb'))
        edges = pkl.load(open(fname_edge_df, 'rb'))
        encoder = None
        embeddings = None; batch_size = 512 # if mem empty 768 else 488
        if self.encoder_name == 'ts':
            if self.pre_compute_emb:
                print('loading precomputed embeddings')
                nodes = pkl.load(open('data/ecore-parsed-df/nodes_emb_ecore-parsed_all.pkl','rb'))
            else:
                encoder = SentenceTransformer('output/tsdae-ecore-parsed')
                nodes['attr_str'] = [' '.join(map(str, l)) for l in nodes['attrs']]
                nodes_sent = nodes['node_name'].astype(str) + ' ' + nodes['attr_str'].astype(str)
                print('encoding model elements into vectors')
                embeddings = encoder.encode(nodes_sent, batch_size=288, show_progress_bar=True,device='cuda:0')
                nodes['embeddings'] = list(embeddings)
        elif self.encoder_name == 'bert':
            if self.pre_compute_emb:
                nodes = pkl.load(open('data/ecore-parsed-df/nodes_emb_bert_ecore-parsed_all.pkl','rb'))
            else:
                model_name = 'bert-base-uncased'
                word_embedding_model = models.Transformer(model_name)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
                encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

                nodes['attr_str'] = [' '.join(map(str, l)) for l in nodes['attrs']]
                nodes_sent = nodes['node_name'].astype(str) + ' ' + nodes['attr_str'].astype(str)
                print('encoding model elements into vectors')
                embeddings = encoder.encode(nodes_sent, batch_size=batch_size, show_progress_bar=True,device='cuda:0')
                nodes['embeddings'] = list(embeddings)
                pkl.dump(nodes,open('data/ecore-parsed-df/nodes_emb_bert_ecore-parsed_all.pkl','wb'),protocol=pkl.HIGHEST_PROTOCOL)

        elif self.encoder_name == 'sbert':
            if self.pre_compute_emb:
                nodes = pkl.load(open('data/ecore-parsed-df/nodes_emb_sbert_ecore-parsed_all.pkl','rb'))
            else:
                #print(torch.cuda.is_available())
                encoder = SentenceTransformer('all-mpnet-base-v2')
                encoder.to('cuda')
                use_attr = self.use_attr
                print('use_attr: ',use_attr)
                print('attr:',use_attr)
                if use_attr:
                    nodes['attr_str'] = [' '.join(map(str, l)) for l in nodes['attrs']]
                else:
                    nodes['attr_str'] = ''
                nodes_sent = 'name:'+nodes['node_name'].astype(str) + ' '+'attributes:' + nodes['attr_str'].astype(str)
                print('encoding model elements into vectors')
                embeddings = encoder.encode(nodes_sent, batch_size=batch_size, show_progress_bar=True)
                nodes['embeddings'] = list(embeddings)
                #pkl.dump(nodes,open('data/ecore-parsed-df/nodes_emb_bert_ecore-parsed_all.pkl','wb'),protocol=pkl.HIGHEST_PROTOCOL)
        if self.pre_compute_emb is False and self.pca>0:
            print('doing pca')
            new_dimension = self.pca
            pca_train_sentences = nodes_sent[0:200000]
            train_embeddings = embeddings
            pca = PCA(n_components=new_dimension)
            pca.fit(train_embeddings)
            pca_comp = np.asarray(pca.components_)
            explain_pca(pca)
            # We add a dense layer to the model, so that it will produce directly embeddings with the new size
            dense = models.Dense(in_features=encoder.get_sentence_embedding_dimension(),
                                 out_features=new_dimension, bias=False,
                                 activation_function=torch.nn.Identity())
            dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
            encoder.add_module('dense', dense)
            embeddings = encoder.encode(nodes_sent, batch_size=int(batch_size*0.8), show_progress_bar=True)
            nodes['embeddings'] = list(embeddings)
            del encoder
            # pkl.dump(nodes, open('data/' + 'ecore dim ' + str(new_dimension) + '_nodes_emb_'+self.encoder_name+'.pkl', 'wb'),
            #          protocol=pkl.HIGHEST_PROTOCOL)
        self.valid_graphs,self.valid_graph_id = [],[]
        self.labels = []
        self.num_nodes_dict={}
        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():

            num_nodes_dict[row['graph_id']] = row['num_nodes']

        len2g = defaultdict(list)
        points = [0, 5, 10, 50, 100, max(num_nodes_dict.values())+1]
        keys = ['0', '0-5', '5-10', '10-50', '50-100', '100-max']
        for i in range(len(num_nodes_dict.keys())):
            num_n = num_nodes_dict[i]
            for j in range(1, len(points)):
                if num_n >= points[j - 1] and num_n < points[j]:
                    len2g[keys[j]].append(i)
                    break
        self.length2graph = len2g
        self.num_nodes_dict = num_nodes_dict

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby('graph_id') # nodes_group >edges_group means there are graphs that do not have any edge!
        # For each graph ID...
        egg = edges_group.groups
        all_g = []
        non_empty_graphs = list(set(list(edges['graph_id']))) # there are graphs without any edge
        for graph_id in edges_group.groups:
        # Find the edges as well as the number of nodes and its label.
         # only consider graph with at leaset 5 nodes
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            # label = label_dict[graph_id]
            nodes_of_id = nodes_group.get_group(graph_id)

            hybrid_features = np.stack(nodes_of_id['embeddings'].to_numpy())
            #node_name_features = np.stack(nodes_of_id['name_embedding'].to_numpy())
            #node_features = np.stack(nodes_of_id['attr_embedding'].to_numpy())
            # Create a graph and add it to the list of graphs and labels.
            #node_feats_all = np.hstack((node_name_features,node_features))
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['feat'] = torch.from_numpy(hybrid_features)
            all_g.append(g)
            if (num_nodes_dict[graph_id] >2 and graph_id in non_empty_graphs):
                self.valid_graphs.append(g)
                self.valid_graph_id.append(graph_id)
            # self.labels.append(label)

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes_dict = num_nodes_dict
        self.all_g = all_g

    def __getitem__(self, i):
        return self.valid_graphs[i]

    def __len__(self):
        return len(self.valid_graphs)

class MorganNodupBaseDataset(DGLDataset):

    def __init__(self):
        self.no_edges_graph = []
        super().__init__(name='refrec')

    def process(self):
        fname_node_df = 'data/emb_nodes_nodup_all.pkl'
        fname_graph_df = 'data/graphs_nodup_all.pkl'
        fname_edge_df = 'data/edges_nodup_all.pkl'
        nodes = pkl.load(open(fname_node_df, 'rb'))
        properties = pkl.load(open(fname_graph_df, 'rb'))
        edges = pkl.load(open(fname_edge_df, 'rb'))

        # encoding_node_df(nodes)
        #edges = pkl.load(open('data/edges_all.pkl','rb'))
        #properties1 = pkl.load(open('data/graphs_all.pkl','rb'))
        #nodes = pkl.load(open('data/emb_tok_nodes_all.pkl','rb'))
        all_morgan_g = []
        self.graphs = []
        self.labels = []
        self.num_nodes_dict={}
        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():

            num_nodes_dict[row['graph_id']] = row['num_nodes']

        len2g = defaultdict(list)
        points = [0, 5, 10, 50, 100, max(num_nodes_dict.values())+1]
        keys = ['0', '0-5', '5-10', '10-50', '50-100', '100-max']
        for i in range(len(num_nodes_dict.keys())):
            num_n = num_nodes_dict[i]
            for j in range(1, len(points)):
                if num_n >= points[j - 1] and num_n < points[j]:
                    len2g[keys[j]].append(i)
                    break
        self.length2graph = len2g
        self.num_nodes_dict = num_nodes_dict

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby('graph_id') # nodes_group >edges_group means there are graphs that do not have any edge!
        # For each graph ID...
        egg = edges_group.groups
        non_emp_graphs = np.array(list(egg.keys()))
        e_arr = np.zeros(properties.shape[0])
        e_arr[non_emp_graphs] = 1
        bug_g = np.argwhere(e_arr==0)
        all_g = []
        no_edges_graph_ids = []
        for graph_id in edges_group.groups:
        # Find the edges as well as the number of nodes and its label.
         # only consider graph with at leaset 5 nodes
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            if len(src) ==0:
                no_edges_graph_ids.append(graph_id)
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            # label = label_dict[graph_id]
            nodes_of_id = nodes_group.get_group(graph_id)

            node_name_features = np.stack(nodes_of_id['name_embedding'].to_numpy())
            node_features = np.stack(nodes_of_id['attr_embedding'].to_numpy())
            # Create a graph and add it to the list of graphs and labels.
            node_feats_all = np.hstack((node_name_features,node_features))
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['feat'] = torch.from_numpy(node_feats_all)
            mg_g = morgan_graph(g,names = nodes_of_id['node_name'].to_numpy().tolist(),attrs=nodes_of_id['attrs'].to_numpy().tolist(),edge_features=edges_of_id)
            all_g.append(g)
            all_morgan_g.append(mg_g)
            if (num_nodes_dict[graph_id] > 4):
                self.graphs.append(g)
            # self.labels.append(label)

        # Convert the label list to tensor for saving.

        self.all_morgan_g = all_morgan_g
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes_dict = num_nodes_dict
        self.all_g = all_g
        self.no_edges_graph = no_edges_graph_ids
    def __getitem__(self, i):
        return self.all_morgan_g[i]

    def __len__(self):
        return len(self.all_g)

#tt = EmbNodupBaseDataset()
class UMLBaseDatset(DGLDataset):
    def __init__(self, encoder_name='bert', pre_compute_emb=True,uml_type = 'class',pca=False):
        self.encoder_name = encoder_name
        self.pre_compute_emb = pre_compute_emb
        self.uml_type = uml_type
        self.pca = pca
        super().__init__(name='UMLBase')

    def combine_tuples(row):
        A = row['src_lower_upper'] if row['src_lower_upper'] is not None else []
        B = row['dst_lower_upper'] if row['dst_lower_upper'] is not None else []
        return tuple(zip(A, B))
    def process(self):

        uml_dict_dir = 'data/uml/uml-dict.pkl'
        uml_dict = pkl.load(open(uml_dict_dir, 'rb'))
        nodes = uml_dict[(self.uml_type,'node')]
        #properties = pkl.load(open(fname_graph_df, 'rb'))
        edges_uni = uml_dict[(self.uml_type,'edge')]

        if self.uml_type in ['class', 'usecase']:
            print('add reverse edges')
            #edges_uni['src_lower_upper','dst_lower_upper'] = edges_uni.apply(self.combine_tuples, axis=1)
            name_a,name_b = 'src_lower_upper','dst_lower_upper'
            edges_uni[[name_a,name_b]] = edges_uni.apply(lambda row: (tuple(row[name_a]) if row[name_a] else None,
                                                           tuple(row[name_b]) if row[name_b] else None), axis=1,
                                              result_type='expand')            # drop original A and B columns if desired
            edges_uni = edges_uni.drop([name_a,name_b],axis=1)
            #df = df.drop(['A', 'B'], axis=1)
            edges = add_reverse_edge(edges_uni)
        else:
                edges = edges_uni
        encoder = None
        if self.encoder_name == 'ts':
            if self.pre_compute_emb:
                print('loading precomputed embeddings')
                nodes = pkl.load(open('data/ecore-parsed-df/nodes_emb_ecore-parsed_all.pkl', 'rb'))
            else:
                encoder = SentenceTransformer('output/tsdae-ecore-parsed')
                nodes['attr_str'] = [' '.join(map(str, l)) for l in nodes['attrs']]
                nodes_sent = nodes['node_name'].astype(str) + ' ' + nodes['attr_str'].astype(str)
                print('encoding model elements into vectors')
                embeddings = encoder.encode(nodes_sent, batch_size=512, show_progress_bar=True)
                nodes['embeddings'] = list(embeddings)
        elif self.encoder_name in ['bert','sbert']:
            if False:
                if not self.pca:
                    nodes = pkl.load(open('data/uml/' + self.uml_type + '_nodes_emb_bert.pkl', 'rb'))
                else:
                    nodes = pkl.load(open('data/uml/'+self.uml_type+' dim '+'128'+'_nodes_emb_bert.pkl', 'rb'))
            else:
                if self.encoder_name == 'bert':
                    model_name = 'bert-base-uncased'
                    word_embedding_model = models.Transformer(model_name)
                    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
                    encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                else:
                    encoder = SentenceTransformer('all-mpnet-base-v2')
                    encoder.to('cuda')
                #maybe we can serialize node's neighbor as sentence nodes['neighbors'] =
                if self.uml_type == 'class':
                    #nodes['attr_str'] = [' '.join(map(str, l[0])) for l in nodes['attribute']]
                    nodes['attr_str'] = nodes['attribute'].apply(list_tup2str)
                    nodes_sent = 'name: '+nodes['node_name'].astype(str) + ' attribute: ' + nodes['attr_str'].astype(str)
                elif self.uml_type in ['activity','usecase']:
                    nodes_sent = 'name: '+nodes['node_name'].astype(str)
                print('encoding model elements into vectors')
                pca_red = True
                if pca_red:
                    print('doing pca')
                    new_dimension  = 128
                    pca_train_sentences = nodes_sent[0:200000]
                    if self.uml_type in ['activity','usecase']: batch_size = 1024
                    else: batch_size = 512
                    train_embeddings = encoder.encode(pca_train_sentences,batch_size=batch_size, convert_to_numpy=True)
                    pca = PCA(n_components=new_dimension)
                    pca.fit(train_embeddings)
                    pca_comp = np.asarray(pca.components_)
                    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
                    dense = models.Dense(in_features=encoder.get_sentence_embedding_dimension(),
                                         out_features=new_dimension, bias=False,
                                         activation_function=torch.nn.Identity())
                    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
                    encoder.add_module('dense', dense)
                embeddings = encoder.encode(nodes_sent, batch_size=batch_size, show_progress_bar=True,device='cuda:0')
                nodes['embeddings'] = list(embeddings)
                # pkl.dump(nodes, open('data/uml/'+self.uml_type+self.encoder_name+' dim '+str(new_dimension)+'_nodes_emb_bert.pkl', 'wb'),
                #          protocol=pkl.HIGHEST_PROTOCOL)
        # encoding_node_df(nodes)
        # edges = pkl.load(open('data/edges_all.pkl','rb'))
        # properties1 = pkl.load(open('data/graphs_all.pkl','rb'))
        # nodes = pkl.load(open('data/emb_tok_nodes_all.pkl','rb'))

        self.valid_graphs, self.valid_graph_id = [], []
        self.labels = []
        self.num_nodes_dict = {}
        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}

        num_nodes_dict = {}
        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        nodes_group = nodes.groupby(
            'graph_id')  # nodes_group >edges_group means there are graphs that do not have any edge!
        # For each graph ID...
        egg = edges_group.groups
        all_g = []
        non_empty_graphs = list(set(list(edges['graph_id'])))  # there are graphs without any edge
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            # only consider graph with at leaset 5 nodes
            edges_of_id = edges_group.get_group(graph_id)

                # add reverse edge for associations
            src = edges_of_id['src_id'].to_numpy()
            dst = edges_of_id['dst_id'].to_numpy()
            nodes_of_id = nodes_group.get_group(graph_id)
            num_nodes = nodes_of_id.shape[0]
            num_nodes_dict [graph_id] = num_nodes
            # label = label_dict[graph_id]


            hybrid_features = np.stack(nodes_of_id['embeddings'].to_numpy())
            # node_name_features = np.stack(nodes_of_id['name_embedding'].to_numpy())
            # node_features = np.stack(nodes_of_id['attr_embedding'].to_numpy())
            # Create a graph and add it to the list of graphs and labels.
            # node_feats_all = np.hstack((node_name_features,node_features))
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['feat'] = torch.from_numpy(hybrid_features)
            all_g.append(g)
            if True:
                self.valid_graphs.append(g)
                self.valid_graph_id.append(graph_id)
            # self.labels.append(label)

        # Convert the label list to tensor for saving.
        #self.labels = torch.LongTensor(self.labels)
        self.num_nodes_dict = num_nodes_dict
        self.all_g = all_g

    def __getitem__(self, i):
        return self.valid_graphs[i]

    def __len__(self):
        return len(self.valid_graphs)
