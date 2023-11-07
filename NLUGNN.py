"""
This file implements the core function of ReCo

"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from dgl.nn import SAGEConv ,GINConv, GATv2Conv, EdgePredictor
from recodataset import *
import dgl.function as fn
import pickle as pkl
from tqdm import tqdm
from dataset_utils import compute_rec_metrics
from prepare_data import save_uml_bert,save_ecore
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class CosPredictor(nn.Module):
    def __init__(self,mode,in_feats):
        super().__init__()
        self.pred = EdgePredictor(mode,in_feats,out_feats=1)
    # def apply_edges(self,edges):
    #     h1 = edges.src['h']
    #     h2 = edges.dst['h']
    #     return {'score':}
    def forward(self, g, h):
        with g.local_scope():
            src, dst = g.edges_by_tuple()
            return (self.pred(h[src],h[dst])).view(-1)

class MyMLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 2)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

'''class GATv2Model(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.gat = GATv2Model(in_feats, h_feats, numheads=6)
    def'''

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class TwoLayerGraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats1,h_feats2):
        super(TwoLayerGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats1, 'mean')
        self.conv2 = SAGEConv(h_feats1, h_feats2, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GraphSAGEOne(nn.Module):
    def __init__(self, in_feats, h_feats1):
        super(GraphSAGEOne, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats1, 'mean')
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        return h

def compute_loss_single(pos_score, neg_score):
    scores = (torch.cat([pos_score, neg_score]))
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels.to('cuda'))

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])

    neg_label = torch.transpose(torch.stack((torch.ones(neg_score.shape[0]), torch.zeros(neg_score.shape[0]))), 0, 1)
    pos_label = torch.transpose(torch.stack((torch.zeros(pos_score.shape[0]), torch.ones(pos_score.shape[0]))), 0, 1)

    labels = torch.cat([pos_label, neg_label]).to('cuda')
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc_bi(pos_score_output, neg_score_output):
    pos_score = pos_score_output[:,1]
    neg_score = neg_score_output[:,1]
    scores = torch.cat([pos_score, neg_score]).to('cpu').numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_metric_bi(pos_score_output, neg_score_output):
    pos_score = torch.argmax(pos_score_output, dim=1)
    neg_score = torch.argmax(neg_score_output, dim=1)
    my_metric = {}
    scores = torch.cat([pos_score, neg_score]).to('cpu').numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to('cpu').numpy()
    acc = accuracy_score(labels, scores)
    prec = precision_score(labels, scores)
    rec = recall_score(labels, scores)

    my_metric['acc'] = acc
    my_metric['prec'] = prec
    my_metric['rec'] = rec
    return my_metric


def compute_metric(pos_score, neg_score):
    my_metric = {}
    scores = torch.cat([pos_score, neg_score]).to('cpu').numpy()
    y_pred = np.zeros(len(scores))
    pos_index = np.argpartition(scores, len(pos_score))[-1 * len(pos_score):] # the index of top N edges with high score
    y_pred[pos_index] = 1

    tp = y_pred[:len(pos_score)].sum()
    tpr = tp / len(pos_score)
    fp = y_pred[len(pos_score):].sum()
    fpr = fp / len(pos_score)
    fn = len(pos_score) - tp
    tn = len(neg_score) - fp
    my_metric['tpr'] = tpr
    my_metric['fpr'] = fpr
    my_metric['acc'] = (tp + tn) / len(scores)
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return my_metric

#TODO This func is potential buggy because it does not check the neg edge's existence, but the prob is so low
def construct_negative_graph(graph, k):
    # for every pos e ,generate k neg edges
    src, dst = graph.edges_by_tuple()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))

    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())



def construct_neg_from_batch(batched_pos_g,sample_all=False):
    ori_g_list = dgl.unbatch(batched_pos_g)
    neg_g_list = []
    for g in tqdm(ori_g_list):
        u, v = g.edges()

        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.1)

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.number_of_nodes(),g.number_of_nodes()))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u_all, neg_v_all = np.where(adj_neg != 0)
        neg_g = None
        all_zero_neg = not adj_neg.any()
        if not all_zero_neg:
            if not sample_all:
                np.random.seed(1)
                neg_eids = np.random.choice(len(neg_u_all), g.number_of_edges()*4)
                neg_u, neg_v = neg_u_all[neg_eids],neg_v_all[neg_eids]
                neg_g = dgl.graph((neg_u,neg_v), num_nodes=g.number_of_nodes()).to('cuda')
            else:
                neg_u,neg_v = neg_u_all,neg_v_all
                neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes()).to('cuda')
        else:
            neg_u,neg_v = [],[]
            neg_g = dgl.graph((neg_u,neg_v), num_nodes=g.number_of_nodes()).to('cuda')
        neg_g_list.append(neg_g)
    return (dgl.batch(neg_g_list)).to('cuda')
# during test, we need to restore context edges
def construct_neg_test_from_batch(batched_pos_g,batched_context_g,sample_all=False):

    ori_g_list = dgl.unbatch(batched_pos_g)
    ori_context_list = dgl.unbatch(batched_context_g)
    neg_g_list = []
    for i in tqdm(range(len(ori_g_list))):
        g, con_g = ori_g_list[i],ori_context_list[i]
        u, v = g.edges()
        cu,cv  = con_g.edges()
        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.1)

        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.number_of_nodes(),g.number_of_nodes()))
        adj_con = sp.coo_matrix((np.ones(len(cu)), (cu.numpy(), cv.numpy())), shape=(con_g.number_of_nodes(), con_g.number_of_nodes()))
        adj_con = adj_con.todense()

        np.fill_diagonal(adj_con,0)
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes()) - adj_con
        np.fill_diagonal(adj_neg,0)
        neg_u_all, neg_v_all = np.where(adj_neg !=0)
        neg_g = None
        all_zero_neg = not adj_neg.any()
        if not all_zero_neg:
            if not sample_all:
                neg_eids = np.random.choice(len(neg_u_all), g.number_of_edges()*2)
                neg_u, neg_v = neg_u_all[neg_eids],neg_v_all[neg_eids]
                neg_g = dgl.graph((neg_u,neg_v), num_nodes=g.number_of_nodes()).to('cuda')
            else:
                neg_u,neg_v = neg_u_all,neg_v_all
                neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes()).to('cuda')
        else:
            neg_u,neg_v = [],[]
            neg_g = dgl.graph((neg_u,neg_v), num_nodes=g.number_of_nodes()).to('cuda')
        neg_g_list.append(neg_g)
    return (dgl.batch(neg_g_list)).to('cuda')

def NoGraphModel(g,feat):
    return feat

def run_rq3(dataset,dataset_name,model_name='SAGE',exp_id=0, sage_hidden1 = 128, sage_hidden2=128,self_loop=True,test_ratio=0.7,pca=False,encoder ='bert',use_attr = True):
    run_result = {'train_acc':[],'loss':[],'test_acc':[],'test_tpr':[],'test_fpr':[],'test_auc':[]}
    # dataset = MergedRefRecDataset()

    ## transform g into a simple graph
    dataset, dataset_dict = get_dataset(dataset, dataset_name,pca,encoder,use_attr=use_attr)
    dset_train, dset_valid, dset_test = dataset_dict['train'], dataset_dict['valid'], dataset_dict['test']
    # add_reverse  = True
    # if False:
    #     dset_train[0] = dgl.add_reverse_edges(dset_train[0])
    #     dset_test[0] = dgl.add_reverse_edges(dset_test[0])
    g = dataset[0]
    print('dataset : ', dataset_name, ' model ',model_name,' self_loop', self_loop, ' num nodes', g.num_nodes(), 'num edges', g.num_edges(),' node dim ',g.ndata['feat'].shape[1] )
    print('model ', model_name)


    dset_valid_old = dset_valid
    neg_train = construct_neg_from_batch(dset_train[0])
    # TODO


    # once neg_graph was load or generated, we can build neg_eids
    print('build train test batch')
    train_g = dset_train[0].to('cuda')
    train_pos_g = train_g
    train_neg_g = neg_train

    all_test_g = dset_test[0]
    if 'parse' in dataset_name:
        test_context_g, test_neg_g, test_pos_g = construct_test_legacy(all_test_g)
    elif 'uml' in dataset_name :
        test_context_g, test_neg_g, test_pos_g = construct_test(all_test_g)

    print('building model')
    if 'parse' in dataset_name: # for quickly  *0.5
        lr = 5e-4 # -8 to uml
        epochs = 6000
    elif 'uml' in dataset_name :
        lr = 5e-15
        epochs = 6000
    model = None
    if model_name == 'SAGE':
        model = TwoLayerGraphSAGE(train_g.ndata['feat'].shape[1], sage_hidden1,sage_hidden2).to('cuda')
    elif model_name == 'GATv2':
        model = GATv2Conv(in_feats = train_g.ndata['feat'].shape[1],out_feats = sage_hidden2,num_heads=6).to('cuda')
    elif model_name == 'SAGE1':
        model =  GraphSAGEOne(in_feats=train_g.ndata['feat'].shape[1],h_feats1=sage_hidden1).to('cuda')
    elif model_name == 'NoGraph':
        #test_context_g = dgl.remove_edges(test_context_g,np.arange(test_context_g.number_of_edges()))
        model = NoGraphModel
        epochs = 3000
    # pred = MyMLPPredictor(sage_hidden2).to('cuda')
    pred = MLPPredictor(sage_hidden2).to('cuda')
    #pred = DotPredictor().to('cuda') # change dot predictor
    #pred = CosPredictor('cos',in_feats=sage_hidden2).to('cuda')

    print('learning rate', lr)
    if model_name == 'NoGraph':
        optimizer = torch.optim.Adam( pred.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=lr)

    # ----------- 4. training -------------------------------- -#
    all_logits = []
    print('training')
    train_g = train_g.to('cuda')
    run_metrics_all = []
    for e in range(epochs): # change to 2000 for uml
        # forward
        h = model(train_g, train_g.ndata['feat']).to('cuda')
        pos_score = pred(train_pos_g, h).to('cuda')
        neg_score = pred(train_neg_g, h).to('cuda')
        loss = compute_loss_single(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        #if (e + 1) % 100 == 0:
        if (e+1) % 100 == 0 and e>=epochs*0.8:  # do validation
            with torch.no_grad():
                if model_name !='NoGraph':
                    model.eval()
                pred.eval()
                my_metric = compute_metric(pos_score, neg_score)
                #acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']
                tpr, fpr, acc = my_metric['tpr'], my_metric['fpr'], my_metric['acc']
                run_result['loss'].append(loss)
                run_result['train_acc'].append(acc)

                h = model(test_context_g, test_context_g.ndata['feat']).to('cuda')
                print('In training epoch {}, loss: {} acc {} tpr {} fpr {}'.format(e, loss, acc, tpr, fpr))

                test_pos_score = pred(test_pos_g, h)
                test_neg_score = pred(test_neg_g, h)

                loss_test = compute_loss_single(test_pos_score, test_neg_score)
                my_metric = compute_metric(test_pos_score, test_neg_score)
                tpr, fpr, acc = my_metric['tpr'], my_metric['fpr'], my_metric['acc']
                print('test loss: {} acc {} tpr {} fpr {}'.format(loss_test, acc, tpr, fpr))
                run_result['test_acc'].append(acc)
                run_result['test_tpr'].append(tpr)
                run_result['test_fpr'].append(fpr)

                rec_metrics = compute_rec_metrics(test_pos_score,test_neg_score,test_context_g,test_pos_g,test_neg_g)
                regu_rec = {'model':{'frank':rec_metrics['frank'],'suc':rec_metrics['prec'],'reca':rec_metrics['rec'],'mrr':rec_metrics['mrr']},
                'line':{'frank':rec_metrics['line_frank'],'suc':rec_metrics['prec_l'],'reca':rec_metrics['rec_l'],'mrr':rec_metrics['mrr_l']}}
                run_metrics_all.append(regu_rec)
                considered_k = np.arange(50)
                k_rank = [1,2,5,10,20,50]
                #prec_m = ['%.2f'% elem for elem in(rec_metrics['prec'] * considered_k).tolist()]
                prec_m = ['%.3f' % elem for elem in (rec_metrics['prec']).tolist()]
                prec_m = [prec_m[i] for i in k_rank]
               # rec_m =  ['%.2f'% elem for elem in(rec_metrics['rec'] * considered_k).tolist()]
                rec_m = ['%.3f' % elem for elem in (rec_metrics['rec']).tolist()]
                rec_m = [rec_m[i] for i in k_rank]
                #prec_l = ['%.2f'% elem for elem in(rec_metrics['prec_l'] * considered_k).tolist()]
                prec_l = ['%.3f' % elem for elem in (rec_metrics['prec_l'] ).tolist()]
                prec_l = [prec_l[i] for i in k_rank]
                e_keys = ['frank', 'prec', 'rec', 'mrr', 'line_frank', 'prec_l', 'rec_l', 'mrr_l']
                #print('frank:{} line_frank {} rec:{} \n  prec: {}  line_prec:{} '.format(rec_metrics['frank'],rec_metrics['line_frank'],rec_m,prec_m,prec_l ))
                for k in e_keys:
                    v = rec_metrics[k]

                    if not k in ['frank','mrr','line_frank','mrr_l']:
                        eval_considered = ['%.3f' % elem for elem in v.tolist()]
                        eval_considered = [eval_considered[i] for i in k_rank]
                    else:
                        eval_considered = sum(v)/len(v)
                    print(k,':',eval_considered)
                #auc = compute_auc(test_pos_score, test_neg_score)
                #print('test AUC', auc)


                #run_result['test_prec'].append(prec)
                #run_result['test_rec'].append(rec)
                #run_result['test_auc'].append(auc)



                # print('test acc {} prec {} rec {}'.format(acc, prec, rec))

    # ----------- 5. check results ------------------------ #
    return run_metrics_all
        # furthur to make it a real test

def construct_test_legacy(all_test_g, test_ratio = 0.3):

    u, v = all_test_g.edges()
    eids = np.arange(all_test_g.number_of_edges())
    np.random.seed(0)
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.3)

    test_context_g = dgl.remove_edges(all_test_g, eids[:test_size]).to('cuda')
    test_pos_g = (dgl.remove_edges(all_test_g, eids[test_size:])).to('cuda')
    test_neg_g = construct_neg_test_from_batch(test_pos_g.to('cpu'),test_context_g.to('cpu'),sample_all=True)
    return test_context_g, test_neg_g, test_pos_g
def construct_test(all_test_g,test_ratio=0.3):
    u, v = all_test_g.edges()
    u = u.numpy();v = v.numpy()
    # group edges by src,dst; into a list of tuples
    # first select some half tuples,
    # eids = np.arange(all_test_g.number_of_edges())
    # np.random.seed(0)
    # eids = np.random.permutation(eids)
    # test_size = int(len(eids) * 0.3)
    eids = np.arange(all_test_g.number_of_edges())
    edges_by_tuple = {}
    for i, (s, d) in enumerate(zip(u, v)):
        if (d, s) in edges_by_tuple:
            edges_by_tuple[(d, s)].append(i)
        elif (s, d) in edges_by_tuple:
            edges_by_tuple[(s, d)].append(i)
        else:
            edges_by_tuple[(s, d)] = [i]
    e_v = sorted(edges_by_tuple.values())
    tuple_idx_all = np.arange(len(e_v))
    test_size = int(len(tuple_idx_all) * test_ratio)
    np.random.seed(0)
    tup_selected_indices = np.random.choice(len(e_v), test_size, replace=False)
    selected_eids = []
    for i in tup_selected_indices:
        selected_eids.extend(e_v[i])
    print('test_size', len(selected_eids) / len(u))
    remaining = eids[~np.isin(eids,selected_eids)]
    test_context_g = dgl.remove_edges(all_test_g, selected_eids).to('cuda')
    test_pos_g = (dgl.remove_edges(all_test_g, remaining)).to('cuda')
    test_neg_g = construct_neg_test_from_batch(test_pos_g.to('cpu'), test_context_g.to('cpu'), sample_all=True)
    # test_context_g = dgl.remove_edges(all_test_g, eids[:test_size]).to('cuda')
    # test_pos_g = (dgl.remove_edges(all_test_g, eids[test_size:])).to('cuda')
    # test_neg_g = construct_neg_test_from_batch(test_pos_g.to('cpu'), test_context_g.to('cpu'), sample_all=True)
    return test_context_g, test_neg_g, test_pos_g


def get_dataset(dataset, dataset_name, pca = False,encoder = 'bert',use_attr = True):
    if dataset_name in ['_py_parsed_']:
        dataset_dict = None  # TODO 'data/dataset_dict/py_parsed_ts.pkl'
        dataset_dict = pkl.load(open('data/dataset_dict/py_parsed_ts.pkl', 'rb'))
        dset_train, dset_valid, dset_test = dataset_dict['train'], dataset_dict['valid'], dataset_dict['test']
        dataset = dset_train
    elif dataset_name in ['_py_parsed_bert_']:
        dataset_dict = None  # TODO 'data/dataset_dict/py_parsed_ts.pkl'
        # dataset_dict = pkl.load(open('data/dataset_dict/py_parsed_bert.pkl', 'rb'))
        dataset_dict= save_ecore(encoder_name=encoder,pca=pca,use_attr=use_attr)
        dset_train, dset_valid, dset_test = dataset_dict['train'], dataset_dict['valid'], dataset_dict['test']
        dataset = dset_train
    elif dataset_name in ['_uml_class_','_uml_activity_','_uml_usecase_']:
        name_convert = {'_uml_class_':'class','_uml_activity_':'activity','_uml_usecase_':'usecase'}
        dataset_dict = save_uml_bert(name_convert[dataset_name],pca=pca)
        dset_train, dset_valid, dset_test = dataset_dict['train'], dataset_dict['valid'], dataset_dict['test']
        dataset = dset_train
    return dataset, dataset_dict


'''        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        print('AUC', compute_auc_bi(pos_score, neg_score))

        my_metric = compute_metric(pos_score, neg_score)
        tpr, fpr, acc = my_metric['tpr'], my_metric['fpr'], my_metric['acc']
        print('tpr: {} fpr: {} acc {}'.format(tpr, fpr, acc))'''
    # Thumbnail credits: Link Prediction with Neo4j, Mark Needham
    # sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'


load_dataset = False
save_dataset = False # save dataset to erase randomness and make experiment deterministic (train , test become constant rather than random choice)
# run_merged_all_bi()
# run_on_single_graph()
