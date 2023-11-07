from NLUGNN import MyMLPPredictor, compute_metric_bi, compute_auc_bi, compute_loss
import pickle as pkl
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from NLUGNN import construct_negative_graph,construct_neg_from_batch

class NoGNNMLPPredictor(nn.Module):
    def __init__(self, inp_feats, h_feats):
        super().__init__()
        self.Win = nn.Linear(inp_feats*2, 2*h_feats)
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
        return {'score': self.W2(F.relu(self.W1(F.relu(self.Win(h))))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


def s_train(dataset, dataset_name, model_name, exp_id, sage_hidden1, sage_hidden2):
    # dataset = BatchedRefRecDataset()
    print('training with mini-batch')
    run_result = {'train_acc': [], 'loss': [], 'test_acc': [], 'test_prec': [], 'test_rec': [], 'test_auc': []}
    g = dataset[0].to('cuda')
    u, v = g.edges_by_tuple()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    neg_g = construct_neg_from_batch(g.to('cpu'))
    neg_u,neg_v = neg_g.edges_by_tuple()
    neg_eids = np.arange(g.number_of_edges())
    neg_eids = np.random.permutation(eids)
    '''
    file_prefix = 'data/' + dataset_name + '/' + str(exp_id) + dataset_name
    eids = pkl.load(open(file_prefix + 'pos_eids.pkl', 'rb'))
    neg_graph = pkl.load(open(file_prefix + 'neg_graph.pkl', 'rb'))
    neg_u, neg_v = neg_graph.edges()
    neg_eids = pkl.load(open(file_prefix + 'neg_eids.pkl', 'rb'))'''

    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes()).to('cuda')
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes()).to('cuda')

    train_seeds = 0
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_seeds = torch.tensor(range(g.num_edges())).type(torch.int64)
    train_seeds = train_seeds.to('cuda')
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(3))
    dataloader = dgl.dataloading.DataLoader(
        g, train_seeds, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    pred = NoGNNMLPPredictor(sage_hidden2, 128).to('cuda')
    optimizer = torch.optim.Adam(pred.parameters(), lr=0.001)
    print('training')
    for e in range(1000):
        for input_nodes, positive_graph, negative_graph, blocks in dataloader:
            # forward
            blocks = [b.to(torch.device('cuda')) for b in blocks]
            h = blocks[0].srcdata['feat'].to('cuda')
            train_pos_g = positive_graph.to(torch.device('cuda'))
            train_neg_g = negative_graph.to(torch.device('cuda'))
            pos_score = pred(train_pos_g, h).to('cuda')
            neg_score = pred(train_neg_g, h).to('cuda')
            loss = compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        if (e+1) % 50 == 0:  # do validation
            with torch.no_grad():
                pred.eval()
                my_metric = compute_metric_bi(pos_score, neg_score)
                acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']
                run_result['loss'].append(loss)
                run_result['train_acc'].append('acc')

                print('In training epoch {}, loss: {} acc {} prec {} rec {}'.format(e, loss, acc, prec, rec))
                test_pos_score = pred(test_pos_g, h)
                test_neg_score = pred(test_neg_g, h)
                auc = compute_auc_bi(test_pos_score, test_neg_score)
                print('test AUC', auc)

                my_metric = compute_metric_bi(test_pos_score, test_neg_score)
                acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']

                run_result['test_acc'].append(acc)
                run_result['test_prec'].append(prec)
                run_result['test_rec'].append(rec)
                run_result['test_auc'].append(auc)

                print('test acc {} prec {} rec {}'.format(acc, prec, rec))
    pass


def run_no_GNN(dataset,dataset_name,model_name='noGNN',exp_id=0, sage_hidden1 = 128, sage_hidden2=768*2):
    run_result = {'train_acc': [], 'loss': [], 'test_acc': [], 'test_prec': [], 'test_rec': [], 'test_auc': []}

    g = dataset[0]
    if dataset_name == '_merged_simple_':
        m_g = dataset[0]
        g = dgl.to_simple(m_g)


    print('dataset : ', dataset_name,' num nodes', g.num_nodes(), 'num edges', g.num_edges(), )
    print('model ', model_name)
    if (dataset_name == '_batched_'):
        res = s_train(dataset=dataset,dataset_name=dataset_name,model_name=model_name,exp_id=exp_id,sage_hidden1=sage_hidden1,sage_hidden2=sage_hidden2)
        return res
    u, v = g.edges_by_tuple()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    print('building negative edges')
    neg_graph = construct_negative_graph(g, 3)  # test negative keys

    neg_u, neg_v = neg_graph.edges_by_tuple()
    neg_eids = np.arange(neg_graph.number_of_edges())
    neg_eids = np.random.permutation(neg_eids)
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())

    load_dataset =False
    if load_dataset:
        # load eids , neg_eids
        file_prefix = 'data/' + dataset_name + '/' + str(exp_id) + dataset_name
        eids = pkl.load(open(file_prefix + 'pos_eids.pkl', 'rb'))
        neg_graph = pkl.load(open(file_prefix + 'neg_graph.pkl', 'rb'))
        neg_u, neg_v = neg_graph.edges_by_tuple()
        neg_eids = pkl.load(open(file_prefix + 'neg_eids.pkl', 'rb'))

    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(g, eids[:test_size])

    # ----------- 2. create model -------------- #
    print('build train test batch')
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes()).to('cuda')
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes()).to('cuda')

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes()).to('cuda')
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes()).to('cuda')

    # hidden is 128, input_features dim is 768*2
    pred = NoGNNMLPPredictor(sage_hidden2, 128).to('cuda')
    optimizer = torch.optim.Adam(pred.parameters(), lr=0.001)
    print('training')

    for e in range(1000):
        # forward
        h = train_g.ndata['feat'].to('cuda')
        pos_score = pred(train_pos_g, h).to('cuda')
        neg_score = pred(train_neg_g, h).to('cuda')
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if e % 10 == 0:  # do validation
            with torch.no_grad():
                pred.eval()
                my_metric = compute_metric_bi(pos_score, neg_score)
                acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']
                run_result['loss'].append(loss)
                run_result['train_acc'].append('acc')

                print('In training epoch {}, loss: {} acc {} prec {} rec {}'.format(e, loss, acc, prec, rec))
                test_pos_score = pred(test_pos_g, h)
                test_neg_score = pred(test_neg_g, h)
                auc = compute_auc_bi(test_pos_score, test_neg_score)
                print('test AUC', auc)

                my_metric = compute_metric_bi(test_pos_score, test_neg_score)
                acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']

                run_result['test_acc'].append(acc)
                run_result['test_prec'].append(prec)
                run_result['test_rec'].append(rec)
                run_result['test_auc'].append(auc)

                print('test acc {} prec {} rec {}'.format(acc, prec, rec))

    # ----------- 5. check results ------------------------ #
    run_result

'''    with torch.no_grad():
        test_pos_score = pred(test_pos_g, h)
        test_neg_score = pred(test_neg_g, h)
        auc = compute_auc_bi(test_pos_score, test_neg_score)
        print('test AUC', auc)

        my_metric = compute_metric_bi(test_pos_score, test_neg_score)
        acc, prec, rec = my_metric['acc'], my_metric['prec'], my_metric['rec']

        run_result['test_acc'].append(acc)
        run_result['test_prec'].append(prec)
        run_result['test_rec'].append(rec)
        run_result['test_auc'].append(auc)
    return run_result'''