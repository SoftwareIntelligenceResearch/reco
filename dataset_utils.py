import numpy as np
from sentence_transformers import SentenceTransformer
import dgl
import pandas as pd

def combine_lists(lst1, lst2):
    return ' '.join([f"({name},{typ},attr)" for name, typ in zip(lst1, lst2)])

# Apply the function to the `var_names` and `var_types` columns to create a new `vars` column


class LabelGraph4Ecore():
    def setup_node2str(self,node_df,edge_df):
        str_list = []
        nodes_list = self.graph.nodes
        merged_edges = pd.merge(edge_df, node_df.rename(columns={'node_id': 'dst', 'node_name': 'dst_name'}), on='dst')
        merged_edges['edge_str'] = '(' + merged_edges['feats'] + ',' + merged_edges['dst_name'] + ',' + merged_edges['edge_type']+')'
        node_agg = merged_edges.groupby('src')['edge_str'].agg(lambda x: ' '.join(x))
        node_agg = pd.merge(node_df, node_agg, left_on='node_id', right_on='src', how='left')

        # Replace NaN values in `edge_str` with empty strings
        node_agg['edge_str'] = node_agg['edge_str'].fillna('')
        node_agg['vars'] = node_agg.apply(lambda row: combine_lists(row['attrs'], row['attr_types']), axis=1)
        node_agg['var_sep'] = node_agg['vars'].apply(lambda x: '' if len(x) == 0 else ' ')
        node_agg['edge_sep'] = node_agg['edge_str'].apply(lambda x: '' if len(x) == 0 else ' ')
        node_agg['node_str'] = 'modelset\t'+node_agg['node_name']+node_agg['var_sep']+node_agg['vars']+node_agg['edge_sep']+node_agg['edge_str']+' \n'
        return node_agg
    def __init__(self,dgl_graph,node_df,edge_df,gid):
        super(LabelGraph4Ecore, self).__init__()
        self.graph = dgl_graph
        self.gid = gid
        self.node2str = self.setup_node2str(node_df,edge_df)
    def graph2str_list(self):
        str_df = list(self.node2str['node_str'])
        return str_df
    def node_id2str(self,nid):
        return self.node2str[self.node2str['node_id']==nid]['node_str'].values
def encoding_node_df(df_node):
    sentence_name, sentences = df_node['tok_name'].to_numpy(), df_node['tok_attr'].to_numpy()
    model = SentenceTransformer('all-mpnet-base-v2')
    print('encoding names')

    name_emb = model.encode(sentence_name, batch_size=2048, show_progress_bar=True)
    print('encoding attrs')

    attr_embeddings = model.encode(sentences, batch_size=368, show_progress_bar=True)

    name_list = list(name_emb)
    attr_list = list(attr_embeddings)
    df_node['attr_embedding'] = attr_list
    df_node['name_embedding'] = name_list
    return df_node

def compute_rec_metrics(pos_score,neg_score,context_g,pos_g,neg_g):
    # for link level:
    #
    # mean position
    # suc at k
    pos_g_list = dgl.unbatch(pos_g.to('cpu'))
    neg_g_list = dgl.unbatch(neg_g)
    con_g_list = dgl.unbatch(context_g)
    # change if mlp changes
    pos_probs = pos_score.detach().cpu().numpy()
    neg_probs = neg_score.detach().cpu().numpy()

    (p_u, p_v),(n_u,n_v) = pos_g.edges(), neg_g.edges()
    p_u,p_v,n_u,n_v = p_u.cpu().numpy(),p_v.cpu().numpy(),n_u.cpu().numpy(),n_v.cpu().numpy()

    c_u,c_v = context_g.edges()
    c_u,c_v = c_u.cpu().numpy(),c_v.cpu().numpy()
    df_context = pd.DataFrame({'u':c_u, 'v':c_v})
    # p_res/ n_res : every column is (prob,label,u,v)
    df_pos_result = pd.DataFrame({'prob':pos_probs,'u':p_u,'v':p_v})
    df_pos_result['label'] = 1
    df_neg_result = pd.DataFrame({'prob': neg_probs, 'u': n_u, 'v': n_v})
    df_neg_result['label'] = 0

    start_pos,start_neg = 0,0
    # test every 2 graphs  are compleccent
    metric_dict = {}
    line_metrics = []# avg_frank,avg_prec,avg_rec,n_link
    n_test_models,m_frank_sum,m_mass_prec, m_mass_rec = 0,0,np.zeros(51),np.zeros(51)
    l_all_frank , l_all_prec, l_all_rec = 0,np.zeros(51), np.zeros(51)
    l_all_mrr,m_mrr = 0,0
    l_med,m_med = 0,0
    model_frank_list,model_mrr_list ,model_prec_list,model_rec_list= [],[],[],[]
    l_frank_list, l_mrr_list, l_prec_list, l_rec_list = [], [], [], []
    model_suc_r_list,line_suc_r_list = [],[]
    for i in range(len(pos_g_list)):
        gp,gn,gc = pos_g_list[i], neg_g_list[i] ,con_g_list[i]
        n_p_edge, n_n_edge = gp.num_edges(), gn.num_edges()
        result_p, result_n = df_pos_result.iloc[start_pos:start_pos+n_p_edge],df_neg_result[start_neg:start_neg+n_n_edge]
        result_all = pd.concat([result_p, result_n],axis=0)
        _cu,_cv = gc.edges()
        start_pos += n_p_edge
        start_neg += n_n_edge
        if n_p_edge ==0 or n_n_edge == 0:
            continue
        n_test_models +=1
        # else compute metrics
        sorted_res = (result_all.sort_values(by = 'prob',ascending=False)).reset_index()

        # compute line level metrics
        n_link,l_frank_sum, l_mass_prec, l_mass_rec = 0,0,np.zeros(51),np.zeros(51)
        src_groups = sorted_res.groupby('u')
        link_metric = False  # line metrics should be moved up?
        for group_id,rec_links in src_groups:
            # test only if both neg and pos existsw
            if 0 in rec_links['label'].values and 1 in rec_links['label'].values:
                n_link+=1
                link_metric =True
                # compute metric
                num_p_link = rec_links[rec_links['label']==1].shape[0]
                num_n_link = rec_links[rec_links['label']==0].shape[0]
                l_frank, l_prec, l_rec = frank_prec_rec(num_n_link, num_p_link, rec_links.reset_index(drop=True))
                l_frank_sum += l_frank
                l_mass_prec += l_prec
                l_mass_rec += l_rec
                l_frank_list.append(l_frank+1)
                l_prec_list.append(l_prec)
                l_rec_list.append(l_rec)
                l_mrr_list.append((1.0 / (l_frank+1)))
        if not link_metric:
            continue

        # compute model level metrics: treating all equal; only if line level metrics exist, we compute model level metrics
        frank, m_prec, m_rec = frank_prec_rec(n_n_edge, n_p_edge, sorted_res)
        m_med += (sorted_res.shape[0]) / 2
        frank += 1
        m_frank_sum += frank
        m_mass_prec += m_prec
        m_mass_rec += m_rec
        m_mrr = (1.0 / frank)
        # m_suc_rate =
        model_frank_list.append(frank)
        model_prec_list.append(m_prec)
        model_rec_list.append(m_rec)
        model_mrr_list.append(m_mrr)

        # if link_metric:
        #     line_metrics.append([(l_frank_sum/n_link)+1,l_mass_prec/n_link,l_mass_rec/n_link,n_link])
        #     l_all_prec += l_mass_prec/n_link
        #     l_all_rec += l_mass_rec/n_link
        #     l_all_frank += (l_frank_sum/n_link)+1
        #     l_all_mrr += (1.0 / (frank+1))
        #
        # else:
        #     line_metrics.append([])
    l_all_ = np.stack(l_prec_list, axis=0)
    n_link_exp = len(line_metrics)
    avg_frank = (m_frank_sum/n_test_models)
    avg_prec = m_mass_prec/n_test_models
    avg_rec = m_mass_rec/n_test_models
    e_keys = ['frank','prec','rec','mrr','line_frank','prec_l','rec_l','mrr_l']
    metric_dict['frank'] =model_frank_list
    metric_dict['prec'] = np.average(np.stack(model_prec_list, axis=0),axis=0)
    metric_dict['rec'] = np.average(np.stack(model_rec_list, axis=0),axis=0)
    metric_dict['mrr'] = model_mrr_list

    model_frank_list.append(frank + 1)
    model_prec_list.append(m_prec)
    model_rec_list.append(m_rec)
    model_mrr_list.append((1.0 / (frank + 1)))



    # line metrics
    metric_dict['line_frank'] = l_frank_list
    metric_dict['prec_l'] = np.average(np.stack(l_prec_list, axis=0),axis=0)
    metric_dict['rec_l'] = np.average(np.stack(l_rec_list, axis=0),axis=0)
    metric_dict['mrr_l'] = l_mrr_list

    return metric_dict


def frank_prec_rec(n_n_edge, n_p_edge, sorted_res):
    # frank: first hit
    frank_sum,mass_prec,mass_rec = 0,np.zeros(51),np.zeros(51)
    hits = sorted_res.index[sorted_res['label'] == 1].to_frame(index=False)
    frank = hits[0][0]
    considered_k = [1, 2, 5, 10,20,50]
    considered_k = list(range(1,51))
    for k in considered_k:
        prec_k = (sorted_res.iloc[:k]['label'].sum())
        if k >= sorted_res.shape[0]:
            rec_k = 1
        else:
            rec_k = (sorted_res.iloc[:k]['label'].sum()) / n_p_edge
        mass_prec[k] += prec_k
        mass_rec[k] += rec_k
    # acculumate results:
    frank_sum += frank
    return frank_sum,mass_prec,mass_rec

def list_tup2str(li):
    start = ''
    for tup in li:
        start = start +' ' + tup[0]
    return start

def add_reverse_edge(edge_df):
    pd_reverse = edge_df.rename(columns={'src_id': 'dst_id', 'dst_id': 'src_id'})
    pd_reverse = pd_reverse[pd_reverse['src_id'] != pd_reverse['dst_id']]
    new_df = pd.concat([edge_df, pd_reverse])
    new_df = new_df.drop_duplicates()
    #new_df = new_df.drop_duplicates(subset=['edge_type', 'src_id', 'dst_id'])
    return new_df

def combine_tuples(row):
    A = row['A'] if row['A'] is not None else []
    B = row['B'] if row['B'] is not None else []
    return tuple(zip(A, B))