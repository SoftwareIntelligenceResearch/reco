
from recodataset import *


'''
encode labels into vectors
'''
def encoding_node_df(df_node):
    sentence_name, sentences = df_node['tok_name'].to_numpy(), df_node['tok_attr'].to_numpy()
    model = SentenceTransformer('all-mpnet-base-v2')
    print('encoding names')

    name_emb = model.encode(sentence_name, batch_size=512, show_progress_bar=True)
    print('encoding attrs')

    attr_embeddings = model.encode(sentences, batch_size=368, show_progress_bar=True)

    name_list = list(name_emb)
    attr_list = list(attr_embeddings)
    df_node['attr_embedding'] = attr_list
    df_node['name_embedding'] = name_list
    return df_node

#
def save_py_parsed_bert():
    print('loading dataset')
    dataset_all = PyEcoreBaseDataset(encoder_name='bert')
    n_models = len(dataset_all)
    num_val, num_test = int(0.1 * n_models), int(0.1 * n_models)
    np.random.seed(0)
    eval_test_indices = np.random.choice(np.arange(n_models),num_val+num_test,replace=False)
    val_indices,test_indices = eval_test_indices[0:num_val],eval_test_indices[num_val:]
    train_indices = np.delete(np.arange(n_models),eval_test_indices)
    batched_train = BatchedRefRecDataset(baseDataset=dataset_all,used_models=train_indices)
    batched_valid = BatchedRefRecDataset(dataset_all,val_indices)
    batched_test = BatchedRefRecDataset(dataset_all, test_indices)
    dataset_dict = {'train':batched_train, 'valid':batched_valid, 'test':batched_test}
    if True:
        print('dumping dataset_dict')
        pkl.dump(dataset_dict,open('data/dataset_dict/py_parsed_bert.pkl','wb'),protocol=pkl.HIGHEST_PROTOCOL)

def save_uml_bert(dataset_name,pca=False):
    print('loading dataset')
    dataset_all = UMLBaseDatset(encoder_name='sbert',pre_compute_emb=False,uml_type=dataset_name,pca=pca)
    n_models = len(dataset_all)
    num_val, num_test = int(0.1 * n_models), int(0.1 * n_models)
    np.random.seed(0)
    eval_test_indices = np.random.choice(np.arange(n_models), num_val + num_test, replace=False)
    val_indices, test_indices = eval_test_indices[0:num_val], eval_test_indices[num_val:]
    train_indices = np.delete(np.arange(n_models), eval_test_indices)
    batched_train = BatchedRefRecDataset(baseDataset=dataset_all, used_models=train_indices)
    batched_valid = BatchedRefRecDataset(baseDataset=dataset_all, used_models=val_indices)
    batched_test = BatchedRefRecDataset(baseDataset=dataset_all, used_models=test_indices)
    dataset_dict = {'train': batched_train, 'valid': batched_valid, 'test': batched_test}
    if False:
        print('dumping dataset_dict')
        pkl.dump(dataset_dict, open('data/dataset_dict/py_parsed_ts.pkl', 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        print('done')
    return dataset_dict

def save_py_parsed_ts():
    print('loading dataset')
    dataset_all = PyEcoreBaseDataset(encoder_name='ts')
    n_models = len(dataset_all)
    num_val, num_test = int(0.1 * n_models), int(0.1 * n_models)
    np.random.seed(0)
    eval_test_indices = np.random.choice(np.arange(n_models),num_val+num_test,replace=False)
    val_indices,test_indices = eval_test_indices[0:num_val],eval_test_indices[num_val:]
    train_indices = np.delete(np.arange(n_models),eval_test_indices)
    batched_train = BatchedRefRecDataset(baseDataset=dataset_all,used_models= train_indices)
    batched_valid = BatchedRefRecDataset(baseDataset=dataset_all,used_models = val_indices)
    batched_test = BatchedRefRecDataset(baseDataset=dataset_all, used_models =test_indices)
    dataset_dict = {'train':batched_train, 'valid':batched_valid, 'test':batched_test}
    if False:
        print('dumping dataset_dict')
        pkl.dump(dataset_dict,open('data/dataset_dict/py_parsed_ts.pkl','wb'),protocol=pkl.HIGHEST_PROTOCOL)
        print('done')

def save_ecore(encoder_name = 'bert',pre_comp = False,pca = 128,use_attr = True):
    print('loading dataset')
    dataset_all = PyEcoreBaseDataset(encoder_name=encoder_name,pre_compute_emb=pre_comp,pca= pca,use_attr=use_attr)
    n_models = len(dataset_all)
    num_val, num_test = int(0.1 * n_models), int(0.1 * n_models)
    np.random.seed(0)
    eval_test_indices = np.random.choice(np.arange(n_models),num_val+num_test,replace=False)
    val_indices,test_indices = eval_test_indices[0:num_val],eval_test_indices[num_val:]
    train_indices = np.delete(np.arange(n_models),eval_test_indices)
    batched_train = BatchedRefRecDataset(baseDataset=dataset_all,used_models= train_indices)
    batched_valid = BatchedRefRecDataset(baseDataset=dataset_all,used_models = val_indices)
    batched_test = BatchedRefRecDataset(baseDataset=dataset_all, used_models =test_indices)
    dataset_dict = {'train':batched_train, 'valid':batched_valid, 'test':batched_test}
    return dataset_dict


