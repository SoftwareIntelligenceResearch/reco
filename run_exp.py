import argparse
from NLUGNN import  run_rq3
from recodataset import *
import pickle as pkl
from noGNN import run_no_GNN
from datetime import datetime


parser = argparse.ArgumentParser(description="Your project")
parser.add_argument("-sl","--self_loop",type = bool, default = False)
parser.add_argument("-ed","--edge",type = bool, default = True)
parser.add_argument("-p","--pca",type = bool, default = False)
parser.add_argument("-pd","--pca_dim",type = int, default = 0)
parser.add_argument("-e","--encoder",type = str, default = 'sbert') # ReCo-T model
parser.add_argument("-m", "--model", type=str, default='SAGE')
parser.add_argument("-d", "--dataset_name", type=str, default='_py_parsed_')
parser.add_argument("-h1", "--hiden_1", type=int, default=128) # two hidden arguments are for the dimension of ReCo-S
parser.add_argument("-h2", "--hiden_2", type=int, default=1536)
args = parser.parse_args()



experiment_results = []
load_dataset = True
save_dataset = False
dataset = None

add_self_loop = args.self_loop
dataset_name = args.dataset_name  # '_py_parsed_','_py_parsed_bert_' _uml_class_','_uml_activity_','_uml_usecase_
model = args.model # SAGE GATv2 noGNN
encoder = args.encoder # SAGE GATv2 noGNN
''' last run
hiden_1 = 512
hiden_2 = 1024
'''

 # test sage 2
hiden_1 = args.hiden_1
hiden_2 = args.hiden_2
pca = args.pca
# noGNN
'''hiden_1 = 768*2
hiden_2 = 768*2'''
dt = datetime.now()
time_str = str(datetime.timestamp(dt))
print(time_str)
use_attr = True
use_edge = args.edge
for i in range(1): # change it to run experiment multiple times
    print('running experiment ',i)
    if dataset_name in ['_py_parsed_','_py_parsed_bert_']:
        run_result = run_rq3(dataset=dataset,dataset_name=dataset_name,model_name=model,exp_id=i,sage_hidden1=hiden_1,sage_hidden2=hiden_2,pca= args.pca_dim,encoder = encoder,use_attr=use_attr)
    elif dataset_name in ['_uml_class_','_uml_activity_','_uml_usecase_']:
        run_result = run_rq3(dataset=dataset,dataset_name=dataset_name,model_name=model,exp_id=i,sage_hidden1=hiden_1,sage_hidden2=hiden_2,pca=pca)
    elif model=='noGNN':
        run_result = run_no_GNN(dataset=dataset,dataset_name=dataset_name,model_name=model,exp_id=i,sage_hidden1=hiden_1,sage_hidden2=hiden_2)
    else:
        raise Exception("run args error, check dataset_name or model")
    experiment_results.append(run_result)

with open('output/rq-aba/' + 'use_attr_' + str(use_attr) + 'use_edge_' + str(use_edge) + str(
        add_self_loop) + '_' + model + '_' + dataset_name + '_' +'pca_'+ str(args.pca_dim) + time_str + 'results.pkl', 'wb') as outf:
    pkl.dump(run_result, outf, protocol=pkl.HIGHEST_PROTOCOL)
