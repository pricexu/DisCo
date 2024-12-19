import pickle
import moses
from random import sample

# dataset = 'moses'
# backbone = 'GT'
# epochs = list(range(155,185,5))
# list_of_generated_smiles = []
# for epoch in epochs:
#     print(epoch)
#     file_name = "generated_graphs/"+dataset+'_'+backbone+'_smile_'+str(epoch)
#     with open(file_name, "rb") as fp:
#         list_of_generated_smiles += pickle.load(fp)


# dataset, backbone, n_layer, n_dim, num_graphs = 'moses', 'GT', 12, 256, 5000
dataset, backbone, n_layer, n_dim, num_graphs = 'moses', 'MPNN', 10, 512, 5000
file_name = "generated_graphs/"+dataset+'_'+backbone+'_'+str(n_layer)+'_'+str(n_dim)+'_smile_'+str(num_graphs)
print(file_name)
with open(file_name, "rb") as fp:
    list_of_generated_smiles = pickle.load(fp)

# metrics = moses.get_all_metrics(sample(list_of_generated_smiles,2000))
metrics = moses.get_all_metrics(list_of_generated_smiles)
print(metrics)
print("Filters: {}".format(metrics['Filters']*100))
print("FCD/Test: {}".format(metrics['FCD/Test']))
print("SNN/Test: {}".format(metrics['SNN/Test']))
print("Scaf/TestSF: {}".format(metrics['Scaf/TestSF']*100))