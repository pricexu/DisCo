import pickle
import os, sys
from guacamol.distribution_learning_benchmark import KLDivBenchmark
from guacamol.frechet_benchmark import FrechetBenchmark
from guacamol.assess_distribution_learning import _assess_distribution_learning
import numpy as np
from random import sample

from typing import List

from guacamol.distribution_matching_generator import DistributionMatchingGenerator


class MockGenerator(DistributionMatchingGenerator):
    """
    Mock generator that returns pre-defined molecules,
    possibly split in several calls
    """

    def __init__(self, molecules: List[str]) -> None:
        self.molecules = molecules
        self.cursor = 0

    def generate(self, number_samples: int) -> List[str]:
        end = self.cursor + number_samples

        sampled_molecules = self.molecules[self.cursor:end]
        self.cursor = end
        return sampled_molecules

remove_h = True
smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
smiles_path = os.path.join('data/guacamol', smiles_file_name)
if os.path.exists(smiles_path):
    print("Dataset smiles were found.")
    train_smiles = np.load(smiles_path)
else:
    print("No training smiles")
    sys.exit()

dataset, backbone, n_layer, n_dim, num_graphs = 'guacamol', 'GT', 10, 256, 5000
file_name = "generated_graphs/"+dataset+'_'+backbone+'_'+str(n_layer)+'_'+str(n_dim)+'_smile_'+str(num_graphs)
with open(file_name, "rb") as fp:
    list_of_generated_smiles = pickle.load(fp)

generator = MockGenerator(list_of_generated_smiles)

# benchmark = KLDivBenchmark(number_samples=len(train_smiles), training_set=train_smiles)
benchmark = KLDivBenchmark(number_samples=10000, training_set=sample(list(train_smiles),10000))
result = benchmark.assess_model(generator)
print(result.metadata)
print(result.score)

# benchmark = FrechetBenchmark(sample_size=len(train_smiles), training_set=train_smiles)
benchmark = FrechetBenchmark(sample_size=10000, training_set=sample(list(train_smiles),10000))
result = benchmark.assess_model(generator)
print(result.metadata)
print(result.score)