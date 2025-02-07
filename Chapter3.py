from pprint import pprint

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image
from pgmpy.models import BayesianNetwork


# Listing 3.1 DAG rock-throwing example


def true_dgp(jenny_inclination, brian_inclination, window_strength):
    jenny_throws_rock = jenny_inclination > 0.5
    brian_throws_rock = brian_inclination > 0.5
    if jenny_throws_rock and brian_throws_rock:
        strength_of_impact = 0.8
    elif jenny_throws_rock or brian_throws_rock:
        strength_of_impact = 0.6
    else:
        strength_of_impact = 0.0
    window_breaks = window_strength < strength_of_impact
    return jenny_throws_rock, brian_throws_rock, window_breaks


# Listing 3.2 Building the transportation DAG in pgmpy
model = BayesianNetwork(
    [
        ("Age", "Education"),
        ("Sex", "Education"),
        ("Education", "Occupation"),
        ("Education", "Residence"),
        ("Occupation", "Transportation"),
        ("Residence", "Transportation"),
    ]
)


viz = model.to_daft()
plt.figure(figsize=(10, 10))
viz.show()

# Listing 3.3 Loading transportation data
data = pd.read_csv("data/transportation_survey.csv")
data.head()
data.dtypes
data = data.rename(
    columns={
        "A": "Age",
        "S": "Sex",
        "E": "Education",
        "O": "Occupation",
        "R": "Residence",
        "T": "Transportation",
    }
)

# Listing 3.4 Learning parameters for the causal Markov kernels

model.fit(data)
causal_markov_kernels = model.get_cpds()
pprint(causal_markov_kernels)

cmk_tranportation = causal_markov_kernels[-1]
print(cmk_tranportation)
data.Occupation.unique()

cmk_tranportation.get_values()
cmk_tranportation.get_evidence()
