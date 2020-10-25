# Probablistic graphical model Imports
import pgmpy
import networkx as nx
# Standard Machine Learning Imports
import pandas as pd
# Display up to 60 columns of a dataframe
pd.set_option('display.max_columns', 60)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

# _________________________________________________________________________________________________________________________________

# Imports for Bayesian Networks
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator, ParameterEstimator
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch, TreeSearch
from sklearn.utils import shuffle
from pgmpy.inference import VariableElimination

# Dataset Name: Breast Cancer Dataset Location: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/ Date: 10/17/2020 Algorithm: Bayesian Belief Networks

# Attributes:
# Class: no-recurrence-events, recurrence-events
# age: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99.
# menopause: lt40, ge40, premeno.
# tumor-size: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59.
# inv-nodes: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39.
# node-caps: yes, no.
# deg-malig: 1, 2, 3.
# breast: left, right.
# breast-quad: left-up, left-low, right-up, right-low, central.
# irradiat: yes, no.


# Import Car Data into a dataframe
columns = ['class', 'age', 'menopause', 'tumor_size','inv_nodes', 'node_caps', 'deg_malig', \
           'breast', 'breast_quad', 'irradiant']

# Load the Data
df = pd.read_csv('breast-cancer.data', header=0, names=columns)
df = shuffle(df)

# Train on 80% of the Data
training = df.iloc[:230,:]
testing = df.iloc[230:,:]

# Format data for PGMPY inference
# Evidence = List of dicts ... e.g. evidence[0] = {col_1 : val_1, col_2: val_2 ....}
evidence = []
# The target list is for checking accuracy in the future
targets = []

for row in testing.iterrows():
    blank_row = {i:None for i in df.columns if i != 'class'}
    for cls in blank_row:
        blank_row[cls] = row[1].to_dict()[cls]
    targets.append(row[1].to_dict()['class'])
    evidence.append(blank_row)


# ******* The task is to infer the "class" node from the data. *******

# Test out Hill climb search. Hill climb is one of the various structure leanring
# Algorithms in PGMPY
est = HillClimbSearch(data=training)

# Remove the paths that go from "class" to another node
blacklisted = [("class",i) for i in df.columns if i != 'class']
estimated_model = est.estimate(black_list=blacklisted)

# Make a Bayesian Model with the edges of the graph
edges = estimated_model.edges()
model = BayesianModel(edges)

# Bayes networks work off conditional probablities ... Estimate with MLE
mle = MaximumLikelihoodEstimator(model, df)
# Prior type Bdeu? Used default from docs
model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")

# Visualize the network
nx.draw(model, with_labels=True)
pylab.show()

# Make a Variable Elimination Object that some how does inference
infer = VariableElimination(model)

# Initialize a list for the predictions we will be making
preds = []

# Loop through the test example evidance to make predictions
for ev in evidence:

    # This is a cheap solution to a key error ... For the purpose of this
    # dataset/first attempt it will do for now
    for key in df.columns:
        if key not in model.nodes():
            del ev[key]
    # Get condtional probabiliy of each class given the evidence
    c_distribution = infer.query(['class'], evidence=ev, show_progress=False)
    # Make a prediciton
    pred = c_distribution.state_names['class'][np.argmax(c_distribution.values)]
    # Add the predictions to all the predictions
    preds.append(pred)

# Check accuracy
count_correct = 0
for pred, tar in zip(preds,targets):
    if pred==tar:
        count_correct+=1

print("Accuracy: ",count_correct/len(testing))

# Accuracy is from 67% to 78% which normally beats prior probability of the classes

# _________________________________________________________________________________________________________________________________
