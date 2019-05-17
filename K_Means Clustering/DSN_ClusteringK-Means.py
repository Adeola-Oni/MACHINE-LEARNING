#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#get dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:4]
Y = dataset.iloc[:, 4]
