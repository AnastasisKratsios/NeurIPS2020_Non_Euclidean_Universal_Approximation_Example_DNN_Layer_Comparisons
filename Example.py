#!/usr/bin/env python
# coding: utf-8

# # **Example for Paper**: [Non-Euclidean Universal Approximation](https://arxiv.org/abs/2006.02341)

# ## Preping
# 
# We compare three models in this implementation.  Each are feed-forward networks of the same dimensions:
# - **Good model**: repsects our assumptions
# - **Bad model**: does not
# - **Vanilla model**: is a naive feed-forward benchmark
# #### Import Libraries

# In[1]:


# CV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# DL: Tensorflow
import tensorflow as tf
from keras.utils.layer_utils import count_params
from tensorflow.python.framework import ops # Custome Tensorflow Functions
from tensorflow.keras.models import Sequential, Model
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Dense, Input
# DL: Tensorflow - Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Formatting:
import pandas as pd
import numpy as np

# Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Structuring
from pathlib import Path

# Visulatization
import matplotlib.pyplot as plt

# Writing, Reading, Exporting, and Importing
#from sklearn.externals import joblib
import pickle

# Timing
import time

# Misc
import gc
from sklearn.pipeline import Pipeline
import os

# Set-Seed
np.random.seed(2020)


# #### Load Externally-Defined Functions

# In[2]:


# Main Training Utility
exec(open('TCP_Util.py').read())
# Helper Functions Utility
exec(open('Optimal_Deep_Feature_and_Readout_Util.py').read())
# Extra Utilities
exec(open('Grid_Enhanced_Network.py').read())


# #### Load Data

# In[3]:


# load dataset
data_path = "./data/housing_complete.csv"
X = pd.read_csv(data_path)

# Parse/Prepare Data
X_train, y_train, X_test, y_test= prepare_data(data_path, True)


# #### Check and Make Paths

# In[4]:


Path('./outputs/models/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Vanilla/').mkdir(parents=True, exist_ok=True)
Path('./outputs/models/Deep_Features/').mkdir(parents=True, exist_ok=True)
Path('./outputs/tables/').mkdir(parents=True, exist_ok=True)
Path('./outputs/results/').mkdir(parents=True, exist_ok=True)


# ---
# ---
# ---

# # Good Model:
# Build and train the good model:
# $$
# \rho \circ f\circ \phi:\mathbb{R}^m\rightarrow \mathbb{R}^n.
# $$
#  - $f$ is a shallow feed-forward network with ReLU activation.  
#  - Readout: $\rho(x) = \operatorname{Leaky-ReLU}\bullet (\exp(\tilde{A}_n)x+\tilde{b}_n)\circ \dots \circ \operatorname{Leaky-ReLU}\bullet (\exp(\tilde{A}_1)x+\tilde{b}_1)$
#  - Feature Map: $\phi(x) = \operatorname{Leaky-ReLU}\bullet (\exp(A_n)x+b_n)\circ \dots \circ\operatorname{Leaky-ReLU}\bullet (\exp(A_1)x+b_1)$
# 
# where $A_i,\tilde{A}_j$ are square matrices.  
# 
# 
# The matrices $\exp(A_i)$, and $\exp(\tilde{A}_i)$ are therefore invertible since $\exp$ maps any square matrix into the associated [General Linear Group](https://en.wikipedia.org/wiki/General_linear_group).  

# In[5]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Nice_Input_Output(height, Depth_Feature_Map, Depth_Readout_Map, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #------------------#
    # Deep Feature Map #
    #------------------#
    for i_feature_depth in range(Depth_Feature_Map):
        # First Layer
        if i_feature_depth == 0:
            deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(input_layer)
            deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
        else:
            deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
            deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(deep_feature_map)
    # Activation
    core_layers = tf.nn.relu(core_layers)
    # Affine Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)
    
    
    #------------------#
    #  Readout Layers  #
    #------------------#   
    for i_depth_readout in range(Depth_Readout_Map):
        # First Layer
        if i_feature_depth == 0:
            output_layers = fullyConnected_Dense_Invertible(output_dim)(output_layers)
            output_layers = tf.nn.leaky_relu(output_layers)
        else:
            output_layers = fullyConnected_Dense_Invertible(output_dim)(output_layers)
            output_layers = tf.nn.leaky_relu(output_layers)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model

#------------------------------------------------------------------------------------------------#
#                                      Build Predictive Model                                    #
#------------------------------------------------------------------------------------------------#

def build_and_predict_nice_model(n_folds , n_jobs):

    # Deep Feature Network
    Nice_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Nice_Input_Output, verbose=True)
    
    # Randomized CV
    Nice_Model_CVer = RandomizedSearchCV(estimator=Nice_Model_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_Nice_Nets,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit
    Nice_Model_CVer.fit(X_train,y_train)

    # Write Predictions
    y_hat_train = Nice_Model_CVer.predict(X_train)
    y_hat_test = Nice_Model_CVer.predict(X_test)
    
    # Return Values
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Built Model')


# ### Make Predictions

# In[ ]:


# Initialize & User Updates
#--------------------------#
y_hat_train_good, y_hat_test_good = build_and_predict_nice_model(n_folds = 2, n_jobs = 2)
print('Cross-Validated: Good Model')


# # Bad Model:
# Build and train the *bad* model:
# $$
# \rho \circ f\circ \phi:\mathbb{R}^m\rightarrow \mathbb{R}^n.
# $$
#  - $f$ is a shallow feed-forward network with ReLU activation.  
#  - Readout: $\rho(x) = \operatorname{ReLU}\bullet (\exp(\tilde{A}_n)x+\tilde{b}_n)\circ \dots \circ \operatorname{ReLU}\bullet (\exp(\tilde{A}_1)x+\tilde{b}_1)$
#  - Feature Map: $\phi(x) = \operatorname{ReLU}\bullet (\exp(A_n)x+b_n)\circ \dots \circ\operatorname{ReLU}\bullet (\exp(A_1)x+b_1)$
# 
# where $A_i,\tilde{A}_j$ are square matrices.  The maps $\rho$ and $\phi$ are neither injective nor are they surjective since $x\mapsto \operatorname{ReLU}(x)$ is neither injective nor surjective as a map from $\mathbb{R}^k$ to $\mathbb{R}^k$; where $m=n$.  
# 
# *Note*:  The key point here is that the input and output maps are forced to be of the same dimension.  Note that, this also violates the minimal bounds derivated in [this paper](https://arxiv.org/abs/1710.11278) for deep ReLU networks.  

# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Bad_Input_Output(height, Depth_Feature_Map, Depth_Readout_Map, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #------------------#
    # Deep Feature Map #
    #------------------#
    for i_feature_depth in range(Depth_Feature_Map):
        # First Layer
        if i_feature_depth == 0:
            deep_feature_map = fullyConnected_Dense(input_dim)(input_layer)
            deep_feature_map = tf.nn.relu(deep_feature_map)
        else:
            deep_feature_map = fullyConnected_Dense(input_dim)(deep_feature_map)
            deep_feature_map = tf.nn.relu(deep_feature_map)
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(deep_feature_map)
    # Activation
    core_layers = tf.nn.relu(core_layers)
    # Affine Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)
    
    
    #------------------#
    #  Readout Layers  #
    #------------------#   
    for i_depth_readout in range(Depth_Readout_Map):
        # First Layer
        if i_feature_depth == 0:
            output_layers = fullyConnected_Dense(output_dim)(output_layers)
            output_layers = tf.nn.relu(output_layers)
        else:
            output_layers = fullyConnected_Dense(output_dim)(output_layers)
            output_layers = tf.nn.relu(output_layers)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model

#------------------------------------------------------------------------------------------------#
#                                      Build Predictive Model                                    #
#------------------------------------------------------------------------------------------------#

def build_and_predict_bad_model(n_folds , n_jobs):

    # Deep Feature Network
    Bad_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Bad_Input_Output, verbose=True)
    
    # Randomized CV
    Bad_Model_CVer = RandomizedSearchCV(estimator=Bad_Model_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_Nice_Nets,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit
    Bad_Model_CVer.fit(X_train,y_train)

    # Write Predictions
    y_hat_train = Bad_Model_CVer.predict(X_train)
    y_hat_test = Bad_Model_CVer.predict(X_test)
    
    # Return Values
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Built Bad Model')


# In[ ]:


# Initialize & User Updates
#--------------------------#
y_hat_train_bad, y_hat_test_bad = build_and_predict_bad_model(n_folds = 2, n_jobs = 2)
print('Cross-Validated: Vanilla Model')


# # Vanilla Model

# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Vanilla(height, Depth_Feature_Map, Depth_Readout_Map, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.relu(core_layers)
    # Affine Layer (Dense Fully Connected)
    output_layers = fullyConnected_Dense(output_dim)(core_layers)
    
    
    # Define Input/Output Relationship (Arch.)
    trainable_layers_model = tf.keras.Model(input_layer, output_layers)
    
    
    #----------------------------------#
    # Define Optimizer & Compile Archs.
    #----------------------------------#
    opt = Adam(lr=learning_rate)
    trainable_layers_model.compile(optimizer=opt, loss="mae", metrics=["mse", "mae", "mape"])

    return trainable_layers_model

#------------------------------------------------------------------------------------------------#
#                                      Build Predictive Model                                    #
#------------------------------------------------------------------------------------------------#

def build_and_predict_Vanilla_model(n_folds , n_jobs):

    # Deep Feature Network
    Vanilla_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Vanilla, verbose=True)
    
    # Randomized CV
    Vanilla_Model_CVer = RandomizedSearchCV(estimator=Vanilla_Model_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(CV_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_Nice_Nets,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit
    Vanilla_Model_CVer.fit(X_train,y_train)

    # Write Predictions
    y_hat_train = Vanilla_Model_CVer.predict(X_train)
    y_hat_test = Vanilla_Model_CVer.predict(X_test)
    
    # Return Values
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Built Vanilla Model')


# ### Make Predictions

# In[ ]:


# Initialize & User Updates
#--------------------------#
y_hat_train_Vanilla, y_hat_test_Vanilla = build_and_predict_Vanilla_model(n_folds = 2, n_jobs = 2)
print('Cross-Validated: Vanilla Model')


# # Record Predictions/ Comparisons
# Generate Classes

# In[ ]:


# Results with Nice Model
#------------------------#
Train_Good = y_hat_train_good - y_train
Test_Good = y_hat_test_good - y_test
score_Train_good = np.mean(np.abs(Train_Good))
score_Test_good = np.mean(np.abs(Test_Good))

# Results with Bad Model
#-----------------------#
Train_Bad = y_hat_train_bad - y_train
Test_Bad = y_hat_test_bad - y_test
score_Train_bad = np.mean(np.abs(Train_Bad))
score_Test_bad = np.mean(np.abs(Test_Bad))

# # Results Vanilla #
# #-----------------#
Train_Vanilla = y_hat_train_Vanilla - y_train
Test_Vanilla = y_hat_test_Vanilla - y_test
score_Train_Vanilla = np.mean(np.abs(Train_Vanilla))
score_Test_Vanilla = np.mean(np.abs(Test_Vanilla))


# In[ ]:


# Performance Metrics
#----------------------#
performance_out = pd.DataFrame({
'Good': np.array([np.mean(score_Train_good),np.mean(score_Test_good)]),
'Bad': np.array([np.mean(score_Train_bad),np.mean(score_Test_bad)]),
'Vanilla': np.array([np.mean(score_Train_Vanilla),np.mean(score_Test_Vanilla)])
},index=['MAE: Train','MAE: Test'])

# Write Results
#---------------#
# LaTeX
performance_out.to_latex('./outputs/results/Performance.txt')
# Write to Txt
cur_path = os.path.expanduser('./outputs/results/Performance_text.txt')
with open(cur_path, "w") as f:
    f.write(str(performance_out))


# # Live Readings

# In[ ]:


print('Et-Voila!')
print(performance_out)


# ---
# #### 😊 Fin 😊
# ---
