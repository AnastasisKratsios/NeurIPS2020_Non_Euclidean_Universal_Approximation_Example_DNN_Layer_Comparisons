#!/usr/bin/env python
# coding: utf-8

# # **Example for Paper**: [Non-Euclidean Universal Approximation](https://arxiv.org/abs/2006.02341)
# ---

# ### Mode:
# Use this to test script before running with "train_mode" $\triangleq$ False.

# In[8]:


train_mode = True 


# ### Meta-parameters

# In[9]:


# Test-size Ratio
test_size_ratio = 0.3
min_height = 50


# In[10]:


# load dataset
results_path = "./outputs/models/"
results_tables_path = "./outputs/results/"
raw_data_path_folder = "./inputs/raw/"
data_path_folder = "./inputs/data/"


# ### Imports

# In[11]:


# Load Packages/Modules
exec(open('Init_Dump.py').read())
# Load Hyper-parameter Grid
exec(open('Hyperparameter_Grid.py').read())
# Load Helper Function(s)
exec(open('Helper_Functions.py').read())
exec(open('Helper_Utility.py').read())
exec(open('Optimal_Deep_Feature_and_Readout_Util.py').read())
# Pre-process Data
exec(open('Prepare_Data_California_Housing.py').read())
# Import time separately
import time


# ## Preparing
# 
# We compare three models in this implementation.  Each are feed-forward networks of the same dimensions:
# - **Good model**: repsects our assumptions
# - **Bad model**: does not
# - **Vanilla model**: is a naive feed-forward benchmark
# #### Import Libraries

# #### Set Seed(s):

# In[12]:


# Set seed Tensorflow:
tf.random.set_seed(2020)
# Set seed Numpy:
np.random.seed(2020)


# ---
# ---
# ---

# # Good Model $I$:
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

# In[13]:


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
        if i_depth_readout == 0:
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
print('Built Mode: <Good I>')


# ### Make Predictions

# In[14]:


# Initialize & User Updates
#--------------------------#
y_hat_train_good, y_hat_test_good = build_and_predict_nice_model(n_folds = n_jobs, n_jobs = n_jobs)
print('Cross-Validated Model: <Good I>')


# # Good Model $II$:
# Build and train the good model:
# $$
# \rho \circ f\circ (x,\phi_{\operatorname{Random}}(x)):\mathbb{R}^m\rightarrow \mathbb{R}^n.
# $$
#  - $f$ is a shallow feed-forward network with ReLU activation.  
#  - Readout: $\rho(x) = \operatorname{Leaky-ReLU}\bullet (\exp(\tilde{A}_n)x+\tilde{b}_n)\circ \dots \circ \operatorname{Leaky-ReLU}\bullet (\exp(\tilde{A}_1)x+\tilde{b}_1)$
#  - Feature Map: $\phi_{\operatorname{Random}}(x) = \operatorname{Leaky-ReLU}\bullet (\exp(A_n)x+b_n)\circ \dots \circ\operatorname{Leaky-ReLU}\bullet (\exp(A_1)x+b_1)$,
# 
# where $A_i,\tilde{A}_j$ are square matrices, and $A_i,b_i$ are generated randomly by drawing their components from the standardized Bernoulli distribution.
# 
# 
# The matrices $\exp(A_i)$, and $\exp(\tilde{A}_i)$ are therefore invertible since $\exp$ maps any square matrix into the associated [General Linear Group](https://en.wikipedia.org/wiki/General_linear_group).  

# ## Generate Random Deep Feature(s)

# In[15]:


### Initialize Parameters
#------------------------#
# Initialize History
Randomized_Depth = np.random.poisson(2)
past_val = -1
current_position = 0
# Initalize Features
X_train_features = X_train
X_test_features = X_test

# Construct Deep Randomized Features
#------------------------------------#
# Set Seed
np.random.seed(2020)


# Builds Features
for i in range(N_Features):    
    # Transformations
    #-----------------#
    # Build
    if Randomized_Depth > 0:
        # Note: Write Non-Liearly Transformed Features only if transformation has been applied, only if Depth >0
        
        # Apply Activation
        X_train_features_loop = compositer(X_train_features)
        X_test_features_loop = compositer(X_test_features)
        # Apply Random Weights
        Weights_random = (np.random.binomial(1,.5,(X_train_features_loop.shape[1],X_train_features_loop.shape[1])) - .5)*2 # Generate Random Weights
        X_train_features_loop = np.matmul(X_train_features_loop,Weights_random)
        X_test_features_loop = np.matmul(X_test_features_loop,Weights_random)
        # # Apply Bias
        biases_random = (np.random.binomial(1,.5,X_train_features_loop.shape[1]) -.5)*2         # Generate Random Weights and Biases from Recentered Binomial Law
        X_train_features_loop = X_train_features_loop + biases_random
        X_test_features_loop = X_test_features_loop + biases_random
        
    else:
        X_train_features_loop = X_train_features
        X_test_features_loop = X_test_features

    # Update User #
    #-------------#
    print("Current Step: " +str((i+1)/N_Features))
    
# Coerce into nice form:
X_train_features = pd.DataFrame(X_train_features)
X_test_features = pd.DataFrame(X_test_features)
X_train.reset_index(drop=True, inplace=True)
X_train_features.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test_features.reset_index(drop=True, inplace=True)

# Create Features
Random_Feature_Space_train = pd.concat([X_train,X_train_features],axis=1)
Random_Feature_Space_test = pd.concat([X_test,X_test_features],axis=1)

# Update User #
#-------------#
print('Generated Features: Done!')
print(Random_Feature_Space_train.head())
print(Random_Feature_Space_test.head())


# ## Train DNN Model

# In[16]:


# Reload Grid
exec(open('Hyperparameter_Grid.py').read())
# Adjust Input Space's Dimension
param_grid_Nice_Nets['input_dim'] = [Random_Feature_Space_train.shape[1]]

def def_trainable_layers_Randomized_Feature(height, Depth_Feature_Map, Depth_Readout_Map, learning_rate, input_dim, output_dim):
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
    
    
    #------------------#
    #  Readout Layers  #
    #------------------#   
#     for i_depth_readout in range(Depth_Readout_Map):
#         # First Layer
#         if i_depth_readout == 0:
#             output_layers = fullyConnected_Dense_Invertible(output_dim)(output_layers)
#             output_layers = tf.nn.leaky_relu(output_layers)
#         else:
#             output_layers = fullyConnected_Dense_Invertible(output_dim)(output_layers)
#             output_layers = tf.nn.leaky_relu(output_layers)
    
    
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

def build_and_predict_nice_modelII(n_folds , n_jobs):

    # Deep Feature Network
    Nice_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Randomized_Feature, verbose=True)
    
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
    Nice_Model_CVer.fit(Random_Feature_Space_train,y_train)

    # Write Predictions
    y_hat_train = Nice_Model_CVer.predict(Random_Feature_Space_train)
    y_hat_test = Nice_Model_CVer.predict(Random_Feature_Space_test)
    
    # Return Values
    return y_hat_train, y_hat_test

# Update User
#-------------#
print('Built Mode: <Good II>')


# ### Make Predictions

# In[17]:


# Initialize & User Updates
#--------------------------#
y_hat_train_goodII, y_hat_test_goodII = build_and_predict_nice_modelII(n_folds = n_folds, n_jobs = n_jobs)
print('Cross-Validated Model: "Good II"')


# ---
# ---
# ---

# ---
# # Benchmark(s)
# ---

# #### Reload CV Grid

# In[18]:


exec(open('Hyperparameter_Grid.py').read())


# ---
# ---
# ---

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

# In[19]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Bad_Input_Output(height, Depth_Feature_Map, Depth_Readout_Map, learning_rate, input_dim,output_dim):
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
        if i_depth_readout == 0:
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


# In[20]:


# Initialize & User Updates
#--------------------------#
y_hat_train_bad, y_hat_test_bad = build_and_predict_bad_model(n_folds = n_iter, n_jobs = n_jobs)
print('Cross-Validated: Bad Model')


# ---

# # Benchmark ffNN Model (Vanilla)

# ---

# In[22]:


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

# In[23]:


# Initialize & User Updates
#--------------------------#
y_hat_train_Vanilla, y_hat_test_Vanilla = build_and_predict_Vanilla_model(n_folds = n_jobs, n_jobs = n_jobs)
print('Cross-Validated: Vanilla Model')


# # Record Predictions/ Comparisons
# Generate Classes

# In[24]:


# Benchmark Models #
#------------------#
# Results with Good I Model
Perform_GoodI = reporter(y_hat_train_good,y_hat_test_good,y_train,y_test)
# Results with Good II Model
Perform_GoodII = reporter(y_hat_train_goodII,y_hat_test_goodII,y_train,y_test)


# Benchmark Models Performance #
#------------------------------#
# Results with Bad Model
Perform_Bad = reporter(y_hat_train_bad,y_hat_test_bad,y_train,y_test)
# Results Vanilla
Perform_Vanilla = reporter(y_hat_train_Vanilla,y_hat_test_Vanilla,y_train,y_test)


# In[25]:


# Performance Metrics
#----------------------#
performance_train = pd.DataFrame({
                    'Good I': Perform_GoodI.train,
                    'Good II': Perform_GoodII.train,
                    'Bad': Perform_Bad.train,
                    'Vanilla': Perform_Vanilla.train})

performance_test = pd.DataFrame({
                    'Good I': Perform_GoodI.test,
                    'Good II': Perform_GoodII.test,
                    'Bad': Perform_Bad.test,
                    'Vanilla': Perform_Vanilla.test})

# Write Results
#---------------#
# LaTeX
performance_train.to_latex('./outputs/results/Performance_train.txt')
performance_test.to_latex('./outputs/results/Performance_test.txt')
# Write to Txt
cur_path = os.path.expanduser('./outputs/results/Performance_train_text.txt')
with open(cur_path, "w") as f:
    f.write(str(performance_train))
cur_path = os.path.expanduser('./outputs/results/Performance_test_text.txt')
with open(cur_path, "w") as f:
    f.write(str(performance_test))


# # Live Readings

# In[26]:


print('Et-Voila!')
print(' ')
print(' ')
print('#-------------------#')
print(' PERFORMANCE SUMMARY:')
print('#-------------------#')
print(' ')
print(' ')
print('---------------------')
print('Training Performance')
print('---------------------')
print('-------------------------------------------------------------')
print(performance_train)
print('-------------------------------------------------------------')
print('---------------------')
print('Testing Performance')
print('---------------------')
print('-------------------------------------------------------------')
print(performance_test)
print('-------------------------------------------------------------')
print(' ')
print(' ')
print('😊😊 Fin 😊😊')


# ---
# #### 😊 Fin 😊
# ---
