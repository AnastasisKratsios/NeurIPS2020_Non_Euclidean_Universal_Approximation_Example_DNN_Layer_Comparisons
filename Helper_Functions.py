#!/usr/bin/env python
# coding: utf-8

# # Helper Function(s)
# A little list of useful helper functions when building the architope!

# In[ ]:


# MAPE, between 0 and 100
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true.shape = (y_true.shape[0], 1)
    y_pred.shape = (y_pred.shape[0], 1)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Deep Learning Helper(s)

# ## Custom Layers
#  - Fully Conneted Dense: Typical Feed-Forward Layer
#  - Fully Connected Dense Invertible: Necessarily satisfies for input and output layer(s)
#  - Fully Connected Dense Destructor: Violates Assumptions for both input and ouput layer(s) (it is neither injective nor surjective)

# In[ ]:


class fullyConnected_Dense(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
class fullyConnected_Dense_Invertible(tf.keras.layers.Layer):

    def __init__(self, units=16, input_dim=32):
        super(fullyConnected_Dense_Invertible, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='Weights_ffNN',
                                 shape=(input_shape[-1], input_shape[-1]),
                               initializer='zeros',
                               trainable=True)
        self.b = self.add_weight(name='bias_ffNN',
                                 shape=(self.units,),
                               initializer='zeros',
                               trainable=True)

    def call(self, inputs):
        expw = tf.linalg.expm(self.w)
        return tf.matmul(inputs, expw) + self.b


# In[ ]:


#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_trainable_layers_Nice_Input_Output(height, depth, learning_rate, input_dim, output_dim):
    #----------------------------#
    # Maximally Interacting Layer #
    #-----------------------------#
    # Initialize Inputs
    input_layer = tf.keras.Input(shape=(input_dim,))
    
    
    #------------------#
    # Deep Feature Map #
    #------------------#
    # For this implementation we do not use a "deep feature map!"
#     if Depth_Feature_Map >0:
#         for i_feature_depth in range(Depth_Feature_Map):
#             # First Layer
#             if i_feature_depth == 0:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(input_layer)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#             else:
#                 deep_feature_map = fullyConnected_Dense_Invertible(input_dim)(deep_feature_map)
#                 deep_feature_map = tf.nn.leaky_relu(deep_feature_map)
#     else:
#         deep_feature_map = input_layer
        
    
    #------------------#
    #   Core Layers    #
    #------------------#
    core_layers = fullyConnected_Dense(height)(input_layer)
    # Activation
    core_layers = tf.nn.swish(core_layers)
    # Train additional Depth?
    if depth>1:
        # Add additional deep layer(s)
        for depth_i in range(1,depth):
            core_layers = fullyConnected_Dense(height)(core_layers)
            # Activation
            core_layers = tf.nn.swish(core_layers)
    
    #------------------#
    #  Readout Layers  #
    #------------------# 
    # Affine (Readout) Layer (Dense Fully Connected)
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

def build_ffNN(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train, X_test_partial,X_test):

    # Deep Feature Network
    Nice_Model_CV = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=def_trainable_layers_Nice_Input_Output, verbose=True)
    
    # Randomized CV
    Nice_Model_CVer = RandomizedSearchCV(estimator=Nice_Model_CV, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit Model #
    #-----------#
    Nice_Model_CVer.fit(X_train,y_train)

    # Write Predictions #
    #-------------------#
    y_hat_train = Nice_Model_CVer.predict(X_test_partial)
    y_hat_test = Nice_Model_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = Nice_Model_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_ffNN = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])
    
    # Return Values #
    #---------------#
    return y_hat_train, y_hat_test, N_params_best_ffNN

# Update User
#-------------#
print('Deep Feature Builder - Ready')



#------------------------------------------------------------------------------------------------#
#                                      Define Predictive Model                                   #
#------------------------------------------------------------------------------------------------#

def def_simple_deep_classifer(height, depth, learning_rate, input_dim, output_dim):
    # Initialize Simple Deep Classifier
    simple_deep_classifier = tf.keras.Sequential()
    for d_i in range(depth):
        simple_deep_classifier.add(tf.keras.layers.Dense(height, activation='relu'))

    simple_deep_classifier.add(tf.keras.layers.Dense(output_dim, activation='sigmoid'))

    # Compile Simple Deep Classifier
    simple_deep_classifier.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Return Output
    return simple_deep_classifier

#------------------------------------------------------------------------------------------------#
#                                  Build Deep Classifier Model                                   #
#------------------------------------------------------------------------------------------------#
from tensorflow.keras import Sequential
def build_simple_deep_classifier(n_folds , n_jobs, n_iter, param_grid_in, X_train, y_train,X_test):

    # Deep Feature Network
    CV_simple_deep_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=def_simple_deep_classifer, verbose=True)
    
    # Randomized CV
    CV_simple_deep_classifier_CVer = RandomizedSearchCV(estimator=CV_simple_deep_classifier, 
                                    n_jobs=n_jobs,
                                    cv=KFold(n_folds, random_state=2020, shuffle=True),
                                    param_distributions=param_grid_in,
                                    n_iter=n_iter,
                                    return_train_score=True,
                                    random_state=2020,
                                    verbose=10)
    
    # Fit
    CV_simple_deep_classifier_CVer.fit(X_train,y_train)

    # Make Prediction(s)
    predicted_classes_train = CV_simple_deep_classifier_CVer.predict(X_train)
    predicted_classes_test = CV_simple_deep_classifier_CVer.predict(X_test)
    
    # Counter number of parameters #
    #------------------------------#
    # Extract Best Model
    best_model = CV_simple_deep_classifier_CVer.best_estimator_
    # Count Number of Parameters
    N_params_best_classifier = np.sum([np.prod(v.get_shape().as_list()) for v in best_model.model.trainable_variables])

    
    # Return Values
    return predicted_classes_train, predicted_classes_test, N_params_best_classifier

# Update User
#-------------#
print('Deep Classifier - Ready')



#-------------------------------#
#=### Results & Summarizing ###=#
#-------------------------------#
def reporter(y_train_hat_in,y_test_hat_in,y_train_in,y_test_in):
    # Training Performance
    Training_performance = np.array([mean_absolute_error(y_train_hat_in,y_train_in),
                                mean_squared_error(y_train_hat_in,y_train_in),
                                   mean_absolute_percentage_error(y_train_hat_in,y_train_in)])
    # Testing Performance
    Test_performance = np.array([mean_absolute_error(y_test_hat_in,y_test_in),
                                mean_squared_error(y_test_hat_in,y_test_in),
                                   mean_absolute_percentage_error(y_test_hat_in,y_test_in)])
    # Organize into Dataframe
    Performance_dataframe = pd.DataFrame({'train': Training_performance,'test': Test_performance})
    Performance_dataframe.index = ["MAE","MSE","MAPE"]
    # return output
    return Performance_dataframe