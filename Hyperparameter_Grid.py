# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

# Training Parameters
#----------------------#
# Number of Jobs (Cores to use)
n_jobs = 2
# Number of Random CV Draws
n_iter = 1
n_iter_trees = 1#20
# Number of CV Folds
CV_folds = 2#

#----------------------#
########################
# Hyperparameter Grids #
########################
#----------------------#

if train_mode == True:

    Random_Depths_Readout = 10
    Random_Depths = np.array([2])#, 100, 500])#, 3,10, 20, 50, 100, 150,200, 500])
    N_Features = 4# Random_Depths.shape[0]
    N_Features_Search_Space_Dimension = 10**4


    # Hyperparameter Grid (Readout)
    #------------------------------#
    param_grid_Vanilla_Nets = {'batch_size': [8],
                        'epochs': [50],
                          'learning_rate': [0.001],
                          'height': [20],
                           'depth': [2],
                          'input_dim':[10],
                           'output_dim':[1]}

    param_grid_Nice_Nets = {'batch_size': [8],
                        'epochs': [50],
                          'learning_rate': [0.001],
                          'height': [20],
                            'Depth_Feature_Map': [2],
                            'Depth_Readout_Map': [2],
                          'input_dim':[10],
                           'output_dim':[1]}


else:
    Random_Depths_Readout = 10
    Random_Depths = np.array([2, 100, 500])
    N_Features = 20
    N_Features_Search_Space_Dimension = 10**4


    # Hyperparameter Grid (Readout)
    #------------------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16,32,64],
                        'epochs': [200, 400, 800, 1000, 1200, 1600],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [2, 400, 800, 1000],
                           'depth': [2, 4, 6],
                          'input_dim':[10],
                           'output_dim':[1]}

    param_grid_Nice_Nets = {'batch_size': [16,32,64],
                        'epochs': [2, 400, 800, 1000, 1200, 1600],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [1,50, 200, 400, 800, 1000],
                            'Depth_Feature_Map': [1,2,3,4],
                            'Depth_Readout_Map': [1,2,3,4],
                              'input_dim':[10],
                               'output_dim':[1]}
