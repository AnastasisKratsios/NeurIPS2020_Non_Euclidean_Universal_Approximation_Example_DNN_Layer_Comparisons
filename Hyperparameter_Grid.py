# This file contains the hyper-parameter grids used to train the imprinted-tree nets.

# Training Parameters
#----------------------#
# Number of Jobs (Cores to use)
n_jobs = 70
# Number of Random CV Draws
n_iter = 10
# Number of CV Folds
CV_folds = 4#
n_folds = CV_folds

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
                        'epochs': [20],
                          'learning_rate': [0.001],
                          'height': [20],
                           'depth': [1],
                          'input_dim':[15],
                           'output_dim':[1]}

    param_grid_Nice_Nets = {'batch_size': [8],
                        'epochs': [20],
                          'learning_rate': [0.001],
                          'height': [20],
                            'Depth_Feature_Map': [3],
                            'Depth_Readout_Map': [3],
                          'input_dim':[15],
                           'output_dim':[1]}


else:
    Random_Depths_Readout = 10
    Random_Depths = np.array([2, 100, 500])
    N_Features = 1000
    N_Features_Search_Space_Dimension = 10**4


    # Hyperparameter Grid (Readout)
    #------------------------------#
    param_grid_Vanilla_Nets = {'batch_size': [16,32],
                        'epochs': [200, 400, 800, 1000, 1200],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [200, 250, 400, 600,800, 1000],
                           'depth': [1,2],
                          'input_dim':[15],
                           'output_dim':[1]}

    param_grid_Nice_Nets = {'batch_size': [16,32],
                        'epochs': [200,400, 800, 1000, 1200],
                          'learning_rate': [0.0001,0.0005,0.005],
                          'height': [200, 250, 400, 600,800, 1000],
                            'Depth_Feature_Map': [2,3,4,5],
                            'Depth_Readout_Map': [2,3,4,5],
                              'input_dim':[15],
                               'output_dim':[1]}
