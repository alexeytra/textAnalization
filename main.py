import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from neural_networks.mlpNN import MLPModel
from neural_networks.cnn import CNNModel

# simple model
mlpmodel = MLPModel()
mlpmodel.activate_mlp_model_v1()

#convolutional neural networ
# cnn_model = CNNModel()
# cnn_model.activate_cnn_model_v1()

# Recurrent neural network


