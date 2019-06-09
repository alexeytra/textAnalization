import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from neural_networks.mlpNN import MLPModel
from neural_networks.cnn import CNNModel
from neural_networks.rnn import RNNModel
from neural_networks.RnnAndCnnNN import  RnnAndCnn

# simple model
# mlp_model = MLPModel()
# mlp_model.activate_mlp_model_v2()

#convolutional neural networ
# cnn_model = CNNModel()
# cnn_model.activate_cnn_model_v3()

# Recurrent neural network
# rnn_model = RNNModel()
# rnn_model.activate_rnn_model_v3()

#Recurrent neural network (LTSM) and Convolutional neural networ
rcnn = RnnAndCnn()
rcnn.activate_RnnAndCnn_model_v1()
