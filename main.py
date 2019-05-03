import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from neural_networks import mlpNN
from neural_networks.cnn import CNNModel

# simple model
# mlpNN.call_mlpNN()

#convolutional neural networ
cnn_model = CNNModel()
cnn_model.get_cnn_model_v1()

# Recurrent neural network


