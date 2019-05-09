# Models
CNN models.
To run the python files. put balancedData_shuffled, *.py, and *_weights.h5 under the same directory.
The result of training, validation and testing are images names "accuracy.png" and "loss.png"

# DenseNet
1. Compile dense121_train.py.py 
  read images at '/projectnb/cs542sp/idc_classification/data/'
  process the Dataset and shuffle it, name as "balancedData_shuffled" then store it at "./balancedData_shuffled"
  Train the model and get the weights called "densenet_weights.h5".
2. Compile dense121_pred.py
  load dataset called "./balancedData_shuffled"
  load weights for the model ('densenet_weights.h5')
  
  return the predicted result.

# VGG
  The file newvgg.py is our final model.
  1. read dataset at "./balancedData_shuffled"
  2. output weights of the model: 'newvgg_weights.h5'

# AlexNet
  Run alex_net.py
