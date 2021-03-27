# AI_Internship
This is vietnhh branch, AI intership at BAP software.

### Folder tree:
```
|   ActivationFunction.py
|   ModelStructure.PNG
|   README.md
|   Sequential.py
|   SimpleNeuralNetwork.py
|   SimpleNNVietnhh.ipynb
|
\---__pycache__
```
# Libraries
- tqdm
- numpy
- matplotlib
- time

# Simple Neural Network
```
cd vietnhh
cd SimpleNeuralNetwork
python SinpleNeuralNetwork.py
```

# Model Structure

<p align="center">
<img src="vietnhh/SimpleNeuralNetwork/image/ModelStructure.png">
</p>

# Dataset

We'll use 100 points 2D data showed in the iamge below:

<p align="center">
<img src="vietnhh/SimpleNeuralNetwork/image/data_point.png">
</p>

# Result

This is loss graph after each epochs in training phase:

<p align="center">
<img src="vietnhh/SimpleNeuralNetwork/image/loss_graph.png">
</p>

```
Accuracy Train =  1.0
Accuracy Test =  0.99
```

 Because we have very few in training set, so when you choose another seed, you should choose another hyper parameters like number epochs, learning rate, ... Or you can even change the neural network structure using "add" method or change the number of neural or activation function in this models.

