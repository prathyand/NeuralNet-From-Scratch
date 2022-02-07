## KNN from scratch: 

The goal is to implement KNN classifier from scratch using Python Numpy library, and test the performance of the classifier with sklearn's Knn classifier for Iris and Digits dataset

### Approach:
We have to mainly implement the functions 'fit' and 'predict' for the given skeleton code of KNN classifier. KNN model training is very simple as KNN simply 'remembers' all the training examples. So the fit method simply stores the inputs X(features) and y(labels) to local class variables. 
Predict method is implemented to predict the labels of the input data using the trained model. For KNN this takes most time as for every test example, predict method finds K nearest neighbors and based on hyperparameter 'weights', assigns most suitable label to the test example. First step of the predict method is to create a list of k neighbors sorted based on distance (either Euclidean or Manhattan) for every training example. If the weight parameter is 'uniform', it then simply assigns the most frequently occurring class of k neighbors. If the weight parameter is 'distance', then it assigns the weights to each neighbor inversely proportional to its distance from the test datapoint (1/(1+distance)). Note that here 1 is being added to the denominator to avoid division by 0 error in case of test case happens to be one of the training examples. For each class of the K nearest neighbors, a value is calculated using the weight defined above. The class with maximum total weight is assigned as the prediction label to the test datapoint.

### Performance and result:
Algorithm performs very well in terms of accuracy on both Iris as well as Digits dataset. Accuracy of all the hyperparameter combination matches exactly with SKlearns KNN classifier. Runtime for digits dataset is slightly higher due to the size of the dataset however all the test runs are completed within 1~2 mins for both datasets combined.

### Challenges and issues faced
* Implementation of more efficient data structure like KD tree was attempted to speed up the prediction process but it increased the complexity of the model
* Prediction time for digits dataset was worse compared to iris due to the data size

--------------------------------------------------------------------------------------------

## NN from scratch: 

The goal is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer from scratch, and compare it against SKlearns's MLPClassifier for Iris and Digits dataset

### Approach:
We have to implement different activation functions and cross entropy function before implementing the neural network fit and predict methods. Each of the activation functions also return a derivative based on an input parameter 'derivative' if is value is 'True'. 
After these activation function and their derivatives were implemented, one_hot_encoding function was implemented to transform the class label array into a onehotencoded array for the output layer of the neural network.

Next step was to initialize different parameters of out neural network class. Hidden layer and output layer weights and biases were initialized with random values in the range of (-1,1). In the fit method, feed forward step was implemented and intermediate variables in the feed forward calculations (z1,z2,a1) were stored for backpropagation step. Error at the output layer was calculated as (ypred-yactual) and then backpropagated to calculate gradients of the cross entropy loss w.r.t weights at each layer. Weights were then updated based on these calculated gradients. This process runs iteratively through a for loop till the limit of the iterations is reached. 

predict method was implemented to simply calculate the output layer with feedforward step and the labels were predicted using the np.argmax to identify class corresponding to the most active neuron in the output layer. 

### Performance and result:
Model performs decently in terms of accuracy compared to SKlearn implementation of the MLP for a subset of hyperparameter combinations. For some of the hyperparameters, the accuracy suffers possibly due to the random weight initialization. Model runs fairly quickly for the Iris dataset but takes longer time for Digits dataset

### Challenges and issues faced
* Since this model is not designed to take advantage of multiple CPU/GPU cores, it takes a long time for Digits dataset for training and prediction.
* Accuracy suffers in some hyperparameters due to random weight initialization.
