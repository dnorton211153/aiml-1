# AI/Machine Learning capstone project - Roadsign Detection
Objective: Use Keras to train a model which can identify roadsigns in a video.

Source training data: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Images and annotations ("classes") are loaded into numpy arrays.

There are 128 filters in the first Conv2D layer, each with a 3x3 kernel; 128 * (3x3x3 + 1) = 3584 parameters to learn

(If I were interested in larger features I might choosen to start with a 5x5 kernel instead of a 3x3 kernel,
which would increase the number of parameters to learn to 128 * (5x5x3 + 1) = 9600 parameters to learn.)

After the Conv2D layer is a "Max Pooling" layer to take the max value from each 2x2 pixel block.

(Repeating the pattern (Conv2d + MaxPooling2D) extracts more features from the image.)

Then, the output is flattened to a 1D array (vector) to prepare it for the dense layers.

Each dense layer is a fully connected layer (each node is connected to every node in the previous layer).
Dense layers are used to classify the features extracted by the convolutional layers.

The first Dense layer uses 128 neurons to classify the features extracted by the convolutional layers,
where each neuron is connected to every node in the previous layer and analyzes the features to determine
if they are present in the image.  It uses relu (rectified linear unit) activation to make the determination,
returning the input activation value if it is positive, otherwise 0.

The second Dense layer uses 43 neurons to classify the features extracted by the first Dense layer.
As before, each neuron is connected to every node in the previous layer and analyzes if features are present.
The softmax activation function is used to determine the probability that the image belongs to each class.
(It takes the exponential of each value in the output, divides each value by the sum of the exponential values,
so each value is between 0 and 1, and the sum of all the values is 1.)

The reason we want 43 neurons in the final Dense layer is because there are 43 classes of road signs.
The softmax activation function will return a probability for each class, and the class with the highest
probability will be the class that the model predicts for the image.

(More dense layers improves classification of extracted features.

The model is compiled using the Adam optimizer, categorical crossentropy loss function, and accuracy metric.

The Adam optimizer will adjust the weights of the neural network to minimize the loss function.
The categorical crossentropy loss function serves well in this purpose, for multi-class classification.

After 5 epochs with this simple model, the accuracy is ~95%.