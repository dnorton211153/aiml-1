# AI/Machine Learning capstone project - Roadsign Detection
Objective: Use Keras to train a model which can identify roadsigns in a video.

Source training data: https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Images and annotations ("classes") are loaded into numpy arrays;
The idea here is that for a given video, each frame will be passed through the model,
and the model will output a prediction of which traffic sign is on each frame.

The model will be trained by feeding it a series of frames and the correct traffic sign
for each frame.  The model will then adjust its weights to make better predictions.

The first layer in the model is a convolutional layer, which is a type of neural network layer
that is designed to recognize patterns of changes in images.  You can think of each filter
in the convolutional layer as a "feature detector", so for each image, all of the filters
in the convolutional layer will be applied to the image, and the output of the convolutional
layer will be a set of feature maps, where each feature map is the result of applying a
single filter to the image.

The "weights" are the values in the filters.

Neurons are related to filters in that they are also designed to recognize patterns of
changes in the input.  The difference is that a neuron is a single number, whereas a filter
is a matrix.  The neuron is the sum of the products of the corresponding elements in the
input and the filter.  The neuron is the sum of the products of the corresponding elements
in the input and the filter.

So for each image, the convolutional layer will output a set of feature maps, where each
feature map is the result of applying a single filter to the image (the filter is a matrix).
Successful feature detection will result in a feature map that contains a pattern of changes
that matches the pattern of changes in the filter.

During training, it is the job of the optimizer to adjust the weights in the filters to
detect the features in the image.  The optimizer will adjust the weights in the filters
by comparing the pattern of changes in the feature map to the pattern of changes in the
filter.  The optimizer adjusts weights in the filter to make the pattern of changes
in the feature map match the pattern of changes in the filter.

The way the filter identifies whether its pattern of changes matches the pattern of changes
in the image is by multiplying the filter by the image.  The result of the multiplication
is a single number, which is the sum of the products of the corresponding elements in the
filter and the image.  This number is called the "activation" of the filter for that image,
or the neuron for that image.

Then, this number is passed through an activation function, which
is a function that maps the result of the multiplication to a value between 0 and 1.

The way the activation function works is that if the result of the multiplication is
positive, then the activation function will output a value that is close to 1.  If the
result of the multiplication is negative, then the activation function will output a value
that is close to 0.

So, you can imagine two curves overlaying each other, where one curve is the filter and
the other curve is the image.  The result of the multiplication is the area under the
filter curve that is above the image curve.  If the filter curve is above the image curve,
then the result of the multiplication will be positive.  If the filter curve is below the
image curve, then the result of the multiplication will be negative.

So, during training, for each frame of the video, each filter makes a guess as to whether
the pattern of changes in the image matches the pattern of changes in the filter.  If the
pattern of changes in the image matches the pattern of changes in the filter, then the
filter will output a value that is close to 1.  If the pattern of changes in the image
does not match the pattern of changes in the filter, then the filter will output a value
that is close to 0 (the result of matrix).

In this case, the training data is a set of images, and the labels are the correct traffic
sign for each image.  If a filter is perfect, then it will output a value that is close to
1 for each image that contains the feature that the filter is designed to detect, and it
will output a value that is close to 0 for each image that does not contain the feature.

Since there are 43 different traffic signs and initially 128 filters, so the model will
output 43 different values for each image.  It starts with 128 filters/neurons, and then
it will reduce the number of filters/neurons to 43.  The idea is that the filters will
be reduced to the 43 filters that are most effective at detecting the 43 different traffic
signs, which are unique enough that they stand out as the most important feature in a
given image, so it's safe to assume that the 43 filters will map to the 43 signs.   

Each filter tries to detect a specific feature in the image, where each feature is a pattern
The purpose of a filter is to detect a specific feature in the image.
    
A "feature" is recognized by the filter when the pattern of changes in the image matches the pattern of changes in the filter.
In other words, the filters are used to detect features in the image by looking for *patterns of changes* in the image.

The filter stores its pattern of changes in a matrix, which is called a "kernel", "weight", or "weight matrix".
The weights in the filter are adjusted during training to detect the feature in the image.  The weights are adjusted
by comparing the pattern of changes in the image to the pattern of changes in the filter.  The weights are adjusted
to minimize the difference between the pattern of changes in the image and the pattern of changes in the filter.

Matrix multiplication is used to determine the pattern of changes in the image.  So, the pattern of changes in the image
is represented by a matrix.  The pattern of changes (aka gradient/slope/direction/derivative/angle) in the filter is 
also represented by a matrix.  The pattern of changes in the image is compared to the pattern of changes in the filter by
multiplying the two matrices together.  The result of the multiplication is a new matrix (with one fewer dimension than
the original matrices).  The result of the multiplication is called the "convolution" of the image and the filter.

A filter scans the entire image (or at least the majority with fringes ignored, depending on the exact sizing/settings)
with its kernel.  After each shift of the kernel across the image, the convolution is calculated.  If is value is above
a certain threshold, the filter is said to have "activated" at that specific location.  The "location" means the specific

Patterns of change are identified in matrices by looking for the largest change in the matrix (i.e. most significant change),
which is called the "gradient" of the matrix, also known as the "derivative" of the matrix.
    
The gradient of a matrix can be represented by a vector, which is a matrix with a single row or column,
(i.e. one dimension less than the original matrix).  It is used to determine the "direction" of the largest change 
in the matrix.  This "direction of the largest change" is represented by the angle of the vector, i.e. the slope of the line.

The activation function is used to determine if the feature is present in the image.
The relu activation function returns the input value if it is positive, otherwise it returns 0

There are 128 filters in the first Conv2D layer, each with a 3x3 kernel; 128 * (3x3x3 + 1) = 3584 parameters to learn
If I were interested in larger features I might choose a 5x5 kernel instead of a 3x3 kernel,
which would increase the number of parameters to learn to 128 * (5x5x3 + 1) = 9600 parameters to learn

*output: model.output_shape =  (None, 32, 32, 128)*

This is like the model says, "Let's identify the 128 features which are most important, in the sense
that they are the most unique to the traffic sign, and ignore the rest."

If a given image had multiple traffic signs, then the model would have to identify the most important
feature in the image, which is the traffic sign that is most unique to the image.  The model would
ignore the other traffic signs in the image, because they are not unique to the image, and therefore
they are not the most important feature in the image. 

"Bias" shifts the activation function to the left (negative) or right (positive), so when the kernel finds a feature
and guesses correctly (or close to correct), the bias will be adjusted to make the next guess more accurate.

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
You can think of weights as the parameters of the neural network that are adjusted to minimize the loss function.

The learning rate determines how much the weights are adjusted by.
The decay parameter is used to adjust the learning rate over time, which helps the model converge.
The categorical crossentropy loss function is used for multi-class classification.
The accuracy metric is used to determine the accuracy of the model.

The model is trained using the training data, and the validation data is used to determine the accuracy of the model.

After 5 epochs with this simple model, the accuracy is ~95%.