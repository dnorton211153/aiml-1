# @author Norton 2023 (AI/Machine Learning capstone project - Roadsign Detection
#
# Objective: Use Keras to train a model which can identify roadsigns in a video.

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras
import os

# Source training data:
# https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
#
# GTSRB_Final_Training_Images\GTSRB\Final_Training\Images contains
# one directory for each of the 43 classes (0000 - 00042).
# Each directory contains the corresponding training images and one
# text file with annotations, eg. GT-00000.csv.

# **********************************************
# Image format and naming
# **********************************************
# The images are PPM images (RGB color). Files are numbered in two parts:
#    XXXXX_YYYYY.ppm
# The first part, XXXXX, represents the track number. All images of one class
# with identical track numbers originate from one single physical traffic sign.
# The second part, YYYYY, is a running number within the track. The temporal order
# of the images is preserved.
#
# **********************************************
# Annotation format
# **********************************************
#
# The annotations are stored in CSV format (field separator
# is ";" (semicolon) ). The annotations contain meta information
# about the image and the class id.
#
# In detail, the annotations provide the following fields:
#
# Filename        - Image file the following information applies to
# Width, Height   - Dimensions of the image
# Roi.x1,Roi.y1,
# Roi.x2,Roi.y2   - Location of the sign within the image
# 		  (Images contain a border around the actual sign
#                   of 10 percent of the sign size, at least 5 pixel)
# ClassId         - The class of the traffic sign

modelFile = "model.h5"
learning_rate = 0.001
# The learning_rate is the rate at which the model learns.
# The learning_rate is multiplied by the gradient to determine the amount to adjust the weights.
# The learning_rate is a hyperparameter that can be tuned to improve the model, like how much
# room to wiggle the weights when adjusting them on each pass.

epochs = 5

# Load the training data
def loadTrainingData():

    # Use stored files if they exist
    framesFilename = "frames.npy"
    classesFilename = "classes.npy"

    if os.path.exists(framesFilename) and os.path.exists(classesFilename):
        # load the frames and classes from the files
        frames = np.load(framesFilename)
        classes = np.load(classesFilename)
        print("loaded frames and classes from files")
        return frames, classes

    # Load the images and annotations into a numpy array

    # (1) Iterate through .\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images
    #     and read the sub-directories. Each sub-directory contains the images
    rootdir = "GTSRB_Final_Training_Images\GTSRB\Final_Training\Images"
    frames = []
    classes = []

    for _, dirs, _ in os.walk(rootdir):
        # (2) For each directory, read the corresponding csv file (GT-XXXXX.csv)
        for i, dir in enumerate(dirs):  # each directory represents a roadsign class
            # csv file is {dir}/GT-{dir}.csv
            dataset = os.path.join(rootdir, dir, "GT-" + dir + ".csv")
            print("dataset = ", dataset)

            rows = pd.read_csv(dataset, sep=';')
            rows = rows.reset_index()

            # (3) For each row in the dataframe, read the image and append to the images array
            #     and the class to the classes array
            for _, row in rows.iterrows():

                # store the image in the images array
                image = cv2.imread(os.path.join(rootdir, dir, row["Filename"]))

                # # crop the image based on the Roi.x1,Roi.y1, Roi.x2,Roi.y2
                # image = image[int(row["Roi.Y1"]):int(row["Roi.Y2"]), int(
                #     row["Roi.X1"]):int(row["Roi.X2"])]

                # xScalingFactor = 32 / row["Width"]
                # yScalingFactor = 32 / row["Height"]

                # log the image dimensions
                # print("image.shape = ", image.shape)

                image = cv2.resize(image, (32, 32))
                # Append the image and class to the images and classes arrays

                frames.append(image)
                classes.append(row["ClassId"])

            print("Processed ", i, " directories")

    # (4) Convert the images and classes arrays to numpy arrays
    frames = np.array(frames)
    classes = np.array(classes)
    print("frames.shape = ", frames.shape)
    print("classes.shape = ", classes.shape)

    # dump frames and classes to a file
    np.save("frames.npy", frames)
    np.save("classes.npy", classes)

    return frames, classes

# Define and return the compiled model
def loadModel():
    model = keras.Sequential()

    # The sequential model is a linear stack of neural-network layers.
    
    # The idea here is that for a given video, each frame will be passed through the model,
    # and the model will output a prediction of which traffic sign is on each frame.

    # The model will be trained by feeding it a series of frames and the correct traffic sign
    # for each frame. The model will then adjust its weights to make better predictions.

    # The first layer in the model is a convolutional layer, which is a type of neural network layer
    # that is designed to recognize patterns of changes in images.  You can think of each filter
    # in the convolutional layer as a "feature detector", so for each image, all of the filters
    # in the convolutional layer will be applied to the image, and the output of the convolutional
    # layer will be a set of feature maps, where each feature map is the result of applying a
    # single filter to the image.

    # The "weights" are the values in the filters.

    # Neurons are related to filters in that they are also designed to recognize patterns of
    # changes in the input.  The difference is that a neuron is a single number, whereas a filter
    # is a matrix.  The neuron is the sum of the products of the corresponding elements in the
    # input and the filter.  The neuron is the sum of the products of the corresponding elements
    # in the input and the filter.

    # So for each image, the convolutional layer will output a set of feature maps, where each
    # feature map is the result of applying a single filter to the image (the filter is a matrix).
    # Successful feature detection will result in a feature map that contains a pattern of changes
    # that matches the pattern of changes in the filter.

    # During training, it is the job of the optimizer to adjust the weights in the filters to
    # detect the features in the image.  The optimizer will adjust the weights in the filters
    # by comparing the pattern of changes in the feature map to the pattern of changes in the
    # filter.  The optimizer adjusts weights in the filter to make the pattern of changes
    # in the feature map match the pattern of changes in the filter.

    # The way the filter identifies whether its pattern of changes matches the pattern of changes
    # in the image is by multiplying the filter by the image.  The result of the multiplication
    # is a single number, which is the sum of the products of the corresponding elements in the
    # filter and the image.  This number is called the "activation" of the filter for that image,
    # or the neuron for that image.
    # 
    # Then, this number is passed through an activation function, which
    # is a function that maps the result of the multiplication to a value between 0 and 1.

    # The way the activation function works is that if the result of the multiplication is
    # positive, then the activation function will output a value that is close to 1.  If the
    # result of the multiplication is negative, then the activation function will output a value
    # that is close to 0.
    # 
    # So, you can imagine two curves overlaying each other, where one curve is the filter and
    # the other curve is the image.  The result of the multiplication is the area under the
    # filter curve that is above the image curve.  If the filter curve is above the image curve,
    # then the result of the multiplication will be positive.  If the filter curve is below the
    # image curve, then the result of the multiplication will be negative.

    # So, during training, for each frame of the video, each filter makes a guess as to whether
    # the pattern of changes in the image matches the pattern of changes in the filter.  If the
    # pattern of changes in the image matches the pattern of changes in the filter, then the
    # filter will output a value that is close to 1.  If the pattern of changes in the image
    # does not match the pattern of changes in the filter, then the filter will output a value
    # that is close to 0 (the result of matrix).

    # In this case, the training data is a set of images, and the labels are the correct traffic
    # sign for each image.  If a filter is perfect, then it will output a value that is close to
    # 1 for each image that contains the feature that the filter is designed to detect, and it
    # will output a value that is close to 0 for each image that does not contain the feature.
    # 
    # Since there are 43 different traffic signs and initially 128 filters, so the model will
    # output 43 different values for each image.  It starts with 128 filters/neurons, and then
    # it will reduce the number of filters/neurons to 43.  The idea is that the filters will
    # be reduced to the 43 filters that are most effective at detecting the 43 different traffic
    # signs, which are unique enough that they stand out as the most important feature in a
    # given image, so it's safe to assume that the 43 filters will map to the 43 signs.   

    # Each filter tries to detect a specific feature in the image, where each feature is a pattern
    # The purpose of a filter is to detect a specific feature in the image.
    
    # A "feature" is recognized by the filter when the pattern of changes in the image matches the pattern of changes in the filter.
    # In other words, the filters are used to detect features in the image by looking for patterns of changes in the image.

    # The filter stores its pattern of changes in a matrix, which is called a "kernel", "weight", or "weight matrix".
    # The weights in the filter are adjusted during training to detect the feature in the image.  The weights are adjusted
    # by comparing the pattern of changes in the image to the pattern of changes in the filter.  The weights are adjusted
    # to minimize the difference between the pattern of changes in the image and the pattern of changes in the filter.

    # Matrix multiplication is used to determine the pattern of changes in the image.  So, the pattern of changes in the image
    # is represented by a matrix.  The pattern of changes (aka gradient/slope/direction/derivative/angle) in the filter is 
    # also represented by a matrix.  The pattern of changes in the image is compared to the pattern of changes in the filter by
    # multiplying the two matrices together.  The result of the multiplication is a new matrix (with one fewer dimension than
    # the original matrices).  The result of the multiplication is called the "convolution" of the image and the filter.

    # A filter scans the entire image (or at least the majority with fringes ignored, depending on the exact sizing/settings)
    # with its kernel.  After each shift of the kernel across the image, the convolution is calculated.  If is value is above
    # a certain threshold, the filter is said to have "activated" at that specific location.  The "location" means the specific

    # Patterns of change are identified in matrices by looking for the largest change in the matrix (i.e. most significant change),
    # which is called the "gradient" of the matrix, also known as the "derivative" of the matrix.
    
    # The gradient of a matrix can be represented by a vector, which is a matrix with a single row or column,
    # (i.e. one dimension less than the original matrix).  It is used to determine the "direction" of the largest change 
    # in the matrix.  This "direction of the largest change" is represented by the angle of the vector, i.e. the slope of the line.

    #
    # The activation function is used to determine if the feature is present in the image.
    # The relu activation function returns the input value if it is positive, otherwise it returns 0
    # The input shape is the dimensions of the image (32x32x3)
    model.add(keras.layers.Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
    # There are 128 filters, each with a 3x3 kernel, so there are 128 * (3x3x3 + 1) = 3584 parameters to learn
    # If I were interested in larger features I might choose a 5x5 kernel instead of a 3x3 kernel,
    # which would increase the number of parameters to learn to 128 * (5x5x3 + 1) = 9600 parameters to learn
    # output: model.output_shape =  (None, 32, 32, 128)

    # This is like the model says, "Let's identify the 128 features which are most important, in the sense
    # that they are the most unique to the traffic sign, and ignore the rest."
    # 
    # If a given image had multiple traffic signs, then the model would have to identify the most important
    # feature in the image, which is the traffic sign that is most unique to the image.  The model would
    # ignore the other traffic signs in the image, because they are not unique to the image, and therefore
    # they are not the most important feature in the image. 

    # Bias shifts the activation function to the left (negative) or right (positive), so when the kernel finds a feature
    # and guesses correctly (or close to correct), the bias will be adjusted to make the next guess more accurate.

    # Batch normalization layer to center and scale the input to improve training (Is this necessary with padding='same'?)
    # model.add(keras.layers.BatchNormalization())

    # Max pooling layer with a 2x2 pool size to take the max value from each 2x2 pixel block
    model.add(keras.layers.MaxPooling2D((2, 2)))

    # Repeat with 3x3 filters
    model.add(keras.layers.Conv2D(128, (3, 3), input_shape=(
        32, 32, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    # # Repeat with 3x3 filters
    # model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(
    #     32, 32, 3), activation='relu', padding='same'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))

    # Flatten the output of the convolutional layers to a 1D array (vector) to prepare it for the dense layers
    model.add(keras.layers.Flatten())

    # Dropout layer with a dropout rate of 0.5 to reduce overfitting
    model.add(keras.layers.Dropout(0.5))

    # Each dense layer is a fully connected layer (each node is connected to every node in the previous layer).
    # Dense layers are used to classify the features extracted by the convolutional layers.

    # The first Dense layer uses 128 neurons to classify the features extracted by the convolutional layers,
    # where each neuron is connected to every node in the previous layer and analyzes the features to determine
    # if they are present in the image.  It uses relu (rectified linear unit) activation to make the determination,
    # returning the input activation value if it is positive, otherwise 0.
    model.add(keras.layers.Dense(128, activation='relu'))

    # The second Dense layer uses 43 neurons to classify the features extracted by the first Dense layer.
    # As before, each neuron is connected to every node in the previous layer and analyzes if features are present.
    # The softmax activation function is used to determine the probability that the image belongs to each class.
    # (It takes the exponential of each value in the output, divides each value by the sum of the exponential values,
    # so each value is between 0 and 1, and the sum of all the values is 1.)
    model.add(keras.layers.Dense(43, activation='softmax'))

    # The reason we want 43 neurons in the final Dense layer is because there are 43 classes of road signs.
    # The softmax activation function will return a probability for each class, and the class with the highest
    # probability will be the class that the model predicts for the image.
    #
    # Convolutional layers try to extract features in a differentiable manner,
    # while fully connected layers try to classify the features.
    #
    # So, more dense layers improves classification of extracted features.
    print("After Dense: model.output_shape = ", model.output_shape)

    # -----
    # Don't use this here, but it's an interesting usage of Lambda as a Keras layer:
    # model.add(keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis=1)))
    # -----

    # Compile the model using the adam optimizer, categorical crossentropy loss function, and accuracy metric.
    # The Adam optimizer will adjust the weights of the neural network to minimize the loss function.
    # You can think of weights as the parameters of the neural network that are adjusted to minimize the loss function.
    # Weights are adjusted by taking the derivative of the loss function with respect to the weights.
    # The derivative is used to determine the direction to adjust the weights to minimize the loss function.
    # The learning rate determines how much the weights are adjusted by.
    # The decay parameter is used to adjust the learning rate over time, which helps the model converge.
    # The categorical crossentropy loss function is used for multi-class classification.
    # The accuracy metric is used to determine the accuracy of the model.
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, decay=learning_rate/epochs)
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# trainModel will train the model on the training data, save the model, and return the model
def trainModel(model, trainingFrames, classes):

    # Train the model; epochs is the number of times the model will be trained
    # print size of trainingFrames and classes
    print("trainingFrames.shape = ", trainingFrames.shape)
    print("trainingClasses.shape = ", classes.shape)
    model.fit(trainingFrames, classes, epochs=epochs)

    model.save(modelFile)
    return

# testModel will test the model on the test data and create a prediction file
def testModel(testFrames):

    trainedModel = keras.models.load_model(modelFile)
    # testFrames.shape is currently (12630, 32, 32, 3)

    outputFromPredict = trainedModel.predict(testFrames)
    # outputFromPredict.shape should be (12630, 43),
    # where each row has the array of probabilities that the image belongs to each class

    predictions = np.empty((12630, 1), np.dtype('int'))

    for i in range(0, 12630):
        # get the highest-probability class per row
        greatestChange = np.argmax(outputFromPredict[i])
        greatestChangeInt = np.round(greatestChange, 0).astype(int)
        predictions[i] = greatestChangeInt

    # Create a prediction file from the predictions
    np.savetxt("predictions.txt", predictions, fmt='%d')

    return

# TODO: This is not working yet
def showResults(predictions):
    # # plot the training loss and accuracy
    # N = np.arange(0, NUM_EPOCHS)
    # plt.style.use("ggplot")
    # plt.figure()
    # plt.plot(N, H.history["loss"], label="train_loss")
    # plt.plot(N, H.history["val_loss"], label="val_loss")
    # plt.plot(N, H.history["accuracy"], label="train_acc")
    # plt.plot(N, H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy on Dataset")
    # plt.xlabel("Epoch #")
    # plt.ylabel("Loss/Accuracy")
    # plt.legend(loc="lower left")
    # plt.savefig(args["plot"])
    return


# loadTestingData will load the testing data and return the frames
def loadTestingData():

    # ppm and csv files are in .\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images
    rootdir = "GTSRB_Final_Test_Images\GTSRB\Final_Test\Images"
    dataset = os.path.join(rootdir, "GT-final_test.test.csv")

    testFrames = []

    rows = pd.read_csv(dataset, sep=';')
    rows = rows.reset_index()

    # (3) For each row in the dataframe, read the image and append to the images array
    #     and the class to the classes array
    for _, row in rows.iterrows():
        image = cv2.imread(os.path.join(rootdir, row["Filename"]))

        # crop the image based on the ROI (Region of Interest)
        # image = image[row["Roi.X1"]:row["Roi.X2"], row["Roi.Y1"]:row["Roi.Y2"]]

        image = cv2.resize(image, (32, 32))
        testFrames.append(image)

    # (4) Convert the images and classes arrays to numpy arrays
    testFrames = np.array(testFrames)

    print("testFrames.shape = ", testFrames.shape)

    return testFrames


def main():

    print("Loading training data...")
    frames, classes = loadTrainingData()

    print("Loading model...")
    model = loadModel()

    model.summary()

    print("Training model...")
    trainModel(model, frames, classes)

    print("Training complete, loading Testing data...")
    testFrames = loadTestingData()

    print("Testing complete, creating predictions...")
    predictions = testModel(testFrames)

    showResults(predictions)

    return


if __name__ == "__main__":
    main()
