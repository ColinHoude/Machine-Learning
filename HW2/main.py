import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# X1 and Y1 are the respective columns for the first question
X1 = pd.read_csv('HW2_linear_data.csv', usecols=[0]).to_numpy()
Y1 = pd.read_csv('HW2_linear_data.csv', usecols=[1]).to_numpy()

# X2 and Y2 are the respective columns for the second question
X2 = pd.read_csv('HW2_nonlinear_data.csv', usecols=[0]).to_numpy()
Y2 = pd.read_csv('HW2_nonlinear_data.csv', usecols=[1]).to_numpy()


def forward(input, we, bi):
    # we = weight ; bi = bias
    prediction = np.multiply(we, input) + bi
    return prediction


def forwardCubic(input, we1, we2, we3, bi):
    # we1 = weight1... ; bi = bias
    ax3 = np.multiply(input, we1) ** 3
    bx2 = np.multiply(input, we2) ** 2
    cx = np.multiply(input, we3)
    predict = ax3 + bx2 + cx + bi
    return predict


def backProp(input, output, pred):
    # get the derivatives of the weight and bias and return them
    difference = (output - pred) * -1
    derrWeight = np.mean(np.multiply(input, difference))
    derrBias = np.mean(difference)
    return derrWeight, derrBias


def update(We, Bi, derW, derB, learnRate):
    newWeight = We - learnRate * derW
    newBias = Bi - learnRate * derB
    return newWeight, newBias


def updateCub(We1, We2, We3, Bi, derW, derB, learnRate):
    newWeight1 = We1 - learnRate * derW
    newWeight2 = We2 - learnRate * derW
    newWeight3 = We3 - learnRate * derW
    newBias = Bi - learnRate * derB
    return newWeight1, newWeight2, newWeight3, newBias


def startingValues():
    # randomly generate a value, doesn't really matter what it is at step 0
    return np.random.uniform(0, 5) * -1


def train(X, Y, learnRate, epochs):
    # get random number to initially set the Weight and Bias
    Weight1 = startingValues()
    Bias = startingValues()
    print("Starting Weight: " + str(Weight1))
    print("Starting Bias: " + str(Bias))
    # loop for number of total iterations
    for i in range(epochs):
        prediction = forward(X, Weight1, Bias)

        # back propagation
        derW, derB = backProp(X, Y, prediction)

        # update the model
        Weight1, Bias = update(Weight1, Bias, derW, derB, learnRate)

    print("final Weight: " + str(Weight1))
    print("final Bias: " + str(Bias))
    return Weight1, Bias


def trainCube(X, Y, learnRate, epochs):
    # get random number to initially set the Weight and Bias
    Weight1 = startingValues()
    Weight2 = startingValues()
    Weight3 = startingValues()
    Bias = startingValues()

    print("Starting Weight: " + str(Weight1))
    print("Starting Weight: " + str(Weight2))
    print("Starting Weight: " + str(Weight3))
    print("Starting Bias: " + str(Bias))
    # this is the cubic function train
    # loop for number of total iterations
    for i in range(epochs):
        prediction = forwardCubic(X, Weight1, Weight2, Weight3, Bias)

        # back propagation
        derW, derB = backProp(X, Y, prediction)

        # update the model
        Weight1, Weight2, Weight3, Bias = updateCub(Weight1, Weight2, Weight3, Bias, derW, derB, learnRate)

    print("final Weight1: " + str(Weight1))
    print("final Weight2: " + str(Weight2))
    print("final Weight3: " + str(Weight3))
    print("final Bias: " + str(Bias))
    return Weight1, Weight2, Weight3, Bias


if __name__ == '__main__':
    print('Colin Houde - Machine Learning HW2')
    # Question 1 train
    trainedWeight, trainedBias = train(X1, Y1, 0.0001, 1000)
    predictedY1 = X1*trainedWeight + trainedBias

    # plot the results for question 1
    plt.plot(X1, Y1, '+', label='Actual values')
    plt.plot(X1, predictedY1, label='Predicted values')
    plt.legend()
    plt.show()

    trainedWeight1, trainedWeight2, trainedWeight3, trainedBias1 = trainCube(X2, Y2, 0.000001, 10000)
    predictedY2 = ((np.multiply(trainedWeight1, X2) ** 3) + (np.multiply(trainedWeight2, X2) ** 2) +
                   (np.multiply(trainedWeight3, X2)) + trainedBias1)

    plt.plot(X2, Y2, '+', label='Actual values')
    plt.plot(X2, predictedY2, 'bo', label='Predicted values')
    plt.legend()
    plt.show()





