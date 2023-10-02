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


def startingValues():
    # randomly generate a value, doesn't really matter what it is at step 0
    return np.random.uniform(0, 5) * -1


def train(X, Y, learnRate, epochs):
    # get random number to initially set the Weight and Bias
    Weight = startingValues()
    Bias = startingValues()
    print("Starting Weight: " + str(Weight))
    print("Starting Bias: " + str(Bias))

    Loss = []

    # loop for number of total iterations
    for i in range(epochs):
        # get the forward prediction
        prediction = forward(X, Weight, Bias)

        # back propagation
        derW, derB = backProp(X, Y, prediction)

        # update the model
        Weight, Bias = update(Weight, Bias, derW, derB, learnRate)

    print("final Weight: " + str(Weight))
    print("final Bias: " + str(Bias))
    return Weight, Bias


if __name__ == '__main__':
    print('Colin Houde - Machine Learning HW2')
    # Question 1 train
    trainedWeight, trainedBias = train(X1, Y1, 0.0001, 1000)
    predictedY1 = X1*trainedWeight + trainedBias

    # plot the results
    plt.plot(X1, Y1, '+', label='Actual values')
    plt.plot(X1, predictedY1, label='Predicted values')
    plt.legend()
    plt.show()


