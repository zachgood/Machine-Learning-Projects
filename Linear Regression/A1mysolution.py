import numpy as np
import pandas as pd


def standardize(X):
    return (X - X.mean(0)) / X.std(0)

# Standardize the input data in X and return a dictionary with keys named means, stds, and w
def train_long(X, T):
    # Check that X.shape[0] is equal to T.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError('X and T have different shapes. X = {} & T = {}'.format(X.shape[0], T.shape[0]))

    # Get mean of each column in X
    means = X.mean(0)

    # Get standard dev of each column in X
    stds = X.std(0)

    # Standardize values of X
    Xs = (X - means) / stds

    # Add a column of 1s to X for the bias
    Xs1 = np.insert(Xs, 0, 1, 1)

    # Get the weights
    w = np.linalg.lstsq(Xs1.T @ Xs1, Xs1.T @ T)[0]

    return {'means': means, 'stds': stds, 'w': w}


def train(X, T):
    # Check that X.shape[0] is equal to T.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError('X and T have different shapes. X = {} & T = {}'.format(X.shape[0], T.shape[0]))

    # Standardize and add a column of 1s
    Xs1 = np.insert(standardize(X), 0, 1, 1)

    return {'means': X.mean(0), 'stds': X.std(0), 'w': np.linalg.lstsq(Xs1.T @ Xs1, Xs1.T @ T)[0]}


# Standardize its input data X by using the means and standard deviations in the dictionary returned by train
def use_long(model, X):
    # Standardize values of X
    Xs = (X - model.get('means')) / model.get('stds')

    # Add column of 1s for bias
    Xs1 = np.insert(Xs, 0, 1, 1)

    # Multiply each row by the weights and sum each row.  AKA dot product
    prediction = Xs1 @ model.get('w')

    return prediction


def use(model, X):
    return np.insert(standardize(X), 0, 1, 1) @ model.get('w')


# Returns the square root of the mean of the squared error between predict and T
def rmse_long(predict, T):
    # Get the squared error of each prediction
    sqerr = (predict - T)**2

    # Get the mean of the squared error
    mean = sqerr.mean(0)

    # Take the square root
    root = np.sqrt(mean)
    return root[0]

def rmse(predict, T):
    return np.sqrt(np.mean((predict - T)**2))


# performs the incremental training process described in class as stochastic gradient descent (SGC).
# The result of this function is a dictionary with the same keys as the dictionary returned by the above train function
def trainSGD(X, T, learningRate, numberOfIterations):
    # Check that X.shape[0] is equal to T.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError('X and T have different shapes. X = {} & T = {}'.format(X.shape[0], T.shape[0]))

    # Standardize and add a column of 1s
    Xs1 = np.insert(standardize(X), 0, 1, 1)

    w = np.zeros((Xs1.shape[1], T.shape[1]))

    for iter in range(numberOfIterations):
        print('iter # ', iter)
        for n in range(Xs1.shape[0]):
            predicted = Xs1[n:n + 1, :] @ w  # n:n+1 is used instead of n to preserve the 2-dimensional matrix structure
            # Update w using derivative of error for nth sample
            w += learningRate * Xs1[n:n + 1, :].T * (T[n:n + 1, :] - predicted)

    return {'means': X.mean(0), 'stds': X.std(0), 'w': w}


if __name__ == '__main__':
    X = np.arange(10).reshape((5, 2))
    # # T = X[:, 0:1] + 2 * X[:, 1:2] + np.random.uniform(-1, 1, (5, 1))
    T = np.array([[  2.51574416], [  8.85051723], [ 13.74938177], [ 19.75880574], [ 26.4641601 ]])
    print('Inputs')
    print(X)
    print('Targets')
    print(T)

    model = train(X, T)
    # print(model)
    # model_SGD = trainSGD(X, T, 0.01, 200)
    # print(model_SGD)
    #
    predicted = use(model, X)
    # print(predicted)
    # predicted_SGD = use(model_SGD, X)
    # print(predicted_SGD)
    #
    # print(rmse(predicted, T))
    # print(rmse(predicted_SGD, T))

    # modelSGD = trainSGD(X, T, 0.01, 100)
    # print(modelSGD)
    #
    # predictedSGD = use(modelSGD, X)
    # print(predictedSGD)
    #
    # print(rmse(predictedSGD, T))

    print('predicted')
    print(predicted)

    eachrmse = np.sqrt((predicted - T)**2)
    print(eachrmse)
    check = ((26.4641601 - 26.02874588)**2)**(1/2.0)
    print(check)

    dubT = np.column_stack((T, T))
    dubP = np.column_stack((predicted, predicted))
    eachrmse = np.sqrt((dubP - dubT) ** 2)
    print(eachrmse)
    cnames = ['a', 'b']
    print(cnames)

    for col, name in zip(eachrmse.T, cnames):
        print('name ' + name)
        print(col)




    # X = np.arange(15).reshape((5, 3))
    # T = np.array([[0.2, 1.2], [5, 6], [11.6, 12.6], [20, 21], [30.2, 31.2]])
    #
    # model = trainSGD(X, T, 0.01, 1000)
    # print(model)
    # predicted = use(model, X)
    # print(predicted)
    # print(rmse(predicted, T))

