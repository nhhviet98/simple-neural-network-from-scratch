import numpy as np
import matplotlib.pyplot as plt
from Sequential import Sequential
import time

if __name__ == '__main__':
    # Initialize
    NUMBER_SAMPLES = 100
    PLOT = True
    EPOCHS = 10000
    LEARNING_RATE = 0.1

    # Create training data
    np.random.seed(3010)
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]
    x, y = np.random.multivariate_normal(mean, cov, NUMBER_SAMPLES).T
    x_train = np.concatenate((x.reshape(NUMBER_SAMPLES, 1), y.reshape(NUMBER_SAMPLES, 1)), axis=1)
    y_train = np.where(x >= 0, 1.0, 0.0).reshape(NUMBER_SAMPLES, 1)

    # Plot scatter data point
    if PLOT:
        plt.scatter(x=x_train[:, 0], y=x_train[:, 1], c=np.squeeze(y_train))
        plt.axis('equal')
        plt.show()

    # Model
    model = Sequential()
    model.add(layer_name='input', n_unit=2, activation=None)
    model.add(layer_name='dense', n_unit=3, activation='sigmoid')
    model.add(layer_name='dense', n_unit=2, activation='tanh')
    model.add(layer_name='output', n_unit=1, activation='sigmoid')

    # Training model
    t1 = time.time()
    loss = model.fit(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    t2 = time.time()
    print("time to train = ", t2 - t1)

    # Plot loss graph
    if PLOT:
        plt.figure()
        plt.plot(range(EPOCHS), loss)
        plt.show()

    # Predict training set
    y_pred = model.predict(x_train)
    accuracy = model.accuracy_score(y_train, y_pred)
    print("Accuracy Train = ", accuracy)

    # Create test set
    np.random.seed(1998)
    x, y = np.random.multivariate_normal(mean, cov, NUMBER_SAMPLES).T
    x_test = np.concatenate((x.reshape(NUMBER_SAMPLES, 1), y.reshape(NUMBER_SAMPLES, 1)), axis=1)
    y_test = np.where(x >= 0, 1.0, 0.0).reshape(NUMBER_SAMPLES, 1)

    # Plot scatter data point
    if PLOT:
        plt.scatter(x=x_test[:, 0], y=x_test[:, 1], c=np.squeeze(y_test))
        plt.axis('equal')
        plt.show()

    # Predict test set
    y_pred = model.predict(x_test)
    accuracy = model.accuracy_score(y_test, y_pred)
    print("Accuracy Test = ", accuracy)

    print("End program!")