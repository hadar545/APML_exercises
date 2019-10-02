import keras
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.activations import softmax
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras import backend as K
from keras import regularizers
from sklearn.decomposition import PCA
import time

colors = ['gold', 'lightblue', 'magenta', 'navy', 'coral', 'teal', 'maroon', 'aquamarine', 'purple', 'lime']

# define the functions we would like to predict:
num_of_functions = 3
size = 4
W = 4 * (np.random.random((size, size)) - 0.5)
y = {
    0: lambda x: np.sum(np.dot(x, W), axis=1),
    1: lambda x: np.max(x, axis=1),
    2: lambda x: np.log(np.sum(np.exp(np.dot(x, W)), axis=1))
}
O1 = 0
O2 = 1

def learn_linear(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a linear model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (w, training_loss, test_loss):
         w: the weights of the linear model
         training_loss: the training loss at each iteration
         test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    w = {func_id: np.zeros(input_size) for func_id in range(num_of_functions)}
    for func_id in range(num_of_functions):
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]

            # calculate the loss and derivatives:
            l2_reg = 0.5 * lamb * np.power(LA.norm(w[func_id]), 2)
            p = np.dot(x, w[func_id])
            loss = np.mean(np.power(p - y, 2) + l2_reg)
            p_test = np.dot(X['test'], w[func_id])
            iteration_test_loss = np.mean(np.power(p_test - Y[func_id]['test'], 2) + l2_reg)
            outer_derivative = np.tile(2 * (np.dot(x, w[func_id]) - y), (4, 1)).T
            dl_dw = np.mean(x * outer_derivative + np.tile(lamb * w[func_id], (batch_size, 1)), axis=0)

            # update the model and record the loss:
            w[func_id] -= learning_rate * dl_dw
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)

    return w, training_loss, test_loss

def forward(cnn_model, x):
    """
    Given the CNN model, fill up a dictionary with the forward pass values.
    :param cnn_model: the model
    :param x: the input of the CNN
    :return: a dictionary with the forward pass values
    """

    fwd = {}
    fwd['x'] = x
    fwd['o1'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [np.array(cnn_model['w1'])], mode='same'))
    fwd['o2'] = np.maximum(np.zeros(np.shape(x)), signal.convolve2d(x, [cnn_model['w2']], mode='same'))
    fwd['m1'] = np.vstack(
        (np.maximum(fwd['o1'][:, 0], fwd['o1'][:, 1]), np.maximum(fwd['o1'][:, 2], fwd['o1'][:, 3]))).T
    fwd['m2'] = np.vstack(
        (np.maximum(fwd['o2'][:, 0], fwd['o2'][:, 1]), np.maximum(fwd['o2'][:, 2], fwd['o2'][:, 3]))).T
    fwd['m1_argmax'] = np.vstack((np.argmax(np.dstack((fwd['o1'][:, 0], fwd['o1'][:, 1])), axis=2),
                                  np.argmax(np.dstack((fwd['o1'][:, 2], fwd['o1'][:, 3])), axis=2))).T
    fwd['m2_argmax'] = np.vstack((np.argmax(np.dstack((fwd['o2'][:, 0], fwd['o2'][:, 1])), axis=2),
                                  np.argmax(np.dstack((fwd['o2'][:, 2], fwd['o2'][:, 3])), axis=2))).T
    fwd['p'] = np.hstack((fwd['m1'], fwd['m2']))
    fwd['output'] = np.dot(fwd['p'], cnn_model['u'])
    return fwd


def backprop(model, y, fwd, batch_size):
    """
    given the forward pass values and the labels, calculate the derivatives
    using the back propagation algorithm
    :param model: the model
    :param y: the labels
    :param fwd: the forward pass values
    :param batch_size: the batch size
    :return: a tuple of (dl_dw1, dl_dw2, dl_du)
            dl_dw1: the derivative of the w1 vector
            dl_dw2: the derivative of the w2 vector
            dl_du: the derivative of the u vector
    """

    p_val = fwd['output']
    dl_dp = 2 * (p_val - y)
    dl_du = fwd['p'] * dl_dp[np.newaxis].T
    dp_dm = np.tile(model['u'], (batch_size, 1))

    dm_do1 = np.hstack((1 - fwd['m1_argmax'], fwd['m1_argmax']))
    dm_do1[:, [1, 2]] = dm_do1[:, [2, 1]]
    dm_do1_slice1 = np.swapaxes(np.moveaxis(dm_do1[:, [0, 1]][np.newaxis], 0, -1), 1, 2)
    dm_do1_slice2 = np.swapaxes(np.moveaxis(dm_do1[:, [2, 3]][np.newaxis], 0, -1), 1, 2)
    dm_do2 = np.hstack((1 - fwd['m2_argmax'], fwd['m2_argmax']))
    dm_do2[:, [1, 2]] = dm_do2[:, [2, 1]]
    dm_do2_slice1 = np.swapaxes(np.moveaxis(dm_do2[:, [0, 1]][np.newaxis], 0, -1), 1, 2)
    dm_do2_slice2 = np.swapaxes(np.moveaxis(dm_do2[:, [2, 3]][np.newaxis], 0, -1), 1, 2)

    # calculate dl/dw1
    xs_frame = create_xs_frame(fwd['x'])
    xs_frame = np.swapaxes(xs_frame, 1, 2)
    positive_o1 = np.moveaxis((fwd['o1'] > 0)[np.newaxis], 0, -1)
    do1_dw1 = xs_frame * positive_o1
    dl_dw1_1 = np.einsum('abc,acd->abd', dm_do1_slice1, do1_dw1[:, [0, 1], :])
    dl_dw1_2 = np.einsum('abc,acd->abd', dm_do1_slice2, do1_dw1[:, [2, 3], :])
    dl_dw1 = np.hstack((dl_dw1_1, dl_dw1_2))
    dp_dm_slice = dp_dm[:, [0, 1]]
    dl_dw1 = np.swapaxes(dl_dp[np.newaxis], 0, 1) * np.einsum('ab,abc->ac', dp_dm_slice, dl_dw1)

    # calculate dl/dw2
    positive_o2 = (fwd['o2'] > 0)[np.newaxis]
    positive_o2 = np.moveaxis(positive_o2, 0, -1)
    do2_dw2 = xs_frame * positive_o2
    dl_dw2_1 = np.einsum('abc,acd->abd', dm_do2_slice1, do2_dw2[:, [0, 1], :])
    dl_dw2_2 = np.einsum('abc,acd->abd', dm_do2_slice2, do2_dw2[:, [2, 3], :])
    dl_dw2 = np.hstack((dl_dw2_1, dl_dw2_2))
    dl_dw2 = np.swapaxes(dl_dp[np.newaxis], 0, 1) * np.einsum('ab,abc->ac', dp_dm[:, [2, 3]], dl_dw2)
    return np.mean(dl_dw1, axis=0), np.mean(dl_dw2, axis=0), np.mean(dl_du, axis=0)


def learn_cnn(X, Y, batch_size, lamb, iterations, learning_rate):
    """
    learn a cnn model for the given functions.
    :param X: the training and test input
    :param Y: the training and test labels
    :param batch_size: the batch size
    :param lamb: the regularization parameter
    :param iterations: the number of iterations
    :param learning_rate: the learning rate
    :return: a tuple of (models, training_loss, test_loss):
            models: a model for every function (a dictionary for the parameters)
            training_loss: the training loss at each iteration
            test loss: the test loss at each iteration
    """

    training_loss = {func_id: [] for func_id in range(num_of_functions)}
    test_loss = {func_id: [] for func_id in range(num_of_functions)}
    models = {func_id: {} for func_id in range(num_of_functions)}

    for func_id in range(num_of_functions):

        # initialize the model:
        models[func_id]['w1'] = np.random.randn(3)
        models[func_id]['w2'] = np.random.randn(3)
        models[func_id]['u'] = np.random.randn(4)
        # train the network:
        for _ in range(iterations):
            # draw a random batch:
            idx = np.random.choice(len(Y[func_id]['train']), batch_size)
            x, y = X['train'][idx, :], Y[func_id]['train'][idx]
            fwd = forward(models[func_id], x)
            loss = l2_loss(fwd['output'], y, lamb, models[func_id])
            dl_dw1, dl_dw2, dl_du = backprop(models[func_id], y, fwd, batch_size)

            # record the test loss before updating the model:
            test_fwd = forward(models[func_id], X['test'])
            iteration_test_loss = l2_loss(test_fwd['output'], Y[func_id]['test'], lamb, models[func_id])

            # update the model using the derivatives and record the loss:
            models[func_id]['w1'] -= learning_rate * dl_dw1
            models[func_id]['w2'] -= learning_rate * dl_dw2
            models[func_id]['u'] -= learning_rate * dl_du
            training_loss[func_id].append(loss)
            test_loss[func_id].append(iteration_test_loss)
    return models, training_loss, test_loss


def test_toy_cnn(X, Y):
    """creates,trains and plots the learning curves of CNN toy model"""
    batch_size1 = 100
    lamb1 = 0.5
    iterations1 = 500
    learning_rate1 = 0.001
    models, training_loss, test_loss = learn_cnn(X, Y, batch_size1, lamb1, iterations1, learning_rate1)
    for i in range(3):
        plt.plot(np.arange(len(training_loss[i])), training_loss[i], label='training loss', c=colors[1])
        plt.plot(np.arange(len(test_loss[i])), test_loss[i], label='test loss', c=colors[2])
        plt.legend(loc='upper left')
        plt.show()

def test_linear(X,Y):
    """creates,trains and plots the learning curves of linear toy model"""
    batch_size1 = 100
    lamb1 = 0.5
    iterations1 = 500
    learning_rate1 = 0.001
    models, training_loss, test_loss = learn_linear(X, Y, batch_size1, lamb1, iterations1, learning_rate1)
    for i in range(3):
        plt.plot(np.arange(len(training_loss[i])), training_loss[i], label='training loss',c=colors[1])
        plt.plot(np.arange(len(test_loss[i])), test_loss[i], label='test loss',c=colors[2])
        plt.legend(loc='upper left')
        plt.show()



def l2_loss(p, y, lamb, model):
    """l2 loss of two vectors describing prediction and true values of MNIST presictions"""
    l2_loss = (p - y) ** 2
    l2_reg = 0.5 * lamb * np.power(LA.norm(np.hstack((model['w1'], model['w2'])), 2), 2)
    return np.mean(l2_loss + l2_reg)


def plot_loss_acc(model_log, acc=True):
    """
    plotts the loss and accuracy curves fora single model
    :param model_log:
    :param acc:
    :return:
    """
    if acc:
        plt.subplot(2, 1, 1)
        plt.plot(model_log.history['acc'], c=colors[1])
        plt.plot(model_log.history['val_acc'], c=colors[2])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
    plt.subplot(2, 1, 2)
    plt.plot(model_log.history['loss'], c=colors[1])
    plt.plot(model_log.history['val_loss'], c=colors[2])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.show()


def linear_MNIST_classifier(X, Y, batch_size, epoch):
    """
    creates a linear classifier (with a single layer). training and testing of the model is done using the MNIST
    dataset.
    """
    X_train, y_train, X_test, y_test = [X['train'], Y['train'], X['test'], Y['test']]
    model = Sequential()
    model.add(Dense(10, activation='softmax'))
    learning_rate, decay, momentum, nesterov_bool = [0.002, 1e-6, 0.9, True]
    model_optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov_bool)
    model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])
    idx = np.random.choice(len(Y['train']), batch_size)
    x, y = X['train'][idx, :], y_train[idx, :]
    model_log = model.fit(x, y, batch_size=batch_size, epochs=epoch, validation_data=(X['test'], y_test))
    return model_log


def multi_layer_perceptron(X, Y, params, opt_type, loss_func, depth, activation, batch_size, dropout_const, epochs):
    """
    creates and trains a multi layer perceptron accoring to the multiple optional parametrs. trains and tetsts with
    the MNIST dataset.
    """
    X_train, y_train, X_test, y_test = [X['train'], Y['train'], X['test'], Y['test']]
    idx = np.random.choice(len(Y['train']), batch_size)
    x, y = X_train[idx, :], y_train[idx, :]
    model = Sequential()
    if opt_type == 'RMSprop':
        learning_rate = params[0]
        model_optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    elif opt_type == 'AdaGrad':
        learning_rate = params[0]
        model_optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    else:
        learning_rate, decay, momentum, nesterov_bool = params
        model_optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov_bool)
    if depth == 1:
        model.add(Dense(10, activation=activation, input_dim=X_train.shape[1]))
    else:
        model.add(Dense(64, activation=activation, input_dim=X_train.shape[1]))
    for i in range(depth - 1):
        model.add(Dropout(dropout_const))
        model.add(Dense(64, activation=activation))
        if i == depth - 2:
            model.add(Dropout(dropout_const))
            model.add(Dense(10, activation='softmax'))  # TODO: change these params
    model.compile(loss=loss_func, optimizer=model_optimizer, metrics=['accuracy'])
    model_log = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    return model_log


def plot_loss_acc_comparison(model_logs, labels, linewidth=None, title='', limits=None):
    """
    plotts 4 learning curves describing the convergence results for training and testing multiple models.the left ones
    desctibe the training curves an the right ones describe the testing curves.
    :param model_logs: all the models we wish to plot.
    :param labels: labels
    :param linewidth: the width of the curve
    :param title: plot title
    :param limits: graph limits
    """
    acc_test_labs = ['accuracy', 'loss']
    train_test_labs = ['train', 'test']
    if linewidth is None:
        linewidth = [1.5] * len(model_logs)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i, model_l in enumerate(model_logs):
        ax[0, 0].plot(model_l.history['acc'], c=colors[i], linewidth=linewidth[i])
        ax[0, 1].plot(model_l.history['val_acc'], c=colors[i], linewidth=linewidth[i])
    for j, model_l in enumerate(model_logs):
        ax[1, 0].plot(model_l.history['loss'], c=colors[j], linewidth=linewidth[j])
        ax[1, 1].plot(model_l.history['val_loss'], c=colors[j], linewidth=linewidth[j])
    for i in range(2):
        for j in range(2):
            ax[i, j].set_title('model ' + train_test_labs[j] + ' ' + acc_test_labs[i])
            ax[i, j].set_ylabel(acc_test_labs[i])
            ax[i, j].set_xlabel('epoch')
            ax[i, j].legend(labels, loc='upper right')
    if limits:
        for i in range(2):
            for j in range(2):
                ax[i, j].axis(limits[i])
    plt.tight_layout()
    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.62, top=0.64, wspace=0.3, hspace=0.37)
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.show()


def create_cool_MLP_graphs(X, Y):
    """
    this function plotts multiple graphs comparing between several parameters which can be changed during the
    training of a model (generic one).
    :param X: dataset
    :param Y: labels
    """
    acts = ['relu', 'tanh', 'softmax', 'softplus']
    opts = ['sgd', 'RMSprop', 'AdaGrad']
    losses = ['categorical_hinge', 'categorical_crossentropy']
    depths = [1, 3, 4]
    lrs = [5, 3, 1, 0.5, 0.05, 0.005, 0.0005]
    dropouts = [1, 0.75, 0.5, 0.25, 0.05, 0.005]
    learning_rate, decay, momentum, nesterov_bool = [0.01, 1e-6, 0.9, True]
    params = [learning_rate, decay, momentum, nesterov_bool]
    opt_type = 'sgd'
    loss_func = 'categorical_crossentropy'
    depth = 3
    activation = 'tanh'
    batch_size = 256
    dropout = 0.5
    epochs = 1500
    # acts
    model_logs = []
    for act in acts:
        model = multi_layer_perceptron(X, Y, params, opt_type, loss_func, depth, act, batch_size, dropout, epochs)
        model_logs.append(model)
    print_measures(model_logs, acts)
    plot_loss_acc_comparison(model_logs, acts)
    # opts
    model_logs = []
    for opt in opts:
        model = multi_layer_perceptron(X, Y, params, opt, loss_func, depth, activation, batch_size, dropout, epochs)
        model_logs.append(model)
    plot_loss_acc_comparison(model_logs, opts)
    print_measures(model_logs, opts)
    # losses
    model_logs = []
    for loss in losses:
        model = multi_layer_perceptron(X, Y, params, opt_type, loss, depth, activation, batch_size, dropout, epochs)
        model_logs.append(model)
    plot_loss_acc_comparison(model_logs, losses)
    print_measures(model_logs, losses)
    # depths
    model_logs = []
    times = []
    for d in depths:
        start = time.clock()
        model = multi_layer_perceptron(X, Y, params, opt_type, loss_func, d, activation, batch_size, dropout, epochs)
        model_logs.append(model)
        times.append(time.clock() - start)
    plot_loss_acc_comparison(model_logs, depths)
    print_measures(model_logs, depths)
    print('times: ')
    print(times)
    # lrs
    model_logs = []
    for lr1 in lrs:
        params[0] = lr1
        model = multi_layer_perceptron(X, Y, params, opt_type, loss_func, depth, activation, batch_size, dropout,
                                       epochs)
        model_logs.append(model)
    widths = np.array(lrs) * 2 + 1.5
    plot_loss_acc_comparison(model_logs, lrs, widths, 'Comparing MLP with different learning rates')
    print_measures(model_logs,lrs)
    # dropouts
    model_logs = []
    params[0] = learning_rate
    for dropo in dropouts:
        model = multi_layer_perceptron(X, Y, params, opt_type, loss_func, depth, activation, batch_size, dropo,
                                       epochs)
        model_logs.append(model)
    widths = np.array(dropouts) * 7 + 1.5
    plot_loss_acc_comparison(model_logs, dropouts, widths, 'Comparing MLP with different dropout values')
    print_measures(model_logs,dropouts)


def real_convnet(X, Y, params, opt_type, loss_func, activation, batch_size, dropout_const, epochs):
    """
    creates and trains convolution network
    :param X: dataset
    :param Y: labels
    :param params: parameters for the optimization
    :param opt_type: type of the optimizer we will use during the training
    :param loss_func: type of loss function
    :param activation: type of activation function
    :param batch_size: size of the batch
    :param dropout_const: dropout constant
    :param epochs: the number of iterations (repeatinf the while learning procedure)
    :return: model log
    """
    X_train, y_train, X_test, y_test = [X['train'], Y['train'], X['test'], Y['test']]
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        x_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    idx = np.random.choice(len(Y['train']), batch_size)
    x, y = x_train[idx, :, :, :], y_train[idx, :]
    model = Sequential()
    if opt_type == 'RMSprop':
        learning_rate = params[0]
        model_optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    elif opt_type == 'AdaGrad':
        learning_rate = params[0]
        model_optimizer = keras.optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    else:
        learning_rate, decay, momentum, nesterov_bool = params
        model_optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov_bool)
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=input_shape, strides=(1, 1)))
    model.add(Conv2D(20, (3, 3), activation=activation, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_const))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    if opt_type == 'adam':
        model.compile(loss=loss_func, optimizer=model_optimizer, metrics=['accuracy'])
    else:
        model.compile(loss=loss_func, optimizer=opt_type, metrics=['accuracy'])
    model_log = model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    return model_log


def compare_linaer2MLP2conv(X, Y, model_params):
    """
    compares the performance of three models: linear, MLP and convolutional
    :param X: dataset
    :param Y: labels
    :param model_params: parameters for the optimization
    """
    category_num = 10
    Y['train'] = keras.utils.to_categorical(y_train, category_num)
    Y['test'] = keras.utils.to_categorical(y_test, category_num)
    learning_rate, decay, momentum, nesterov_bool = [0.01, 1e-6, 0.9, True]
    opt_params = [learning_rate, decay, momentum, nesterov_bool]
    opt_type = 'adam'
    loss_func = 'categorical_crossentropy'
    activation = 'relu'
    batch_size = 256
    dropout = 0.05
    epochs = 50
    depth = 3
    times = []
    start = time.clock()
    conv_model_log = real_convnet(X, Y, opt_params, opt_type, loss_func, activation, batch_size, dropout, epochs)
    times.append(time.clock() - start)
    X = {'train': x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2])),
         'test': x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))}
    start = time.clock()
    linear_model_log = linear_MNIST_classifier(X, Y, 60000, epochs)
    times.append(time.clock() - start)
    start = time.clock()
    MLP_model_log = multi_layer_perceptron(X, Y, opt_params, opt_type, loss_func, depth, activation, batch_size,
                                           dropout, epochs)
    times.append(time.clock() - start)
    logs = [linear_model_log, MLP_model_log, conv_model_log]
    labs = ['linear', 'MLP', 'convolution']
    limits = [[0, 40, 0, 1], [0, 40, 0, 14]]
    print('times for convnet,linear and MLP')
    print(times)
    plot_loss_acc_comparison(logs, labs, limits=limits)
    print_measures(logs, labs)


def create_autoencoder():
    """
    creates and trains autoencder
    :return:
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='elu', kernel_initializer='random_normal', bias_initializer='zeros', use_bias=True)(
        input_img)
    encoded = Dense(64, activation='elu')(encoded)
    encoded = Dense(2, activation='elu', name='2Dencoded')(encoded)

    decoded = Dense(64, activation='elu')(encoded)
    decoded = Dense(128, activation='elu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)
    learning_rate = 0.5
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    model_log = autoencoder.fit(x_train, x_train, epochs=1500, batch_size=256, shuffle=True,
                                validation_data=(x_test, x_test))
    print('train accuracy: ' + str(model_log.history['loss'][-1]))
    print('test accuracy: ' + str(model_log.history['val_loss'][-1]))
    plot_loss_acc(model_log, False)
    compare_autoencoder2PCA(autoencoder, '2Dencoded', [x_train, x_test, y_train, y_test], learning_rate)


def compare_autoencoder2PCA(autoencoder_model, desired_layer_name, data, learning_rate):
    """
    plots two graphs of the embedded MNIST dataset. one for each algorithm
    :param autoencoder_model: model of trained autoencoder
    :param desired_layer_name: the name of the layer with 2 units
    :param data: test data
    """
    intermediate_layer_model = Model(inputs=autoencoder_model.input,
                                     outputs=autoencoder_model.get_layer(desired_layer_name).output)
    auto_data = intermediate_layer_model.predict(data[1])[:5000, :]
    fig, ax = plt.subplots(nrows=2, ncols=1)
    pca = PCA(n_components=2)
    pca.fit(data[1])
    x = pca.transform(data[1])[:5000, :]
    c = data[3]
    ys = data[3][:5000]
    for i in range(10):
        relevant_x = x[np.where(ys == i)]
        relevant_auto = auto_data[np.where(ys == i)]
        ax[1].scatter(relevant_x[:, 0], relevant_x[:, 1], c=colors[i], s=3.5, label=str(i))
        ax[0].scatter(relevant_auto[:, 0], relevant_auto[:, 1], c=colors[i], s=3.5, label=str(i))
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].set_title('Compressed Data Using Autoencoder function')
    ax[0].set_ylabel('Autoencoder2')
    ax[0].set_xlabel('Autoencoder1')
    ax[1].set_title('Compressed Data Using PCA')
    ax[1].set_ylabel('PC2')
    ax[1].set_xlabel('PC1')
    plt.show()


def print_measures(model_logs, themes):
    """ prints learning values of the last iteration of a trained model"""
    for i, theme in enumerate(themes):
        print('###### Measures for ' + str(theme) + '######')
        print('train accuracy of the last iteration: ' + str(model_logs[i].history['acc'][-1]))
        print('test accuracy of the last iteration: ' + str(model_logs[i].history['val_acc'][-1]))
        print('train loss of the last iteration: ' + str(model_logs[i].history['loss'][-1]))
        print('test loss of the last iteration: ' + str(model_logs[i].history['val_loss'][-1]))


if __name__ == '__main__':
    # generate the training and test data, adding some noise:
    X = dict(train=5 * (np.random.random((1000, input_size)) - .5),
             test=5 * (np.random.random((200, input_size)) - .5))
    Y = {i: {
        'train': y[i](X['train']) * (1 + np.random.randn(X['train'].shape[0]) * .01),
        'test': y[i](X['test']) * (1 + np.random.randn(X['test'].shape[0]) * .01)} for i in range(len(y))}
    test_linear(X,Y)
    # test_toy_cnn(X,Y)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    category_num = 10
    Y['train'] = keras.utils.to_categorical(y_train, category_num)
    Y['test'] = keras.utils.to_categorical(y_test, category_num)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # linear model
    start = time.clock()
    model_log1 = linear_MNIST_classifier(X, Y, 60000, 1500)
    print('time taken for the linear model: '+str(time.clock() - start))
    print_measures(model_log1,['linear MNIST'])
    plot_loss_acc(model_log1)

    # multi-layer perceptron
    create_cool_MLP_graphs(X, Y)

    # learning_rate, decay, momentum, nesterov_bool = [0.01, 1e-6, 0.9, True]
    params = [learning_rate, decay, momentum, nesterov_bool]
    opt_type = 'sgd'
    loss_func = 'categorical_hinge'
    depth = 3
    activation = 'tanh'
    batch_size = 30000
    dropout = 0.5
    epochs = 150
    multi_layer_perceptron(X, Y, params, opt_type, loss_func, depth, activation, batch_size, dropout, epochs)

    # convnet
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = {'train': x_train, 'test': x_test}
    Y = {'train': y_train, 'test': y_test}
    category_num = 10
    Y['train'] = keras.utils.to_categorical(y_train, category_num)
    Y['test'] = keras.utils.to_categorical(y_test, category_num)
    learning_rate, decay, momentum, nesterov_bool = [0.01, 1e-6, 0.9, True]
    params = [learning_rate, decay, momentum, nesterov_bool]
    model_params = ['sgd', 'categorical_crossentropy', 'relu', 60000, 0.25, 100]
    opt_type, loss_func, activation, batch_size, dropout, epochs = model_params

    # check acc and loss of convolution
    model_log = real_convnet(X, Y, params, opt_type, loss_func, activation, batch_size, dropout, epochs)
    plot_loss_acc(model_log)
    print_measures(model_log,['convnet'])

    # compare between models
    compare_linaer2MLP2conv(X, Y, model_params)

    # Hyper parameter
    variating learning rate
    create_cool_MLP_graphs(X, Y)

    # Autoencoder
    create_autoencoder()
