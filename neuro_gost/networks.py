import numpy as np
import tensorflow as tf

import utils

losses_default = {
    "sigmoid_cross_entropy": tf.losses.sigmoid_cross_entropy,
    "binary_cross_entropy": tf.keras.losses.binary_crossentropy,
    "square_difference": lambda y, x: tf.reduce_mean(tf.square(x - y))
}


def init_one_layer_network(input_data: np.array, output_data: np.array, n_input: int, n_classes: int,
                           training_epochs: int = 100, display_step: int = 10, optimizer=tf.train.AdamOptimizer,
                           loss_fun: str = "sigmoid_cross_entropy", verbose: bool = True):
    """
    Create one-layer neural network.
    :param input_data: np.array with input values
    :param output_data: np.array with output values
    :param n_input: neuron's number on first layer
    :param n_classes: number of output classes
    :param training_epochs: number of training epochs
    :param display_step:
    :param optimizer:
    :param loss_fun:
    :param verbose:
    :return: Accuracy, Loss, Predict Array, Real Array
    """
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    out = tf.layers.dense(x, n_classes, activation=tf.nn.sigmoid)

    hamming_distance = tf.math.count_nonzero(tf.round(out) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)

    loss = losses_default[loss_fun](y, out)
    train_step = optimizer(learning_rate=0.002).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_dataset = np.array(input_data[:9 * len(input_data) // 10])
        train_values = np.array(output_data[: 9 * len(input_data) // 10])

        test_dataset = np.array(input_data[-len(input_data) // 10:])
        test_values = np.array(output_data[-len(input_data) // 10:])
        for i in range(training_epochs):
            if i % display_step == 0:
                feed = {x: train_dataset, y: train_values}
                result = sess.run([loss, accuracy], feed_dict=feed)
                print('Accuracy at step %s: %f - loss: %f' % (i, result[1], result[0]))
            else:
                feed = {x: train_dataset, y: train_values}
            sess.run(train_step, feed_dict=feed)

        predict_values = tf.round(out).eval(feed_dict={x: test_dataset})

        if verbose:
            print(list(map(lambda x: hex(utils.to_int(x)), predict_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_dataset)))

        accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        if verbose:
            print("testing accuracy: {}".format(acc))
            print("testing Loss: {}".format(los))

        sess.close()

    return acc, los, predict_values, test_values


def multilayer_perceptron(x, hidden, num_classes, number_layers, activation_fun=tf.nn.sigmoid,
                          first_activation=False, drop_out=False):
    """
    Function to create layers.
    :param x: input data
    :param hidden: vector of hidden layer's sizes
    :param num_classes:  size of output layer
    :param num: number of layers(>=2)
    :param activation_fun: activation function
    :param first_activation: use or not activation function on first hidden layer
    :param drop_out: using of drop out layer
    :return: output layer
    """

    keep_prob = tf.placeholder(tf.float32)

    # Hidden layer with activation
    layer = tf.layers.dense(x, hidden[0],
                            activation=activation_fun if first_activation else None,
                            kernel_initializer=tf.initializers.ones(),
                            bias_initializer=tf.initializers.ones())

    # Hidden layer with activation
    for i in range(1, number_layers):
        layer_new = tf.layers.dense(layer, hidden[i],
                                    activation=activation_fun)

        if drop_out:
            layer = tf.layers.dropout(layer, keep_prob)
        else:
            layer = layer_new

    # Output layer with linear activation
    out_layer = tf.layers.dense(layer, num_classes, activation=None)
    return out_layer


def recurrent_perceptron(x, hidden, num_classes, number_layers, activation_fun=tf.nn.sigmoid,
                         first_activation=False, drop_out=False):
    """
    Function to create layers.
    :param x: input data
    :param hidden: vector of hidden layer's sizes
    :param num_classes:  size of output layer
    :param num: number of layers(>=2)
    :param activation_fun: activation function
    :param first_activation: use or not activation function on first hidden layer
    :param drop_out: using of drop out layer
    :return: output layer
    """

    keep_prob = tf.placeholder(tf.float32)

    # Hidden layer with activation
    layer = tf.layers.dense(x, hidden[0],
                            activation=activation_fun if first_activation else None,
                            kernel_initializer=tf.initializers.ones(),
                            bias_initializer=tf.initializers.ones())

    # Hidden layer with activation
    for i in range(1, number_layers):
        layer_new = tf.layers.dense(tf.concat([layer, x], 1), hidden[i],
                                    activation=activation_fun)

        if drop_out:
            layer = tf.layers.dropout(layer, keep_prob)
        else:
            layer = layer_new

    # Output layer with linear activation
    out_layer = tf.layers.dense(tf.concat([layer, x], 1), num_classes, activation=None)
    return out_layer


def init_multilayer_network(input_data: np.array, output_data: np.array, n_input: int, n_hidden: list,
                            n_classes: int, number_layers: list, learning_rate=0.001, training_epochs=100,
                            display_step=10, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer,
                            loss_fun: str = "square_difference", verbose: bool = True):
    """
    Create multilayer neural network.
    :param input_data: np.array with input values
    :param output_data: np.array with output values
    :param n_input: neuron's number on first layer
    :param n_hidden: list with neuron's number on hidden layers
    :param n_classes: number of output classes
    :param number_layers: number of layers
    :param training_epochs: number of training epochs
    :param display_step:
    :param optimizer:
    :param loss_fun:
    :param verbose:
    :return: Accuracy, Loss, Predict Array, Real Array
    """
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # n_hidden = [n_hidden * i for i in range(1, number_layers + 1)]
    # n_hidden = [n_hidden] * number_layers
    # n_hidden = [int(sum([binomial(n_input, i)
    # for i in range(1, level + 1)])) for level in range(1, number_layers + 1)]
    # n_hidden = [sum([int(binomial(n_input, level))
    # for level in range(1, number_layers + 1)])] + [n_hidden] * (number_layers - 1)

    prediction = multilayer_perceptron(x, n_hidden, n_classes, number_layers,
                                       activation_fun=activation, first_activation=True)

    hamming_distance = tf.math.count_nonzero(tf.round(prediction) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)
    loss = losses_default[loss_fun](y, prediction)

    optimizer = optimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        train_dataset = np.array(input_data[:9 * len(input_data) // 10])
        train_values = np.array(output_data[: 9 * len(input_data) // 10])

        test_dataset = np.array(input_data[-len(input_data) // 10:])
        test_values = np.array(output_data[-len(input_data) // 10:])

        for i in range(training_epochs):
            if i % display_step == 0:
                feed = {x: train_dataset, y: train_values}
                result = sess.run([loss, accuracy], feed_dict=feed)
                print('Accuracy at step %s: %f - loss: %f\n' % (i, result[1], result[0]))
            else:
                feed = {x: train_dataset, y: train_values}
            sess.run(optimizer, feed_dict=feed)

        predict_values = sess.run(prediction, feed_dict={
            x: test_dataset,
        })

        if verbose:
            print(list(map(lambda x: hex(utils.to_int(x)), predict_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_dataset)))

        accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        if verbose:
            print("testing accuracy: {}".format(acc))
            print("testing Loss: {}".format(los))

        sess.close()

    return acc, los,\
           list(map(lambda x: utils.to_int(x), predict_values)),\
           list(map(lambda x: utils.to_int(x), test_values))


def init_recurrent_network(input_data: np.array, output_data: np.array, n_input: int, n_hidden: list,
                           n_classes: int, number_layers: list, learning_rate=0.001, training_epochs=100,
                           display_step=10, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer,
                           loss_fun: str = "square_difference", verbose: bool = True):
    """
    Create recurrent neural network.
    :param input_data: np.array with input values
    :param output_data: np.array with output values
    :param n_input: neuron's number on first layer
    :param n_hidden: list with neuron's number on hidden layers
    :param n_classes: number of output classes
    :param number_layers: number of layers
    :param training_epochs: number of training epochs
    :param display_step:
    :param optimizer:
    :param loss_fun:
    :param verbose:
    :return: Accuracy, Loss, Predict Array, Real Array
    """

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # n_hidden = [n_hidden * i for i in range(1, number_layers + 1)]
    # n_hidden = [n_hidden] * number_layers
    # n_hidden = [int(sum([binomial(n_input, i)
    # for i in range(1, level + 1)])) for level in range(1, number_layers + 1)]
    # n_hidden = [sum([int(binomial(n_input, level))
    # for level in range(1, number_layers + 1)])] + [n_hidden] * (number_layers - 1)

    prediction = recurrent_perceptron(x, n_hidden, n_classes, number_layers,
                                      activation_fun=activation, first_activation=True)

    hamming_distance = tf.math.count_nonzero(tf.round(prediction) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)
    loss = losses_default[loss_fun](y, prediction)

    optimizer = optimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        train_dataset = np.array(input_data[:9 * len(input_data) // 10])
        train_values = np.array(output_data[: 9 * len(input_data) // 10])

        test_dataset = np.array(input_data[-len(input_data) // 10:])
        test_values = np.array(output_data[-len(input_data) // 10:])

        for i in range(training_epochs):
            if i % display_step == 0:
                feed = {x: train_dataset, y: train_values}
                result = sess.run([loss, accuracy], feed_dict=feed)
                print('Accuracy at step %s: %f - loss: %f\n' % (i, result[1], result[0]))
            else:
                feed = {x: train_dataset, y: train_values}
            sess.run(optimizer, feed_dict=feed)

        predict_values = sess.run(prediction, feed_dict={
            x: test_dataset,
        })

        if verbose:
            print(list(map(lambda x: hex(utils.to_int(x)), predict_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_dataset)))

        accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        if verbose:
            print("testing accuracy: {}".format(acc))
            print("testing Loss: {}".format(los))

        sess.close()

    return acc, los, \
           list(map(lambda x: utils.to_int(x), predict_values)), \
           list(map(lambda x: utils.to_int(x), test_values))