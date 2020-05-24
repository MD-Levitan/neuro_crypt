import numpy as np
import tensorflow as tf

import utils


@tf.RegisterGradient("Hamming")
def hamming_loss_fn(y_true, y_pred) -> tf.Tensor:
    threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)

    # make sure [0, 0, 0] doesn't become [1, 1, 1]
    # Use abs(x) > eps, instead of x != 0 to check for zero
    # y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    #
    # y_true = tf.cast(y_true, tf.int32)
    # y_pred = tf.cast(y_pred, tf.int32)

    nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)

    nonzero = tf.square(nonzero)
    print(nonzero)

    return nonzero


def loss_fn_(y_true, y_pred) -> tf.Tensor:
    diff = tf.abs(y_true - y_pred)
    mask = tf.greater(diff, 0.25)
    print(y_true)
    print(y_pred)
    var = tf.where(mask, diff, diff * 0)

    return tf.reduce_sum(var)


def loss_fn(y_true, y_pred) -> tf.Tensor:
    return tf.reduce_mean(tf.square(y_true - y_pred))
    # return loss_fn_(y_true, y_pred)

losses_default = {
    "sigmoid_cross_entropy": tf.compat.v1.losses.sigmoid_cross_entropy,
    "binary_cross_entropy": tf.keras.losses.binary_crossentropy,
    "square_difference": lambda y, x: tf.reduce_mean(tf.square(x - y)),
}


def init_one_layer_network(input_data: np.array, output_data: np.array, n_input: int, n_classes: int,
                           training_epochs: int = 100, display_step: int = 10, optimizer=tf.compat.v1.train.AdamOptimizer,
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
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

    out = tf.layers.dense(x, n_classes, activation=tf.nn.sigmoid)

    hamming_distance = tf.math.count_nonzero(tf.round(out) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)

    loss = losses_default[loss_fun](y, out)
    train_step = optimizer(learning_rate=0.002).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
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

    keep_prob = tf.compat.v1.placeholder(tf.float32)

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

    keep_prob = tf.compat.v1.placeholder(tf.float32)

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
                            display_step=100, activation=tf.nn.sigmoid, optimizer=tf.compat.v1.train.AdamOptimizer,
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
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

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
    loss = loss_fn(y, prediction) #tf.reduce_mean(tf.square(tf.subtract(y, prediction)))

    optimizer = optimizer(learning_rate=learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
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
                pass
            sess.run(optimizer, feed_dict=feed)

        predict_values = sess.run(prediction, feed_dict={
            x: test_dataset,
        })

        predict_values1 = sess.run(prediction, feed_dict={
            x: train_dataset,
        })

        if verbose:
            print(list(map(lambda x: hex(utils.to_int(x)), predict_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), test_dataset)))

            print(list(map(lambda x: hex(utils.to_int(x)), predict_values1)))
            print(list(map(lambda x: hex(utils.to_int(x)), train_values)))
            print(list(map(lambda x: hex(utils.to_int(x)), train_dataset)))

        # accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))

        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        acc1 = accuracy.eval(feed_dict={x: train_dataset, y: train_values})
        los1 = loss.eval(feed_dict={x: train_dataset, y: train_values})

        if verbose:
            print("testing accuracy: {}".format(acc))
            print("testing Loss: {}".format(los))

            print("testing accuracy: {}".format(acc1))
            print("testing Loss: {}".format(los1))

        sess.close()

    return acc, los,\
           list(map(lambda x: utils.to_int(x), predict_values)),\
           list(map(lambda x: utils.to_int(x), test_values))


def init_recurrent_network(input_data: np.array, output_data: np.array, n_input: int, n_hidden: list,
                           n_classes: int, number_layers: list, learning_rate=0.001, training_epochs=100,
                           display_step=10, activation=tf.nn.sigmoid, optimizer=tf.compat.v1.train.AdamOptimizer,
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

    x = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

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

    with tf.compat.v1.Session() as sess:
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