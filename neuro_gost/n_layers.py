import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def read_from_file(file, num_bytes, format_data=lambda x: int.from_bytes(x, byteorder='little')):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(num_bytes))
        while block:
            data.append(format_data(block))
            block = bytes(f.read(num_bytes))
    return data


def to_float(block, bytes=4):
    return [int.from_bytes(block[:4], byteorder='little') / pow(2, 8 * bytes)]


def get_test_values():
    input_data = read_from_file("GOST_generator/out_x.bin", 8, to_float)
    output_data = read_from_file("GOST_generator/out_y.bin", 8, to_float)

    return np.array(input_data), np.array(output_data)


# Deprecated version
def __multilayer_perceptron(x, weights, biases, num, activaion_fun=tf.nn.sigmoid):
    """
    Function to create layers.
    :param x: input data
    :param weights: vector of weights
    :param biases:  vector of biases
    :param num: number of layers(>=2)
    :return: output layer
    """

    # Hidden layer with activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activaion_fun(layer_1)
    layer = layer_1
    for i in range(2, num + 1):
        # Hidden layer with activation
        layer_new = tf.add(tf.matmul(layer, weights['h%d' % i]), biases['b%d' % i])
        layer_new = activaion_fun(layer_new)
        layer = layer_new

    # Output layer with linear activation
    out_layer = tf.matmul(layer, weights['out']) + biases['out']
    return out_layer


def __generate_weigths(n_input, n_hidden, n_classes, num):
    weights = {}
    for i in range(1, num + 1):
        weights['h%d' % i] = tf.Variable(tf.random.normal([n_input if i == 1 else n_hidden[i - 2], n_hidden[i-1]]))
    weights['out'] = tf.Variable(tf.random.normal([n_hidden[-1], n_classes]))
    return weights


def __generate_biases(n_input, n_hidden, n_classes, num):
    biases = {}
    for i in range(1, num + 1):
        biases['b%d' % i] = tf.Variable(tf.random.normal([n_hidden[i - 1]]))
        biases['out'] = tf.Variable(tf.random.normal([n_classes]))
    return biases


def multilayer_perceptron(x, hidden, num_classes, num, activation_fun=tf.nn.sigmoid,
                          first_activation=False):
    """
    Function to create layers.
    :param x: input data
    :param hidden: vector of hidden layer's sizes
    :param num_classes:  size of output layer
    :param num: number of layers(>=2)
    :param activation_fun: activation function
    :param first_activation: use or not activation function on first hidden layer
    :return: output layer
    """

    # Hidden layer with activation
    layer = tf.layers.dense(x, hidden[0],
                            activation=activation_fun if first_activation else None,
                            kernel_initializer=tf.initializers.ones(),
                            bias_initializer=tf.initializers.random_normal())

    # Hidden layer with activation
    for i in range(1, num):
        layer_new = tf.layers.dense(layer, hidden[i],
                                    activation=activation_fun,
                                    kernel_initializer=tf.initializers.ones(),
                                    bias_initializer=tf.initializers.random_normal())
        layer = layer_new

    # Output layer with linear activation
    out_layer = tf.layers.dense(layer, num_classes, activation=None)
    return out_layer


def init_network(input_data, output_data, n_input, n_hidden, n_classes, number_layers, learning_rate=0.001,
                 training_epochs=100, display_step=10, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer):

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    n_hidden = [n_hidden] * number_layers
    # n_hidden = [int(sum([binomial(n_input, i)
    # for i in range(1, level + 1)])) for level in range(1, number_layers + 1)]
    # n_hidden = [sum([int(binomial(n_input, level))
    # for level in range(1, number_layers + 1)])] + [n_hidden] * (number_layers - 1)

    prediction = multilayer_perceptron(x, n_hidden, n_classes, activaion_fun=activation)

    # cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)
    cost = tf.reduce_mean(tf.square(prediction - y))

    optimizer = optimizer(learning_rate=learning_rate).minimize(cost)
    loss = tf.reduce_mean(cost, name='loss')

    init = tf.global_variables_initializer()

    # StackOverflow
    correct_prediction = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)

        train_dataset = np.array(input_data[:9 * len(input_data) // 10])
        train_values = np.array(output_data[: 9 * len(input_data) // 10])

        test_dataset = np.array(input_data[-len(input_data) // 10:])
        test_values = np.array(output_data[-len(input_data) // 10:])

        for epoch in range(0, training_epochs):
            sess.run(optimizer, feed_dict={
                x: train_dataset,
                y: train_values
            })

            if not epoch % display_step:
                acc = accuracy.eval(feed_dict={x: train_dataset, y: train_values})
                los = loss.eval(feed_dict={x: train_dataset, y: train_values})
                print("Epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, los))

        print("Optimization Finished!")

        train_values1 = sess.run(prediction, feed_dict={
            x: train_dataset,
        })

        plt.plot(train_dataset, train_values, "bo",
                 train_dataset, train_values1, "ro")
        plt.show()

        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})
        print("Testing accuracy: {}".format(acc))
        print("Testing loss: {}".format(los))

        return acc, los


if __name__ == "__main__":
    input_data, output_data = get_test_values()

    # Network Parameters
    n_input = 1
    n_hidden = 4
    n_classes = 1
    number_layers = 2

    x, y = init_network(input_data, output_data, n_input, n_hidden, n_classes, number_layers,
                        activation=tf.nn.relu, training_epochs=10000, display_step=1000)

