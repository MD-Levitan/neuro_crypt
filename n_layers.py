import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def read_x_from_file(file):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(3))
        while block:
            x = block[0] + block[1] * 256 + (block[2] >> 4) * 65536  # transform data to number
            data.append([(x >> i) & 0x01 for i in range(0, 20)])     # transform number as 20 input
            block = bytes(f.read(3))
    return data


def read_y_from_file(file):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(1))
        while block:
            data_x = [0, 0]
            data_x[block[0]] = 1
            data.append(data_x)
            block = bytes(f.read(1))
    return data


input_data = read_x_from_file("legenda/X.bin")
output_data1 = read_y_from_file("legenda/Y1.bin")

# Network Parameters
number_layers = 5
n_hidden = 20  # n-st layer number of features
n_hidden = [n_hidden] * number_layers
n_input = 20  # input layer
n_classes = 2
learning_rate = 0.001
training_epochs = 100
display_step = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(x, weights, biases, num):
    # Hidden layer with SIGMOID activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer = layer_1
    for i in range(2, num + 1):
        # Hidden layer with SIGMOID activation
        layer_new = tf.add(tf.matmul(layer, weights['h%d' % i]), biases['b%d' % i])
        layer_new = tf.nn.sigmoid(layer_new)
        layer = layer_new

    # Output layer with linear activation
    out_layer = tf.matmul(layer, weights['out']) + biases['out']
    return out_layer


def generate_weigths(n_input, n_hidden, n_classes, num):
    weights = {}
    for i in range(1, num + 1):
        weights['h%d' % i] = tf.Variable(tf.random_normal([n_input if i == 1 else n_hidden[i - 2], n_hidden[i - 1]]))
    weights['out'] = tf.Variable(tf.random_normal([n_hidden[-1], n_classes]))
    return weights


def generate_biases(n_input, n_hidden, n_classes, num):
    biases = {}
    for i in range(1, num + 1):
        biases['b%d' % i] = tf.Variable(tf.random_normal([n_hidden[i - 1]]))
        biases['out'] = tf.Variable(tf.random_normal([n_classes]))
    return biases


# cost = tf.losses.mean_squared_error(y, multilayer_perceptron(x, weights, biases))
prediction = multilayer_perceptron(x,
                                   generate_weigths(n_input, n_hidden, n_classes, number_layers),
                                   generate_biases(n_input, n_hidden, n_classes, number_layers), number_layers)
cost = tf.losses.sigmoid_cross_entropy(y, prediction)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# define our loss function as the cross entropy loss
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# functions that allow us to gauge accuracy of our model
correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    train_dataset = np.array(input_data[:9 * len(input_data) // 10])
    train_values = np.array(output_data1[: 9 * len(input_data) // 10])

    test_dataset = np.array(input_data[-len(input_data) // 10:])
    test_values = np.array(output_data1[-len(input_data) // 10:])

    for epoch in range(training_epochs):
        # acc, loss = sess.run([optimizer, cost],
        #                 feed_dict={x: train_dataset,
        #                            y: train_values})
        # if epoch % 100 == 0:
        #     print("Epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, loss))

        optimizer.run(feed_dict={x: train_dataset, y: train_values})
        # every 100 iterations, print out the accuracy
        if epoch % display_step == 0:
            acc = accuracy.eval(feed_dict={x: train_dataset, y: train_values})
            loss = cross_entropy_loss.eval(feed_dict={x: train_dataset, y: train_values})
            print("Epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, loss))

    print("Optimization Finished!")
    #
    # predicted_class = tf.greater(prediction, 0.5)
    # correct = tf.equal(predicted_class, tf.equal(y, True))
    # accuracy = tf.reduce_mean(tf.cast(correct, 'uint8'))
    #
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_dataset,
    #                                                          y: test_values}))

    # print("Accuracy:", accuracy.eval({x: test_dataset, y: test_values}))

    # test_error = tf.nn.l2_loss(prediction - y, name="squared_error_test_cost") / test_values.shape[0]
    # print("Test Error:", test_error.eval({x: test_dataset, y: test_values}))

    acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
    print("testing accuracy: {}".format(acc))

    print(prediction.eval(({x: test_dataset})))
