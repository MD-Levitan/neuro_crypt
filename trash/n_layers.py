import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from test_generator import DataGenerator
from itertools import chain, combinations
from functools import reduce
# from stack_test import init_one_layer_network
from sympy import binomial

optimizers = {"Adam": tf.train.AdamOptimizer, "ProximalAdarad": tf.train.ProximalAdagradOptimizer,
              "Proximal": tf.train.ProximalGradientDescentOptimizer, "Adagard": tf.train.AdagradOptimizer,
              "Gradient": tf.train.GradientDescentOptimizer}


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


def multilayer_perceptron(x, weights, biases, num, activaion_fun=tf.nn.sigmoid):
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


def generate_weigths(n_input, n_hidden, n_classes, num):
    weights = {}
    for i in range(1, num + 1):
        weights['h%d' % i] = tf.Variable(tf.random_normal([n_input if i == 1 else n_hidden[i - 2], n_hidden[i-1]]))
    weights['out'] = tf.Variable(tf.random_normal([n_hidden[-1], n_classes]))
    return weights


def generate_biases(n_input, n_hidden, n_classes, num):
    biases = {}
    for i in range(1, num + 1):
        biases['b%d' % i] = tf.Variable(tf.random.normal([n_hidden[i - 1]]))
        biases['out'] = tf.Variable(tf.random.normal([n_classes]))
    return biases

def init_network(input_data, output_data, n_input, n_hidden, n_classes, number_layers, learning_rate=0.001,
                 training_epochs=100, display_step=10, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer):
    # def combinate(x, vars):
    #     def all_combinations(any_list):
    #         return chain.from_iterable(
    #             combinations(any_list, i + 1)
    #             for i in range(1, len(any_list)))
    #
    #     combinations_ = all_combinations(list(range(0, vars)))
    #     extended_x = []
    #     for part_x in x:
    #         updated_x = part_x
    #         for comb in combinations_:
    #             updated_x.append(reduce(lambda x, y: x & y, [part_x[i] for i in comb]))
    #         extended_x.append(updated_x)
    #     return extended_x
    #
    # input_data = combinate(input_data, n_input)
    # output_data = output_data
    #
    # n_input = pow(2, n_input) - 1
    # n_hidden = n_input

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # n_hidden = [n_hidden] * number_layers

    # n_hidden = [int(sum([binomial(n_input, i) for i in range(1, level + 1)])) for level in range(1, number_layers + 1)]

    n_hidden = [sum([int(binomial(n_input, level)) for level in range(1, number_layers + 1)])] + [n_hidden] * (number_layers - 1)

    print(n_hidden)


    # cost = tf.losses.mean_squared_error(y, multilayer_perceptron(x, weights, biases))
    prediction = multilayer_perceptron(x,
                                       generate_weigths(n_input, n_hidden, n_classes, number_layers),
                                       generate_biases(n_input, n_hidden, n_classes, number_layers), number_layers,
                                       activaion_fun=activation)

    # cost = tf.losses.sigmoid_cross_entropy(y, prediction)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)



    optimizer = optimizer(learning_rate=learning_rate).minimize(cost)
    loss = tf.reduce_mean(cost, name='loss')

    # define our loss function as the cross entropy loss
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    # functions that allow us to gauge accuracy of our model
    # correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # StackOverflow
    pred = tf.cast((prediction > 0.5), tf.float32)
    correct_prediction = tf.equal(pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)

        train_dataset = np.array(input_data[:9 * len(input_data) // 10])
        train_values = np.array(output_data[: 9 * len(input_data) // 10])

        test_dataset = np.array(input_data[-len(input_data) // 10:])
        test_values = np.array(output_data[-len(input_data) // 10:])

        for epoch in range(0, training_epochs):
            # acc, loss = sess.run([optimizer, cost],
            #                 feed_dict={x: train_dataset,
            #                                y: train_values})
            # if epoch % 100 == 0:
            #     print("Epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, loss))

            optimizer.run(feed_dict={x: train_dataset, y: train_values})
            if not epoch % display_step:
                acc = accuracy.eval(feed_dict={x: train_dataset, y: train_values})
                los = loss.eval(feed_dict={x: train_dataset, y: train_values})
                print("Epoch: {}, accuracy: {}, loss: {}".format(epoch, acc, los))

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

        # acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        # print("testing accuracy: {}".format(acc))
        #
        # train_values1 = sess.run(prediction, feed_dict={
        #     x: test_dataset,
        # })
        # print(train_values1)
        # print(test_values)
        # print(prediction.eval(({x: test_dataset})))

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})
        print("testing accuracy: {}".format(acc))
        print("testing loss: {}".format(los))

        result = tf.argmax(prediction, 1).eval({x: test_dataset, y: test_values})
        print(list(result)[0:50])
        print(list(map(lambda x: 1 if x[1] else 0, test_values))[0:50])

        #print(correct_prediction.eval(({x: test_dataset})))
        return acc, los

# if __name__ == "__main__":
#     #input_data = read_x_from_file("legenda/X.bin")
#     #output_data = read_y_from_file("legenda/Y1.bin")
#
#     # Network Parameters
#     n_input = 6  # input layer
#     n_hidden = n_input  # n-st layer number of features
#     n_classes = 2
#     number_layers = 20
#
#     optimizers_stat_4_acc = [0] * len(optimizers)
#     optimizers_stat_4_loss = [0] * len(optimizers)
#
#     optimizers_stat_2_acc = [0] * len(optimizers)
#     optimizers_stat_2_loss = [0] * len(optimizers)
#
#     repeats = 10
#     for f in range(0, repeats):
#         print("FF: {}".format(f))
#         input_data, output_data = DataGenerator(n_input=n_input).generate_data(input_size=pow(2, 10))
#
#         # init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10)
#         i = 0
#         for (_, val) in optimizers.items():
#             x, y = init_network(input_data, output_data, n_input, n_hidden, n_classes, 4, activation=tf.nn.relu,
#                                 training_epochs=1000, display_step=1000, optimizer=val)
#             optimizers_stat_4_acc[i] += x
#             optimizers_stat_4_loss[i] += y
#             x, y = init_network(input_data, output_data, n_input, n_hidden, n_classes, 2, activation=tf.nn.relu,
#                          training_epochs=1000, display_step=1000, optimizer=val)
#             optimizers_stat_2_acc[i] += x
#             optimizers_stat_2_loss[i] += y
#             i += 1
#
#     i = 0
#     for (key, val) in optimizers.items():
#         print("Optimizer: {}".format(key))
#         print("Accurancy for 4: {}".format(optimizers_stat_4_acc[i] / repeats))
#         print("Loss for 4: {}".format(optimizers_stat_4_loss[i] / repeats))
#
#         print("Accurancy for 2: {}".format(optimizers_stat_2_acc[i] / repeats))
#         print("Loss for 2: {}".format(optimizers_stat_2_loss[i] / repeats))
#         print("\n")
#         i += 1
#
#
#     # for i in range(2, number_layers):
#     #     print("\t\t----Number of layers: {}----\t\t".format(i))
#     #     #init_network(input_data, output_data, n_input, n_hidden, n_classes, i)
#     #     init_network(input_data, output_data, n_input, n_hidden, n_classes, i, activation=tf.nn.relu,
#     #                  training_epochs=100, display_step=10)
#
#     #    print("\n")


# if __name__ == "__main__":
#     #input_data = read_x_from_file("legenda/X.bin")
#     #output_data = read_y_from_file("legenda/Y1.bin")
#
#     # Network Parameters
#     n_input = 2  # input layer
#     n_hidden = n_input  # n-st layer number of features
#     n_classes = 2
#     number_layers = 20
#
#     n_input_vector = [2 * x for x in range(1, 11)]
#     n_input_stat_acc = [0] * len(n_input_vector)
#     n_input_stat_loss = [0] * len(n_input_vector)
#
#
#     repeats = 10
#     for f in range(0, repeats):
#         print("FF: {}".format(f))
#
#         # init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10)
#         for n_input in n_input_vector:
#             input_data, output_data = DataGenerator(n_input=n_input).generate_data(input_size=pow(2, 10))
#
#             x, y = init_network(input_data, output_data, n_input, n_input, n_classes, n_input, activation=tf.nn.relu,
#                                 training_epochs=1000, display_step=1000)
#             n_input_stat_acc[n_input // 2 - 1] += x
#             n_input_stat_loss[n_input // 2 - 1] += y
#
#     for n_input in n_input_vector:
#         print("Input: {}".format(n_input))
#         print("Accuracy: {}".format(n_input_stat_acc[n_input // 2 - 1] / repeats))
#         print("Loss: {}".format(n_input_stat_loss[n_input // 2 - 1] / repeats))
#
#         print("\n")
#
#     n_input_stat_acc = list(map(lambda x: x / repeats, n_input_stat_acc))
#     n_input_stat_loss = list(map(lambda x: x / repeats, n_input_stat_loss))
#
#
#     # for i in range(2, number_layers):
#     #     print("\t\t----Number of layers: {}----\t\t".format(i))
#     #     #init_network(input_data, output_data, n_input, n_hidden, n_classes, i)
#     #     init_network(input_data, output_data, n_input, n_hidden, n_classes, i, activation=tf.nn.relu,
#     #                  training_epochs=100, display_step=10)
#
#     #    print("\n")
#
# if __name__ == "__main__":
#     #input_data = read_x_from_file("legenda/X.bin")
#     #output_data = read_y_from_file("legenda/Y1.bin")
#
#     # Network Parameters
#     n_input = 2  # input layer
#     n_hidden = n_input  # n-st layer number of features
#     n_classes = 2
#     number_layers = 20
#
#     n_input_vector = [x for x in range(2, 10)]
#     n_input_stat_acc = [0] * len(n_input_vector)
#     n_input_stat_loss = [0] * len(n_input_vector)
#
#
#     repeats = 10
#     for f in range(0, repeats):
#         print("FF: {}".format(f))
#
#         # init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10)
#         for n_input in n_input_vector:
#             input_data, output_data = DataGenerator(n_input=n_input).generate_data_static(input_size=pow(2, 10))
#
#             x, y = init_network(input_data, output_data, n_input, n_input, n_classes, n_input, activation=tf.nn.relu,
#                                 training_epochs=1000, display_step=1000)
#             n_input_stat_acc[n_input - 2] += x
#             n_input_stat_loss[n_input - 2] += y
#
#     for n_input in n_input_vector:
#         print("Input: {}".format(n_input))
#         print("Accuracy: {}".format(n_input_stat_acc[n_input - 2] / repeats))
#         print("Loss: {}".format(n_input_stat_loss[n_input - 2] / repeats))
#
#         print("\n")
#
#     n_input_stat_acc = list(map(lambda x: x / repeats, n_input_stat_acc))
#     n_input_stat_loss = list(map(lambda x: x / repeats, n_input_stat_loss))
#     print(n_input_stat_acc)
#     print(n_input_stat_loss)
#
#     # for i in range(2, number_layers):
#     #     print("\t\t----Number of layers: {}----\t\t".format(i))
#     #     #init_network(input_data, output_data, n_input, n_hidden, n_classes, i)
#     #     init_network(input_data, output_data, n_input, n_hidden, n_classes, i, activation=tf.nn.relu,
#     #                  training_epochs=100, display_step=10)
#
#     #    print("\n")

if __name__ == "__main__":
    input_data = read_x_from_file("legenda/X.bin")[0:1000]
    output_data = read_y_from_file("legenda/Y3.bin")[0:1000]

    # Network Parameters
    n_input = 20  # input layer
    n_hidden = n_input  # n-st layer number of features
    n_classes = 2
    number_layers = 20

    n_layers_vector = [x for x in range(2, 8)]
    n_layers_stat_acc = [0] * len(n_layers_vector)
    n_layers_stat_loss = [0] * len(n_layers_vector)


    repeats = 10
    for f in range(0, repeats):
        print("FF: {}".format(f))

        # init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10)
        for layers in n_layers_vector:
            x, y = init_network(input_data, output_data, n_input, n_input, n_classes, layers, activation=tf.nn.relu,
                                training_epochs=400, display_step=1000)
            n_layers_stat_acc[layers - 2] += x
            n_layers_stat_loss[layers - 2] += y

    for layers in n_layers_vector:
        print("Input: {}".format(n_input))
        print("Accuracy: {}".format(n_layers_stat_acc[layers - 2] / repeats))
        print("Loss: {}".format(n_layers_stat_loss[layers - 2] / repeats))

        print("\n")

    n_layers_stat_acc= list(map(lambda x: x / repeats, n_layers_stat_acc))
    n_layers_stat_loss = list(map(lambda x: x / repeats, n_layers_stat_loss))
    print(n_layers_stat_acc)
    print(n_layers_stat_loss)

    # for i in range(2, number_layers):
    #     print("\t\t----Number of layers: {}----\t\t".format(i))
    #     #init_network(input_data, output_data, n_input, n_hidden, n_classes, i)
    #     init_network(input_data, output_data, n_input, n_hidden, n_classes, i, activation=tf.nn.relu,
    #                  training_epochs=100, display_step=10)

    #    print("\n")
