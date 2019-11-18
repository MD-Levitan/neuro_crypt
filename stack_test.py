import tensorflow as tf
import numpy as np
# from test_generator import DataGenerator
from itertools import combinations, chain
from functools import reduce
# from n_layers import read_x_from_file, read_y_from_file


def init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10,
                           optimizer = tf.train.AdamOptimizer):

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
    # n_input = pow(2, n_input) - 1

    # Create the model
    x = tf.placeholder(tf.float32, [None, n_input], name='x-input')
    W = tf.Variable(tf.random_normal(shape=[n_input, n_classes]))
    b = tf.Variable(tf.zeros([n_classes], name='bias'))
    logits = tf.matmul(x, W) % 2 + b

    # Define loss and optimizer
    y = tf.placeholder(tf.float32, [None, 2], name='y-input')

    # More name scopes will clean up the graph representation
        # manual calculation : under the hood math, don't use this it will have gradient problems
        # entropy = tf.multiply(tf.log(tf.sigmoid(logits)), y_) + tf.multiply((1 - y_), tf.log(1 - tf.sigmoid(logits)))
        # loss = -tf.reduce_mean(entropy, name='loss')

    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

    # Using Adam instead
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    train_step = optimizer(learning_rate=0.002).minimize(loss)

    preds = tf.nn.sigmoid(logits)
    correct_prediction = tf.equal(tf.round(preds), y)

    # preds = tf.cast((logits > 0.5), tf.float32)
    # correct_prediction = tf.equal(preds, y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    # Train the model, and feed in test data and record summaries every 10 steps

    with tf.Session() as sess:
        sess.run(init)
        for i in range(training_epochs):
            if i % display_step == 0:  # Record summary data and the accuracy
                train_dataset = np.array(input_data[:9 * len(input_data) // 10])
                train_values = np.array(output_data[: 9 * len(input_data) // 10])

                test_dataset = np.array(input_data[-len(input_data) // 10:])
                test_values = np.array(output_data[-len(input_data) // 10:])

                feed = {x: train_dataset, y: train_values}
                result = sess.run([loss, accuracy], feed_dict=feed)
                print('Accuracy at step %s: %f - loss: %f' % (i, result[1], result[0]))
            else:
                feed = {x: train_dataset, y: train_values}
            sess.run(train_step, feed_dict=feed)

        # Test model and check accuracy
        print('Test Accuracy:', sess.run([accuracy, preds], feed_dict={x: test_dataset, y: test_values}))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        print("testing accuracy: {}".format(acc))
        print("testing Loss: {}".format(los))

        result = tf.argmax(logits, 1).eval({x: test_dataset, y: test_values})
        print(list(result)[0:50])
        print(list(map(lambda x: 1 if x[1] else 0, test_values))[0:50])
        sess.close()

    return acc, los

#
# if __name__ == "__main__":
#     n_input = 4  # input layer
#     n_hidden = n_input  # n-st layer number of features
#     n_classes = 2
#     number_layers = 20
#     from test_generator import DataGenerator
#     input_data, output_data = DataGenerator(n_input=n_input).generate_data(input_size=pow(2, 20))
#
#     print("\t\t----Number of layers: 1----\t\t")
#     init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=2000, display_step=200)

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


optimizers = {"Adam": tf.train.AdamOptimizer, "ProximalAdarad": tf.train.ProximalAdagradOptimizer,
              "Proximal": tf.train.ProximalGradientDescentOptimizer, "Adagard": tf.train.AdagradOptimizer,
              "Gradient": tf.train.GradientDescentOptimizer}
from test_generator import DataGenerator

if __name__ == "__main__":
    #input_data = read_x_from_file("legenda/X.bin")
    #output_data = read_y_from_file("legenda/Y1.bin")

    for n_input in range(4, 10):
        print("N_input: {}".format(n_input))
        # Network Parameters
        n_input = 6  # input layer
        n_hidden = n_input  # n-st layer number of features
        n_classes = 2
        number_layers = 20

        optimizers_stat_4_acc = [0] * len(optimizers)
        optimizers_stat_4_loss = [0] * len(optimizers)

        repeats = 10
        input_data, output_data = DataGenerator(n_input=n_input).generate_data_static(input_size=pow(2, 10))

        for f in range(0, repeats):
            print("FF: {}".format(f))

            # init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10)
            i = 0
            for (_, val) in optimizers.items():
                x, y = init_one_layer_network(input_data, output_data, n_input, n_classes,
                                              training_epochs=1000, display_step=1000, optimizer=val)
                optimizers_stat_4_acc[i] += x
                optimizers_stat_4_loss[i] += y
                i += 1

        i = 0
        for (key, val) in optimizers.items():
            print("Optimizer: {}".format(key))
            print("Accurancy for 4: {}".format(optimizers_stat_4_acc[i] / repeats))
            print("Loss for 4: {}".format(optimizers_stat_4_loss[i] / repeats))

            print("\n")
            i += 1


    # for i in range(2, number_layers):
    #     print("\t\t----Number of layers: {}----\t\t".format(i))
    #     #init_network(input_data, output_data, n_input, n_hidden, n_classes, i)
    #     init_network(input_data, output_data, n_input, n_hidden, n_classes, i, activation=tf.nn.relu,
    #                  training_epochs=100, display_step=10)

    #    print("\n")