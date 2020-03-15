import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_files = {
    "g1": ("GOST_generator/bin/out_primitive1_x.bin", "GOST_generator/bin/out_primitive1_y.bin"),
    "g2": ("GOST_generator/bin/out_primitive2_x.bin", "GOST_generator/bin/out_primitive2_y.bin"),
    "g0": ("GOST_generator/bin/out_primitive0_x.bin", "GOST_generator/bin/out_primitive0_y.bin"),

}

def read_from_file(file, num_bytes, format_data=lambda x: int.from_bytes(x, byteorder='little')):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(num_bytes))
        while block:
            data.append(format_data(block))
            block = bytes(f.read(num_bytes))
    return data


def bits(num):
    bits_array = []
    for _ in range(0, 8):
        bits_array.append(num % 2)
        num //= 2
    return list(reversed(bits_array))


def _int(bit_array):
    value = 0
    for i in bit_array:
        value *= 2
        value += i
    return int(value)


def split_by_bit(block, bit=32, order='left'):
    rv = []
    for i in block:
        rv += bits(i)
    return rv[:bit] if order == 'left' else rv[bit:]


def split_by_byte(block, bytes=4):
    rv = []
    for i in block:
        rv.append(int(i))
    return rv[:bytes]


def xor(x1, x2):
    return list(map(lambda x: x[0] ^ x[1], zip(x1, x2)))


def get_test_values_xor(model):
    __in = read_from_file(input_files[model][0], 1,
                          lambda x: split_by_bit(x, 4))

    __xor = read_from_file(input_files[model][0], 1,
                          lambda x: split_by_bit(x, 4, order='right'))

    __out = read_from_file(input_files[model][1], 1,
                           lambda x: split_by_bit(x, 4, order='right'))

    __out = list(map(lambda x: xor(x[0], x[1]), zip(__out, __xor)))

    return np.array(__in), np.array(__out)


def get_test_values1(size=[32, 32]):
    __in = read_from_file("GOST_generator/bin/out_iterate_x.bin", 8,
                          lambda x: split_by_bit(x, size[0]))
    __out = read_from_file("GOST_generator/bin/out_iterate_y.bin", 8,
                           lambda x: split_by_bit(x, size[1]))

    return np.array(__in), np.array(__out)

def get_test_values(model):
    __in = read_from_file(input_files[model][0], 1,
                          lambda x: split_by_bit(x, 8))

    __out = read_from_file(input_files[model][1], 1,
                           lambda x: split_by_bit(x, 4, order='right'))

    return np.array(__in), np.array(__out)


def init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10,
                           optimizer=tf.train.AdamOptimizer):

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Create the model
    # W = tf.Variable(tf.random_normal(shape=[n_input, 1]))
    # b = tf.Variable(tf.random_normal([1]))
    # out = tf.matmul(x, W) + b

    out = tf.layers.dense(x, n_classes, activation=tf.nn.sigmoid)

    hamming_distance = tf.math.count_nonzero(tf.round(out) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)

    loss = tf.losses.sigmoid_cross_entropy(y, out)
    loss1 = tf.keras.losses.binary_crossentropy(y, out)

    cost = tf.reduce_mean(tf.square(out - y))

    cost1 = tf.reduce_mean(tf.cast(tf.math.count_nonzero(out - y, axis=-1), tf.float32))

    train_step = optimizer(learning_rate=0.002).minimize(loss)

    init = tf.global_variables_initializer()
    # Train the model, and feed in test data and record summaries every 10 steps

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

        test_values1 = tf.round(out).eval(feed_dict={x: test_dataset})

        print(list(map(lambda x: hex(_int(x)), test_values1)))
        print(list(map(lambda x: hex(_int(x)), test_values)))
        print(list(map(lambda x: hex(_int(x)), test_dataset)))

        accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        print("testing accuracy: {}".format(acc))
        print("testing Loss: {}".format(los))

        sess.close()

    return acc, los


def multilayer_perceptron(x, hidden, num_classes, number_layers, activation_fun=tf.nn.sigmoid,
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
                            bias_initializer=tf.initializers.ones())

    # Hidden layer with activation
    for i in range(1, number_layers):
        layer_new = tf.layers.dense(layer, hidden[i],
                                    activation=activation_fun,
                                    kernel_initializer=tf.initializers.ones(),
                                    bias_initializer=tf.initializers.ones())
        layer = layer_new

    # Output layer with linear activation
    out_layer = tf.layers.dense(layer, num_classes, activation=None)
    return out_layer


def init_network(input_data, output_data, n_input, n_hidden, n_classes, number_layers, learning_rate=0.001,
                 training_epochs=100, display_step=10, activation=tf.nn.sigmoid, optimizer=tf.train.AdamOptimizer):
    with open("log1.txt", "a+") as f:
        f.write('Start of neuron number_layers - {}\n'.format(number_layers))
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

    # prediction = __multilayer_perceptron(x,
    #                                      __generate_weigths(n_input, n_hidden, n_classes, number_layers),
    #                                      __generate_biases(n_input, n_hidden, n_classes, number_layers),
    #                                      number_layers)

    hamming_distance = tf.math.count_nonzero(tf.round(prediction) - y, axis=-1)
    accuracy = tf.reduce_mean(hamming_distance)
    cost = tf.reduce_mean(tf.square(prediction - y))

    loss = tf.losses.sigmoid_cross_entropy(y, prediction)
    loss1 = tf.keras.losses.binary_crossentropy(y, prediction)

    loss = cost

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
                with open("log1.txt", "a+") as f:
                    f.write('Accuracy at step %s: %f - loss: %f\n' % (i, result[1], result[0]))
                print('Accuracy at step %s: %f - loss: %f\n' % (i, result[1], result[0]))
            else:
                feed = {x: train_dataset, y: train_values}
            sess.run(optimizer, feed_dict=feed)

        print("Optimization Finished!")

        test_values1 = sess.run(prediction, feed_dict={
            x: test_dataset,
        })

        print(list(map(lambda x: hex(_int(x)), test_values1)))
        print(list(map(lambda x: hex(_int(x)), test_values)))
        print(list(map(lambda x: hex(_int(x)), test_dataset)))

        accuracy = tf.reduce_mean(tf.cast(hamming_distance, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        with open("log1.txt", "a+") as f:
            f.write("testing accuracy: {}\n".format(acc))
            f.write("testing Loss: {}\n".format(los))
            f.write("\n\n\n\n")

        print("testing accuracy: {}".format(acc))
        print("testing Loss: {}".format(los))

        sess.close()

    return acc, los


def experiment_std(input_data, output_data, n_input, n_classes, training_epochs=15000, display_step=1000):
    x, y = init_one_layer_network(input_data, output_data, n_input, n_classes,
                                  training_epochs=training_epochs, display_step=display_step)

    x, y = init_network(input_data, output_data, n_input, [4], n_classes, 1,
                        activation=tf.nn.sigmoid, training_epochs=training_epochs, display_step=display_step)

    x, y = init_network(input_data, output_data, n_input, [8], n_classes, 1,
                        activation=tf.nn.sigmoid, training_epochs=training_epochs, display_step=display_step)

    x, y = init_network(input_data, output_data, n_input, [16], n_classes, 1,
                        activation=tf.nn.sigmoid, training_epochs=training_epochs, display_step=display_step)

    x, y = init_network(input_data, output_data, n_input, [32], n_classes, 1,
                        activation=tf.nn.sigmoid, training_epochs=training_epochs, display_step=display_step)

    x, y = init_network(input_data, output_data, n_input, [32, 32], n_classes, 2,
                        activation=tf.nn.sigmoid, training_epochs=training_epochs, display_step=display_step)


def experiment_changeable_0l(input_data, output_data, n_input, n_classes, model_name,
                          activation=tf.nn.sigmoid, training_epochs_min=5000, training_epochs_max=30000, display_step=20000):
    accurancy = []
    loss = []
    train = []
    for i in range(training_epochs_min, training_epochs_max, 5000):
        x, y = init_one_layer_network(input_data, output_data, n_input, n_classes,
                                      training_epochs=i, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        train.append(i)
    create_graph(train, accurancy, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                 "results/acc_" + model_name + "_0.png")
    create_graph(train, loss, ("Number of Training Epochs", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_0.png")


def experiment_changeable_1l(input_data, output_data, n_input, n_classes, left_b, right_b, model_name,
                          activation=tf.nn.sigmoid, training_epochs=10000, display_step=20000):
    accurancy = []
    loss = []
    n_hidden = []
    for i in range(left_b, right_b, 4):
        x, y = init_network(input_data, output_data, n_input, [i], n_classes, 1,
                            activation=activation, training_epochs=training_epochs, display_step=display_step)
        accurancy.append(n_classes - x)
        loss.append(y)
        n_hidden.append(i)
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_" +model_name + "_1layer.png")
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los" + model_name + "_1layer.png")


def experiment_changeable_2l(input_data, output_data, n_input, n_classes, left_b, right_b, model_name,
                          activation=tf.nn.sigmoid, training_epochs=10000, display_step=20000):
    accurancy = []
    tikcs = []
    loss = []
    n_hidden = []
    z = 0
    for i in (8, 16, 32):
        for j in (8, 16, 32):
            x, y = init_network(input_data, output_data, n_input, [i, j], n_classes, 2,
                                activation=activation, training_epochs=training_epochs, display_step=display_step)
            accurancy.append(n_classes - x)
            loss.append(y)
            n_hidden.append(z)
            tikcs.append(str((i, j)))
            z += 1
    create_graph(n_hidden, accurancy, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/acc_2l_" + model_name + "_1layer.png", tikcs)
    create_graph(n_hidden, loss, ("Number of neurons on hidden layer", "Accurancy", "Model " + model_name),
                 "results/los_2l_" + model_name + "_1layer.png", tikcs)


def create_graph(x, y, legend: list, filename: str, ticks=None):
    plt.plot(x, y)
    if ticks is not None:
        plt.xticks(x, ticks)

    plt.xlabel(legend[0])
    plt.ylabel(legend[1])


    plt.title(legend[2])

    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    # # Network Parameters
    # n_input = 8
    # n_classes = 4
    #
    #
    # input_data, output_data = get_test_values(model="g0")
    # print("Experiment g0\n\n")
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g0")
    #
    # input_data, output_data = get_test_values(model="g1")
    # print("Experiment g1\n\n")
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g1")
    #
    # input_data, output_data = get_test_values(model="g2")
    # print("Experiment g2\n\n")
    # experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g2")

    n_input = 64
    n_classes = 32
    input_data, output_data = get_test_values1([64, 32])
    print("Experiment g3\n\n")
    experiment_changeable_0l(input_data, output_data, n_input, n_classes, "g3")
    experiment_changeable_1l(input_data, output_data, n_input, n_classes, 8, 32, "g3")
    experiment_changeable_2l(input_data, output_data, n_input, n_classes, 8, 32, "g3")

# RESULTS #
# 2. Key = 0
# 2. Without xor of blocks: Accuracy = 1.121250033378601, 50000 - traings
# 2. With xor: Accuracy =  1.3699500560760498, 50000 - traings
# 2. With xor: Accuracy =  0.8504499793052673, 15000 - traings, [8]
# 2. With xor: Accuracy =  1.4635499715805054, 15000 - traings, [4]
# 2. With xor: Accuracy =  0.07020000368356705, 15000 - traings, [16]
# 2. With xor: Accuracy =  0.00, 15000 - traings, [32]
# 2. With xor: Accuracy =  2.012, 15000 - traings, [32, 32]

# 1. Without xor of blocks: Accuracy = 1.121250033378601, 50000 - traings
# 1. With xor: Accuracy =  1.6059999465942383, 50000 - traings
# 1. With xor: Accuracy =  1.253999948501587, 15000 - traings, [8]
# 1. With xor: Accuracy =  1.4520000219345093, 15000 - traings, [4]
# 1. With xor: Accuracy =  1.1009999513626099, 15000 - traings, [16]
# 1. With xor: Accuracy =  1.00600004196167, 15000 - traings, [32]
# 1. With xor: Accuracy =  1.5750000476837158, 15000 - traings, [4, 4]

#0.984499990940094
