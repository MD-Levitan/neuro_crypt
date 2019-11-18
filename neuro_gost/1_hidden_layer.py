import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

iterations = 10000
learn_rate = 0.01

hiddenSize = 4


def read_from_file(file, num_bytes, format_data=lambda x: int.from_bytes(x, byteorder='little')):
    data = []
    with open(file, 'rb') as f:
        block = bytes(f.read(num_bytes))
        while block:
            data.append(format_data(block))
            block = bytes(f.read(num_bytes))
    return data


def to_float(block, bytes=4):
    return [int.from_bytes(block[:bytes], byteorder='little') / pow(2, 8 * bytes)]


def to_int(block, bytes=4):
    return [int.from_bytes(block[:bytes], byteorder='little')]


def get_test_values():
    input_data = read_from_file("GOST_generator/out_x.bin", 8, to_float)
    output_data = read_from_file("GOST_generator/out_y.bin", 8, to_float)

    return np.array(input_data), np.array(output_data)


def get_clear_values():
    input_data = read_from_file("GOST_generator/out_x.bin", 8, to_int)
    output_data = read_from_file("GOST_generator/out_y.bin", 8, to_int)

    return np.array(input_data), np.array(output_data)


x = tf.placeholder(tf.float32, [None, 1], name="x")
y = tf.placeholder(tf.float32, [None, 1], name="y")

nn = tf.layers.dense(x, hiddenSize,
                     activation=tf.nn.relu,
                     kernel_initializer=tf.initializers.ones(),
                     bias_initializer=tf.initializers.random_normal(),
                     name="hidden")

model = tf.layers.dense(nn, 1,
                        activation=None,
                        name="output")

cost = tf.reduce_mean(tf.square(model - y))
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
init = tf.initializers.global_variables()


with tf.Session() as session:
    session.run(init)

    train_dataset, train_values = get_test_values()

    for _ in range(iterations):

        session.run(train, feed_dict={
            x: train_dataset,
            y: train_values
        })

        if(_ % 1000 == 999):
            print("cost = {}".format(session.run(cost, feed_dict={
                x: train_dataset,
                y: train_values
            })))

    train_dataset, train_values = get_test_values()
    clear_dataset, clear_values = get_clear_values()

    train_values1 = list(map(lambda x: x * pow(2, 32), session.run(model, feed_dict={
        x: train_dataset,
    })))

    plt.plot(clear_dataset, clear_values, "bo",
             clear_dataset, train_values1, "ro")
    plt.show()

    # Delete
    for i in range(0, 100):
        print("{} -- {}".format(clear_values[i], train_values1[i]))

    acc = cost.eval(feed_dict={x: train_dataset, y: train_values})
    print("Testing accuracy: {}".format(acc))

    with tf.variable_scope("hidden", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("hidden:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())

    with tf.variable_scope("output", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("output:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())