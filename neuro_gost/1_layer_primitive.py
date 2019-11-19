import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def get_test_values():
    __in = read_from_file("GOST_generator/bin/out_primitive_x.bin", 1,
                          lambda x: split_by_bit(x, 4))

    __xor = read_from_file("GOST_generator/bin/out_primitive_x.bin", 1,
                          lambda x: split_by_bit(x, 4, order='right'))

    __out = read_from_file("GOST_generator/bin/out_primitive_y.bin", 1,
                           lambda x: split_by_bit(x, 4, order='right'))

    __out = list(map(lambda x: xor(x[0], x[1]), zip(__out, __xor)))

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


if __name__ == "__main__":
    input_data, output_data = get_test_values()

    # Network Parameters
    n_input = 4
    n_classes = 4

    x, y = init_one_layer_network(input_data, output_data, n_input, n_classes,
                                  training_epochs=10000, display_step=1000)
