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


def split_by_byte(block):
    rv = []
    for i in block:
        rv.append(int(i))
    return rv


def to_float(block, bytes=4):
    return [int.from_bytes(block[:4], byteorder='little') / pow(2, 8 * bytes)]


def get_test_values():
    __in = read_from_file("GOST_generator/out_x.bin", 8, to_float)
    __out = read_from_file("GOST_generator/out_y.bin", 8, to_float)

    return np.array(__in), np.array(__out)


def init_one_layer_network(input_data, output_data, n_input, n_classes, training_epochs=100, display_step=10,
                           optimizer=tf.train.AdamOptimizer):

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, 1])

    # Create the model
    # W = tf.Variable(tf.random_normal(shape=[n_input, 1]))
    # b = tf.Variable(tf.random_normal([1]))
    # out = tf.matmul(x, W) + b

    out = tf.layers.dense(x, 1, activation=tf.nn.sigmoid, name="output")

    loss = tf.reduce_mean(tf.square(out - y))
    train_step = optimizer(learning_rate=0.002).minimize(loss)
    correct_prediction = tf.equal(out, y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

        test_values1 = sess.run(out, feed_dict={
            x: test_dataset,
        })

        plt.plot(test_dataset, test_values, "bo",
                 test_dataset, test_values1, "ro")
        plt.show()

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval(feed_dict={x: test_dataset, y: test_values})
        los = loss.eval(feed_dict={x: test_dataset, y: test_values})

        print("testing accuracy: {}".format(acc))
        print("testing Loss: {}".format(los))

        sess.close()

    return acc, los


if __name__ == "__main__":
    input_data, output_data = get_test_values()

    # Network Parameters
    n_input = 1

    x, y = init_one_layer_network(input_data, output_data, n_input, 1,
                                  training_epochs=10000, display_step=1000)
