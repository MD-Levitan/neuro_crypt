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
n_hidden_1 = 20  # 1st layer number of features
n_hidden_2 = 20  # 2nd layer number of features
n_hidden_3 = 20  # 3nd layer number of features
n_input = 20  # input layer
n_classes = 2
learning_rate = 0.001
training_epochs = 1000
display_step = 100
batch_size = 32

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with SIGMOID activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with SIGMOID activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Hidden layer with SIGMOID activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


# Store layers weight &amp; bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# cost = tf.losses.mean_squared_error(y, multilayer_perceptron(x, weights, biases))
prediction = multilayer_perceptron(x, weights, biases)
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
