from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np


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

# Training Parameters
learning_rate = 0.001
num_steps = 200
display_step = 10

# Network Parameters
n_input = 20
n_classes = 2
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, n_input, n_input])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, n_input, n_input])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([n_input, n_input])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([n_input, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([n_input])),
    'bc2': tf.Variable(tf.random_normal([n_input])),
    'bd1': tf.Variable(tf.random_normal([n_input])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
logits = conv_net(x, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    train_dataset = np.array(input_data[:9 * len(input_data) // 10])
    train_values = np.array(output_data1[: 9 * len(input_data) // 10])

    test_dataset = np.array(input_data[-len(input_data) // 10:])
    test_values = np.array(output_data1[-len(input_data) // 10:])

    for step in range(1, num_steps+1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={x: train_dataset, y: test_values, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: train_dataset,
                                                                 y: train_values,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_dataset,
                                      y: test_values,
keep_prob: 1.0}))