import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

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
training_steps = 10000
display_step = 200

# Network Parameters
n_input = 20
timesteps = len(input_data)  # timesteps
n_hidden = 20
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, timesteps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(x, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
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

    for step in range(1, training_steps+1):
        sess.run(train_op, feed_dict={x: train_dataset, y: train_values})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: train_dataset,
                                                                 y: train_values})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_dataset, y: test_values}))
