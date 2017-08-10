import numpy as np
import os
import random
import tensorflow as tf

from task_io import TaskIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Session3TaskIO(TaskIO):
    def get_processed_training(self):
        pass

    def get_processed_testing(self):
        pass

    def score(self, y_test, y_predicted):
        pass


def get_batch(data, labels, size):
    idx = random.sample(range(len(labels)), size)
    batch_xs = data[idx, :]
    batch_ys = list()
    for label in labels[idx, :]:
        n_label = np.zeros(10)
        n_label[label] = 1
        batch_ys.append(n_label)
    return batch_xs, batch_ys


def main():
    task_io = Session3TaskIO(
        train='./data/all_label.p',
        test='./data/test.p',
        result='./data/submission.csv',
        validation='./data/validation.csv'
    )

    train_data = task_io.import_training_data()

    data = train_data[b'data']
    labels = np.asarray(train_data[b'labels'])

    # Build the training model
    feature_size = 3072
    x = tf.placeholder(tf.float32, [None, feature_size])
    W = tf.Variable(tf.zeros([feature_size, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(10000):
        batch_xs, batch_ys = get_batch(data, labels, 100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_images, test_labels = get_batch(data, labels, 1000)
    print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))

    # Making predictions for testing data
    test_data = task_io.import_testing_data()
    data = test_data[b'data']
    labels = np.asarray(test_data[b'labels'])


if __name__ == '__main__':
    main()
