import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# MNIST 데이터 셋, https://www.tensorflow.org/get_started/mnist/beginners for


# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            # 입력 데이터 placeholder
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])

            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1 - 필터 수 : 32, 필터 사이즈 : 3 by 3, 패딩 : same
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)

            # Pooling Layer #1 - 필터 사이즈 : 2 by 2, 패딩 : same, stride : 2
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #2 - 필터 수 : 64, 필터 사이즈 3 by 3, 패딩 : same
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)

            # Pooling Layer #2 - 필터 사이즈 : 2 by 2, 패딩 : same, stride : 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)

            # dropout : 70% 의 뉴런만으로 학습
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.7, training=self.training)

            # Convolutional Layer #3 : 필터수 : 128, 필터 사이즈 : 3 by 3, 패딩 : same
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                     padding="same", activation=tf.nn.relu)

            #  Pooling Layer #2 : 필터 사이즈 : 2 by 2, 패딩 : same, stride : 2
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="same", strides=2)

            # dropout : 70% 의 뉴런만으로 학습
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.7, training=self.training)

            # Normal Nerual Layer - 출력 뉴런 수 : 625
            flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)

            # dropout : 50% 의 뉴런만으로 학습
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Final Layer - 출력 뉴런 수 : 10
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # Cost 함수/ Gradient Descent 알고리즘 Optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    # 예측값과 실제 Y 데이터의 정확도를 구한다
    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    # 1 epoch의 트레이닝을 수행한다.
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

    # 딥러닝을 통해 얻은 hypothesis를 통해, 입력 데이터의 예측값을 구한다
    def get_one_predicted(self, x_test, training=False):
        return self.sess.run(tf.arg_max(self.logits, 1),
                             feed_dict={self.X: x_test, self.training: training})


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

print("Accuracy:", m1.get_accuracy(mnist.test.images, mnist.test.labels))
while True:
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction:", m1.get_one_predicted(mnist.test.images[r:r + 1]))
    plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    if input("Continue? (y/n) : ") == "n":
        break


