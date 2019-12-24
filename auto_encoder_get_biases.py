import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def auto_encoder():
    # 参数
    learning_rate = 0.01  # 学习速率
    training_epochs = 1  # 训练批次
    batch_size = 256  # 随机选择训练数据大小
    display_epoch = 1  # 展示步骤
    show_num = 10  # 显示示例图片数量

    # 网络参数
    n_input = 784  # 输入
    n_hidden_1 = 30  # 第一隐层神经元数量
    n_hidden_2 = 10  # 第二

    # 权重初始化
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }

    # 偏置值初始化
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }

    # 定义占位符
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    y_true = X  # 不能用y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_input])替代

    # 网络模型
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1, weights['encoder_h2']), biases['encoder_b2']))
    h3 = tf.nn.sigmoid(tf.add(tf.matmul(h2, weights['decoder_h1']), biases['decoder_b1']))
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(h3, weights['decoder_h2']), biases['decoder_b2']))

    # 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
    cost = tf.reduce_mean(tf.pow(y_true- y_pred, 2))
    # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(mnist.train.num_examples / batch_size)
        print('开始训练')
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optimizer, cost], feed_dict={X: batch_xs})

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  cost {2}'.format(epoch + 1, training_epochs, loss))
        print('训练完成')

        new_weights_1 = []
        new_weights_2 = []
        t = tf.cast(weights['encoder_h1'], dtype=tf.float32)
        for i in range(len(t.eval())):
            new_weights_1.append(list(t.eval()[i]))
        t = tf.cast(weights['encoder_h2'], dtype=tf.float32)
        for i in range(len(t.eval())):
            new_weights_2.append(list(t.eval()[i]))
        new_weights = []
        new_weights.append(new_weights_1)
        new_weights.append(new_weights_2)

        t = tf.cast(biases['encoder_b1'], dtype=tf.float32)
        new_biases_1 = np.array(t.eval()).reshape(len(t.eval()), 1)
        t = tf.cast(biases['encoder_b2'], dtype=tf.float32)
        new_biases_2 = np.array(t.eval()).reshape(len(t.eval()), 1)
        new_biases = []
        new_biases.append(new_biases_1)
        new_biases.append(new_biases_2)

        return new_weights

if __name__ == '__main__':
    auto_encoder()