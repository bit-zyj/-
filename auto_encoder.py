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
    training_epochs = 50  # 训练批次
    batch_size = 256  # 随机选择训练数据大小
    display_epoch = 1  # 展示步骤
    show_num = 10  # 显示示例图片数量

    # 网络参数
    n_input = 784  # 输入
    n_hidden_1 = 128  # 第一隐层神经元数量
    n_hidden_2 = 30  # 第二
    n_hidden_3 = 10

    # 权重初始化
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }

    # 偏置值初始化
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    }

    # 定义占位符
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    #y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    y_true = X# 不能用y_true = tf.placeholder(dtype=tf.float32, shape=[None, n_input])替代

    # 网络模型
    input_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
    input_2 = tf.nn.sigmoid(tf.add(tf.matmul(input_1, weights['encoder_h2']), biases['encoder_b2']))
    input_3 = tf.nn.sigmoid(tf.add(tf.matmul(input_2, weights['encoder_h3']), biases['encoder_b3']))
    output_3 = tf.nn.sigmoid(tf.add(tf.matmul(input_3, weights['decoder_h3']), biases['decoder_b3']))
    output_2 = tf.nn.sigmoid(tf.add(tf.matmul(output_3, weights['decoder_h2']), biases['decoder_b2']))
    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(output_2, weights['decoder_h1']), biases['decoder_b1']))

    # 定义代价函数和优化器，最小化平方误差,这里可以根据实际修改误差模型
    cost = tf.reduce_mean(tf.pow(y_true- y_pred, 2))
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_batch = int(mnist.train.num_examples / batch_size)
        print('开始训练')
        for epoch in range(training_epochs):
            total_loss = 0.0
            for i in range(num_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                total_loss += loss
            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}, average cost {2}'.format(epoch + 1, training_epochs, total_loss/num_batch))
        print('训练完成')

        # 测试集数据可视化
        reconstruction = sess.run(y_pred, feed_dict={X: mnist.test.images[:show_num]})

        # 对比原始图片重建图片
        f, a = plt.subplots(2, show_num, figsize=(show_num, 2))
        for i in range(show_num):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


if __name__ == '__main__':
    auto_encoder()