import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def stacked_auto_encoder():
    # 参数
    learning_rate = 0.01  # 学习速率
    training_epochs = 20  # 训练批次
    batch_size = 256  # 随机选择训练数据大小
    display_epoch = 1  # 展示步骤
    show_num = 10  # 显示示例图片数量

    # 网络参数
    n_input = 784  # 输入
    n_hidden_1 = 256  # 第一隐层神经元数量
    n_hidden_2 = 64  # 第二
    n_hidden_3 = 10

    # 权重初始化
    weights = {
        # 网络1 784-256-256-784
        'l1_h1': tf.Variable(tf.random_normal(shape=[n_input, n_hidden_1], stddev=0.1)),  # 级联使用
        'l1_h2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_1], stddev=0.1)),
        'l1_out': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_input], stddev=0.1)),
        # 网络2 256-64-64-256
        'l2_h1': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),  # 级联使用
        'l2_h2': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_2], stddev=0.1)),
        'l2_out': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_1], stddev=0.1)),
        # 网络3 64-10-10-64
        'l3_h1': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),  # 级联使用
        'l3_h2': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_3], stddev=0.1)),
        'l3_out': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_2], stddev=0.1)),
    }

    # 偏置值初始化
    biases = {
        # 网络1 784-256-256-784
        'l1_b1': tf.Variable(tf.random_normal([n_hidden_1])),  # 级联使用
        'l1_b2': tf.Variable(tf.random_normal([n_hidden_1])),
        'l1_out': tf.Variable(tf.random_normal([n_input])),
        # 网络2 256-64-64-256
        'l2_b1': tf.Variable(tf.random_normal([n_hidden_2])),  # 级联使用
        'l2_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'l2_out': tf.Variable(tf.random_normal([n_hidden_1])),
        # 网络3 64-10-10-64
        'l3_b1': tf.Variable(tf.random_normal([n_hidden_3])),  # 级联使用
        'l3_b2': tf.Variable(tf.random_normal([n_hidden_3])),
        'l3_out': tf.Variable(tf.random_normal([n_hidden_2])),
    }
    
    # 第一层输入
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_input])

    # 第二层输入
    l2x = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_1])
    l2y = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_1])

    # 第三层输入
    l3x = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_2])
    l3y = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_2])


    '''
    定义第一层网络结构784-256-256-784
    '''
    l1_h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['l1_h1']), biases['l1_b1']))
    l1_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l1_h1, weights['l1_h2']), biases['l1_b2']))
    l1_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l1_h2, weights['l1_out']), biases['l1_out']))
    # 计算代价
    l1_cost = tf.reduce_mean((l1_reconstruction - y) ** 2)
    # 定义优化器
    l1_optm = tf.train.AdamOptimizer(learning_rate).minimize(l1_cost)


    '''
    定义第二层网络结构256-64-64-256
    '''
    l2_h1 = tf.nn.sigmoid(tf.add(tf.matmul(l2x, weights['l2_h1']), biases['l2_b1']))
    l2_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l2_h1, weights['l2_h2']), biases['l2_b2']))
    l2_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l2_h2, weights['l2_out']), biases['l2_out']))
    # 计算代价
    l2_cost = tf.reduce_mean((l2_reconstruction - l2y) ** 2)
    # 定义优化器
    l2_optm = tf.train.AdamOptimizer(learning_rate).minimize(l2_cost)


    '''
    定义第三层网络结构 64-10-10-64
    '''
    l3_h1 = tf.nn.sigmoid(tf.add(tf.matmul(l3x, weights['l3_h1']), biases['l3_b1']))
    l3_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l3_h1, weights['l3_h2']), biases['l3_b2']))
    l3_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l3_h2, weights['l3_out']), biases['l3_out']))
    # 计算代价
    l3_cost = tf.reduce_mean((l3_reconstruction - l3y) ** 2)
    # 定义优化器
    l3_optm = tf.train.AdamOptimizer(learning_rate).minimize(l3_cost)


    num_batch = int(mnist.train.num_examples / batch_size)

    '''
    训练 网络第一层
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('网络第一层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # 添加噪声 每次取出来一批次的数据，将输入数据的每一个像素都加上0.3倍的高斯噪声
                batch_x_noise = batch_x + 0.3 * np.random.randn(batch_size, 784)  # 标准正态分布
                _, loss = sess.run([l1_optm, l1_cost], feed_dict={x: batch_x_noise, y: batch_x})
                total_cost += loss
            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))
        print('训练完成')


    '''
    训练 网络第二层
    注意：这个网络模型的输入已经不再是MNIST图片了，而是上一层网络中的一层的输出
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('网络第二层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                l1_out = sess.run(l1_h1, feed_dict={x: batch_x})
                _, loss = sess.run([l2_optm, l2_cost], feed_dict={l2x: l1_out, l2y: l1_out})
                total_cost += loss
            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))
        print('训练完成')


    '''
    训练 网络第三层
    注意：同理这个网络模型的输入是要经过前面两次网络运算才可以生成
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('网络第三层 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                l1_out = sess.run(l1_h1, feed_dict={x: batch_x})
                l2_out = sess.run(l2_h1, feed_dict={l2x: l1_out})
                _, loss = sess.run([l3_optm, l3_cost], feed_dict={l3x: l2_out, l3y: l2_out})
                total_cost += loss
            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))
        print('训练完成')

        new_weights_1 = []
        new_weights_2 = []
        new_weights_3 = []

        t = tf.cast(weights['l1_h1'], dtype=tf.float32)
        for i in range(len(t.eval())):
            new_weights_1.append(list(t.eval()[i]))

        t = tf.cast(weights['l2_h1'], dtype=tf.float32)
        for i in range(len(t.eval())):
            new_weights_2.append(list(t.eval()[i]))

        t = tf.cast(weights['l3_h1'], dtype=tf.float32)
        for i in range(len(t.eval())):
            new_weights_3.append(list(t.eval()[i]))

        new_weights = []
        new_weights.append(new_weights_1)
        new_weights.append(new_weights_2)
        new_weights.append(new_weights_3)

        # print("len(weights): ", len(new_weights))
        # print("len(weights[0]), len(weights[1]), len(weights[2]): ",
        #       len(new_weights[0]), len(new_weights[1]), len(new_weights[2]))
        # print("len(weights[0][0]): ", len(new_weights[0][0]))
        # print(new_weights[2][0])

        return new_weights
if __name__ == '__main__':
    stacked_auto_encoder()