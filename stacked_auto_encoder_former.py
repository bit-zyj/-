import warnings
warnings.filterwarnings("ignore")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

# 导入 MNIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

'''
去燥自编码器和栈式自编码器的综合实现
1.我们首先建立一个去噪自编码器(包含输入层四层的网络)
2.然后再对第一层的输出做一次简单的自编码压缩(包含输入层三层的网络)
3.然后再将第二层的输出做一个softmax分类
4.最后把这3个网络里的中间层拿出来，组成一个新的网络进行微调1.构建一个包含输入层的简单去噪自编码其
'''

def stacked_auto_encoder():
    tf.reset_default_graph()
    '''
    栈式自编码器

    最终训练的一个网络为一个输入、一个输出和两个隐藏层
    MNIST输入(784) - > 编码层1(256)- > 编码层3(128) - > softmax分类

    除了输入层，每一层都用一个网络来训练，于是我们需要训练3个网络，最后再把训练好的各层组合在一起，形成第4个网络。
    '''
    n_input = 784
    n_hidden_1 = 256
    n_hidden_2 = 128
    n_classes = 10

    learning_rate = 0.01  # 学习率
    training_epochs = 20  # 迭代轮数
    batch_size = 256  # 小批量数量大小
    display_epoch = 1
    show_num = 10


    # 第一层输入
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    keep_prob = tf.placeholder(dtype=tf.float32)

    # 第二层输入
    l2x = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_1])
    l2y = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_1])

    # 第三层输入
    l3x = tf.placeholder(dtype=tf.float32, shape=[None, n_hidden_2])
    l3y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

    weights = {
        # 网络1 784-256-256-784
        'l1_h1': tf.Variable(tf.truncated_normal(shape=[n_input, n_hidden_1], stddev=0.1)),  # 级联使用
        'l1_h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_1], stddev=0.1)),
        'l1_out': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_input], stddev=0.1)),
        # 网络2 256-128-128-256
        'l2_h1': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),  # 级联使用
        'l2_h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_2], stddev=0.1)),
        'l2_out': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_1], stddev=0.1)),
        # 网络3 128-10
        'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_classes], stddev=0.1))  # 级联使用
    }

    biases = {
        # 网络1 784-256-256-784
        'l1_b1': tf.Variable(tf.zeros(shape=[n_hidden_1])),  # 级联使用
        'l1_b2': tf.Variable(tf.zeros(shape=[n_hidden_1])),
        'l1_out': tf.Variable(tf.zeros(shape=[n_input])),
        # 网络2 256-128-128-256
        'l2_b1': tf.Variable(tf.zeros(shape=[n_hidden_2])),  # 级联使用
        'l2_b2': tf.Variable(tf.zeros(shape=[n_hidden_2])),
        'l2_out': tf.Variable(tf.zeros(shape=[n_hidden_1])),
        # 网络3 128-10
        'out': tf.Variable(tf.zeros(shape=[n_classes]))  # 级联使用
    }

    '''
    定义第一层网络结构  
    注意：在第一层里加入噪声，并且使用弃权层 784-256-256-784
    '''
    l1_h1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['l1_h1']), biases['l1_b1']))
    l1_h1_dropout = tf.nn.dropout(l1_h1, keep_prob)
    l1_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l1_h1_dropout, weights['l1_h2']), biases['l1_b2']))
    l1_h2_dropout = tf.nn.dropout(l1_h2, keep_prob)
    l1_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l1_h2_dropout, weights['l1_out']), biases['l1_out']))

    # 计算代价
    l1_cost = tf.reduce_mean((l1_reconstruction - y) ** 2)

    # 定义优化器
    l1_optm = tf.train.AdamOptimizer(learning_rate).minimize(l1_cost)

    '''
    定义第二层网络结构256-128-128-256
    '''
    l2_h1 = tf.nn.sigmoid(tf.add(tf.matmul(l2x, weights['l2_h1']), biases['l2_b1']))
    l2_h2 = tf.nn.sigmoid(tf.add(tf.matmul(l2_h1, weights['l2_h2']), biases['l2_b2']))
    l2_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(l2_h2, weights['l2_out']), biases['l2_out']))

    # 计算代价
    l2_cost = tf.reduce_mean((l2_reconstruction - l2y) ** 2)

    # 定义优化器
    l2_optm = tf.train.AdamOptimizer(learning_rate).minimize(l2_cost)

    '''
    定义第三层网络结构 128-10
    '''
    l3_logits = tf.add(tf.matmul(l3x, weights['out']), biases['out'])

    # 计算代价
    l3_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l3_logits, labels=l3y))

    # 定义优化器
    l3_optm = tf.train.AdamOptimizer(learning_rate).minimize(l3_cost)

    '''
    定义级联级网络结构

    将前三个网络级联在一起，建立第四个网络，并定义网络结构
    '''
    # 1 联 2
    l1_l2_out = tf.nn.sigmoid(tf.add(tf.matmul(l1_h1, weights['l2_h1']), biases['l2_b1']))

    # 2 联 3
    logits = tf.add(tf.matmul(l1_l2_out, weights['out']), biases['out'])

    # 计算代价
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=l3y))

    # 定义优化器
    optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    num_batch = int(np.ceil(mnist.train.num_examples / batch_size))

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
                _, loss = sess.run([l1_optm, l1_cost], feed_dict={x: batch_x_noise, y: batch_x, keep_prob: 0.5})
                total_cost += loss

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))

        print('训练完成')

        # 数据可视化
        test_noisy = mnist.test.images[:show_num] + 0.3 * np.random.randn(show_num, 784)
        reconstruction = sess.run(l1_reconstruction, feed_dict={x: test_noisy, keep_prob: 1.0})
        plt.figure(figsize=(1.0 * show_num, 1 * 2))
        for i in range(show_num):
            # 原始图像
            plt.subplot(3, show_num, i + 1)
            plt.imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
            plt.axis('off')
            # 加入噪声后的图像
            plt.subplot(3, show_num, i + show_num * 1 + 1)
            plt.imshow(np.reshape(test_noisy[i], (28, 28)), cmap='gray')
            plt.axis('off')
            # 去燥自编码器输出图像
            plt.subplot(3, show_num, i + show_num * 2 + 1)
            plt.imshow(np.reshape(reconstruction[i], (28, 28)), cmap='gray')
            plt.axis('off')
        plt.show()

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
                l1_out = sess.run(l1_h1, feed_dict={x: batch_x, keep_prob: 1.0})

                _, loss = sess.run([l2_optm, l2_cost], feed_dict={l2x: l1_out, l2y: l1_out})
                total_cost += loss

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))

        print('训练完成')

        # 数据可视化
        testvec = mnist.test.images[:show_num]
        l1_out = sess.run(l1_h1, feed_dict={x: testvec, keep_prob: 1.0})
        reconstruction = sess.run(l2_reconstruction, feed_dict={l2x: l1_out})
        plt.figure(figsize=(1.0 * show_num, 1 * 2))
        for i in range(show_num):
            # 原始图像
            plt.subplot(3, show_num, i + 1)
            plt.imshow(np.reshape(testvec[i], (28, 28)), cmap='gray')
            plt.axis('off')
            # 加入噪声后的图像
            plt.subplot(3, show_num, i + show_num * 1 + 1)
            plt.imshow(np.reshape(l1_out[i], (16, 16)), cmap='gray')
            plt.axis('off')
            # 去燥自编码器输出图像
            plt.subplot(3, show_num, i + show_num * 2 + 1)
            plt.imshow(np.reshape(reconstruction[i], (16, 16)), cmap='gray')
            plt.axis('off')
        plt.show()

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
                l1_out = sess.run(l1_h1, feed_dict={x: batch_x, keep_prob: 1.0})
                l2_out = sess.run(l2_h1, feed_dict={l2x: l1_out})
                _, loss = sess.run([l3_optm, l3_cost], feed_dict={l3x: l2_out, l3y: batch_y})
                total_cost += loss

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))


        print('训练完成')

        # 数据可视化
        testvec = mnist.test.images[:show_num]
        l1_out = sess.run(l1_h1, feed_dict={x: testvec, keep_prob: 1.0})
        reconstruction = sess.run(l2_reconstruction, feed_dict={l2x: l1_out})
        plt.figure(figsize=(1.0 * show_num, 1 * 2))
        for i in range(show_num):
            # 原始图像
            plt.subplot(3, show_num, i + 1)
            plt.imshow(np.reshape(testvec[i], (28, 28)), cmap='gray')
            plt.axis('off')
            # 加入噪声后的图像
            plt.subplot(3, show_num, i + show_num * 1 + 1)
            plt.imshow(np.reshape(l1_out[i], (16, 16)), cmap='gray')
            plt.axis('off')
            # 去燥自编码器输出图像
            plt.subplot(3, show_num, i + show_num * 2 + 1)
            plt.imshow(np.reshape(reconstruction[i], (16, 16)), cmap='gray')
            plt.axis('off')
        plt.show()

        '''
        栈式自编码网络验证
        '''
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(l3y, 1))
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, l3y: mnist.test.labels}))

    '''
    级联微调
    将网络模型联起来进行分类训练
    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('级联微调 开始训练')
        for epoch in range(training_epochs):
            total_cost = 0.0
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, loss = sess.run([optm, cost], feed_dict={x: batch_x, l3y: batch_y})
                total_cost += loss

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {0}/{1}  average cost {2}'.format(epoch, training_epochs, total_cost / num_batch))

        print('训练完成')
        print('Accuracy:', accuracy.eval({x: mnist.test.images, l3y: mnist.test.labels}))


if __name__ == '__main__':
    stacked_auto_encoder()