from DenseNet import DenseNet
import tensorflow as tf
import dataset
import os
# 设置调用GPU块数
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# 超参数设置
growth_k = 12
nb_block = 2 # how many (dense block + Transition Layer)
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 16
image_width = 416
image_height = 416
img_channels = 3
class_num = 2
total_epoch = 70
# 定义输入图像的占位符
x = tf.placeholder(tf.float32, shape=[None, image_width, image_height, img_channels], name='inputs')
label = tf.placeholder(tf.float32, shape=[None, class_num], name='label')
training_flag = tf.placeholder(tf.bool, name='flag')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
# 调用 DenseNet 架构
logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits))
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)
# 预测精度
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 模型定义保存
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
# 获取图像数据
train_image_list, train_label_list = dataset.load_satetile_image('./data/train/', dataset='train')
valid_image_list, valid_label_list = dataset.load_satetile_image('./data/valid/', dataset='valid')
# 验证结果
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    for it in range(len(valid_image_list) // batch_size):
        valid_img, valid_label = dataset.batch_image(valid_image_list, valid_label_list, batch_size=batch_size, index=it)
        test_feed_dict = {
            x: valid_img,
            label: valid_label,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }
        _, loss_ = sess.run([train, cost], feed_dict=test_feed_dict)
        acc_ = accuracy.eval(feed_dict=test_feed_dict)
        test_loss += loss_
        test_acc += acc_
        if it == ((len(valid_image_list) // batch_size) - 1):
            test_loss /= (len(valid_image_list) // batch_size)
            test_acc /= (len(valid_image_list) // batch_size)
    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary
# 开始会话
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        # 初始化变量
        sess.run(tf.global_variables_initializer())
    # tensorboard生成
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epoch + 1):
        if epoch == (total_epoch * 0.5) or epoch == (total_epoch * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        train_acc = 0.0
        train_loss = 0.0
        iteration = 20
        for step in range(len(train_image_list) // batch_size):
            # 加载训练集和验证集
            img, img_label = dataset.batch_image(train_image_list, train_label_list, batch_size=batch_size, index=step)
            # print(img.shape, img_label.shape)
            train_feed_dict = {
                x: img,
                label: img_label,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            if step % 20 == 0:
                train_loss /= (step + 1) # average loss
                print('epoch: %d, iteration: %d, train_loss: %.4f' % (epoch, iteration, train_loss))
                iteration += 20
            if step == ((len(train_image_list) // batch_size) - 1):
                train_loss /= (len(train_image_list) // batch_size) # average accuracy
                train_acc /= (len(train_image_list) // batch_size)  # average accuracy
                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                test_acc, test_loss, test_summary = Evaluate(sess)
                #
                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.add_summary(summary=test_summary, global_step=epoch)
                summary_writer.flush()

                line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                    epoch, total_epoch, train_loss, train_acc, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f:
                    f.write(line)
        # 保存训练模型
        saver.save(sess=sess, save_path='./dense.ckpt', global_step=epoch)
