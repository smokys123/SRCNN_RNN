import os
import tensorflow as tf
import numpy as np
from PIL import Image

def get_images():
    dir_path = os.path.join(os.getcwd(), 'SR_dataset', '291')
    image_file_list = os.listdir(dir_path)
    image_list = []
    for image_name in image_file_list:
        file_path = os.path.join(dir_path, image_name)
        image_list.append(file_path)
    # make train , test dataset
    train_image_list = image_list[:-35]
    test_image_list = image_list[-35:]
    return train_image_list, test_image_list

def get_set5_images():
    dir_path = os.path.join(os.getcwd(), 'SR_dataset', 'val_Set5')
    image_file_list = os.listdir(dir_path)
    image_list = []
    for image_name in image_file_list:
        file_path = os.path.join(dir_path, image_name)
        image_list.append(file_path)
    return image_list


def _crop_grayscale_function(image_path, label):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    grayscaled_image = tf.image.rgb_to_grayscale(image_decoded)
    input_image = tf.image.random_crop(grayscaled_image, [32, 32, 1])
    resized_image = tf.image.resize_images(images=input_image, size=(16, 16), method=tf.image.ResizeMethod.BILINEAR)
    label_image = tf.image.resize_images(images=resized_image, size=(32, 32), method=tf.image.ResizeMethod.BILINEAR)
    return input_image, label_image


def _val_image_function(image_path):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    grayscaled_image = tf.image.rgb_to_grayscale(image_decoded)
    return grayscaled_image

#global_step = tf.Variable(0,trainable=False, name='global_step')
train_input_list, test_input_list = get_images()
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_list, train_input_list))
test_dataset = tf.data.Dataset.from_tensor_slices((test_input_list, test_input_list))
val_dataset = tf.data.Dataset.from_tensor_slices(get_set5_images())


train_dataset = train_dataset.map(_crop_grayscale_function)
test_dataset = test_dataset.map(_crop_grayscale_function)
val_dataset = val_dataset.map(_val_image_function)

#dataset = dataset.repeat()
train_dataset = train_dataset.batch(128).repeat()
test_dataset = test_dataset.batch(35).repeat()
val_dataset = val_dataset.batch(1).repeat()

train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

train_image_stacked, train_label_stacked = train_iterator.get_next()
test_image_stacked, test_label_stacked = test_iterator.get_next()
val_image_stacked = val_iterator.get_next()

next_train_images, next_train_labels = train_iterator.get_next()
next_test_images, next_test_labels = test_iterator.get_next()
next_val_images = val_iterator.get_next()

# placeholder
X = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input_images')
val_X = tf.placeholder(tf.float32, [None, None, None, 1], name='val_input_images')
Y = tf.placeholder(tf.float32, [None, 32, 32, 1], name='output_images')



def rnn_model(x, y, h):
    with tf.variable_scope('srcnn_rnn1', reuse=tf.AUTO_REUSE):
        """
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
                                 padding='SAME', name='conv1')
        z_conv = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding='SAME')
        hidden = tf.nn.tanh(z_conv + h)
        h_conv = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=[3, 3], padding='SAME')
        conv2 = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=[3, 3],
                                 padding='SAME', activation=tf.nn.relu, name='conv2')
        conv3 = tf.layers.conv2d(inputs=conv2, filters=1, kernel_size=[3, 3], padding='SAME')
        """
        inputs = tf.concat([x, y], 3)
        z_conv = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], padding='SAME')
        hidden = tf.nn.tanh(z_conv + h)
        h_conv = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        y_conv = tf.layers.conv2d(inputs=h_conv, filters=1, kernel_size=[3, 3], padding='SAME')
    return y_conv, hidden


with tf.name_scope('optimizer'):  # loss
    rnn_y1, rnn_h1 = rnn_model(X, X, 0)
    rnn_y2, rnn_h2 = rnn_model(X, rnn_y1, rnn_h1)
    rnn_y3, rnn_h3 = rnn_model(X, rnn_y2, rnn_h2)

    val_rnn_y1, val_rnn_h1 = rnn_model(val_X, val_X, 0)
    val_rnn_y2, val_rnn_h2 = rnn_model(val_X, val_rnn_y1, val_rnn_h1)
    val_rnn_y3, val_rnn_h3 = rnn_model(val_X, val_rnn_y2, val_rnn_h2)

    cost = tf.reduce_mean(tf.square(rnn_y1-Y))
    cost += tf.reduce_mean(tf.square(rnn_y2-Y))
    cost += tf.reduce_mean(tf.square(rnn_y3-Y))
    # optimizer
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope('psnr'):
    out = tf.convert_to_tensor(rnn_y3, dtype=tf.float32)
    psnr_acc = tf.image.psnr(Y, out, max_val=255)
    mean_psnr = tf.reduce_mean(psnr_acc)
    tf.summary.scalar('psnr', mean_psnr)


# Image와 Label 하나 열어보기
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    sess.run(val_iterator.initializer)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    global_step = 0
    for e in range(1):
        total_cost = 0
        for i in range(2):
            input_images, output_images = sess.run([next_train_images, next_train_labels])
            _, cost_val = sess.run([train_op, cost], feed_dict={X: input_images, Y: output_images})
            total_cost += cost_val

        if e % 100 == 0:
            test_input_images, test_output_images = sess.run([next_test_images, next_test_labels])
            psnr_sum = sess.run(mean_psnr, feed_dict={X: test_input_images, Y: test_output_images})
            summary = sess.run(merged, feed_dict={X: test_input_images, Y: test_output_images})
            writer.add_summary(summary, global_step)
            global_step += 1
            print('epoch: ', '%d'%(e+1),
                  'avg_cost: ', '{:.3f}'.format(total_cost/128),
                  'psnr: ', '{:0.3f}'.format(psnr_sum))

        if e % 1000 == 0:
            saver.save(sess, './model/cnn.ckpt', global_step)

    for i in range(5):
        val_input_images = sess.run(next_val_images)
        set5_outputs = sess.run(val_rnn_y3, feed_dict={val_X: val_input_images})  # batch=1, 512, 512, rgb=1
        set5_outputs = np.array(set5_outputs)
        w = set5_outputs.shape[1]
        h = set5_outputs.shape[2]
        set5_outputs = np.array(set5_outputs).reshape((w, h))
        im = Image.fromarray(set5_outputs, 'L')
        im_path = os.path.join(os.getcwd(), 'SR_dataset', 'val_output_Set5', str(i)+'.png')
        im.save(im_path)
