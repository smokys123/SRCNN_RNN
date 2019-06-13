import os
import tensorflow as tf
import numpy as np
from PIL import Image


def get_set5_images():
    dir_path = os.path.join(os.getcwd(), 'SR_dataset', 'val_Set5')
    image_file_list = os.listdir(dir_path)
    image_list = []
    for image_name in image_file_list:
        file_path = os.path.join(dir_path, image_name)
        image_list.append(file_path)
    return image_list


def _val_image_function(image_path):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    grayscaled_image = tf.image.rgb_to_grayscale(image_decoded)
    return grayscaled_image

#global_step = tf.Variable(0,trainable=False, name='global_step')
val_dataset = tf.data.Dataset.from_tensor_slices(get_set5_images())
val_dataset = val_dataset.map(_val_image_function)

val_dataset = val_dataset.batch(1).repeat()
val_iterator = val_dataset.make_initializable_iterator()
val_image_stacked = val_iterator.get_next()
next_val_images = val_iterator.get_next()

# placeholder
X = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input_images')
val_X = tf.placeholder(tf.float32, [None, None, None, 1], name='val_input_images')
Y = tf.placeholder(tf.float32, [None, 32, 32, 1], name='output_images')


def rnn_model(x, y, h):
    with tf.variable_scope('srcnn_rnn1', reuse=tf.AUTO_REUSE):
        inputs = tf.concat([x, y], 3)
        z_conv = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[3, 3], padding='SAME')
        hidden = tf.nn.tanh(z_conv + h)
        h_conv = tf.layers.conv2d(inputs=hidden, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        y_conv = tf.layers.conv2d(inputs=h_conv, filters=1, kernel_size=[3, 3], padding='SAME')
    return y_conv, hidden


with tf.name_scope('validation'):  # loss
    val_rnn_y1, val_rnn_h1 = rnn_model(val_X, val_X, 0)
    val_rnn_y2, val_rnn_h2 = rnn_model(val_X, val_rnn_y1, val_rnn_h1)
    val_rnn_y3, val_rnn_h3 = rnn_model(val_X, val_rnn_y2, val_rnn_h2)


# Image와 Label 하나 열어보기
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(val_iterator.initializer)
    global_step = 0
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
