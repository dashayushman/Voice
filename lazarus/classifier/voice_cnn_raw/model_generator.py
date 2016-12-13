import tensorflow as tf
from lazarus.utils import dataprep as dp
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.layers.python.layers import layers
import numpy as np

import os
import re
import time, datetime

train_dir = "/home/amit/Desktop/voice/tf_py2.7/checkpoint/cnn"
summary_folder = "/home/amit/Desktop/voice/tf_py2.7/summary/cnn"

n_input = 10  # MNIST data input (img shape: 28*28)
n_steps = 70  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)
batch_size = 20

TOWER_NAME = 'tower'

training_iters = 200000
max_steps = 10000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.
NUM_EPOCHS_PER_DECAY = 100.0  # Epochs after which learning rate decays.
steps_per_checkpoint = 1000

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 160


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/gpu:0'):
        dtype = tf.float32  # if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=dtype)
    return var


# def evaluate(test, shape):
#     testing_iters = len(test.labels)
#
#     config = tf.ConfigProto(log_device_placement= True)
#     # config.gpu_options.per_process_gpu_memory_fraction = 1.0
#     sess = tf.Session(config=config)
#
#     checkpoints_folder = train_dir
#
#     avg_acc = 0.0
#     step = 1
#     while step * batch_size < testing_iters:
#         start_time = time.time()
#         batch_x, batch_y = dp.next_batch(test, batch_size, step)
#
#         batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#         batch_x = tf.cast(batch_x, tf.float32)
#
#         images_eval, labels_eval = batch_x, batch_y
#
#         one_hot_eval_labels = convert_labels_to_one_hot(labels_eval)
#         logits = None
#         top_k_op = None
#
#         logits_eval = gen_model_1(images_eval, is_training)
#         saver = tf.train.Saver(tf.all_variables())
#         if not os.path.exists(checkpoints_folder):
#             os.makedirs(checkpoints_folder)
#         load_model(saver, sess, train_dir)
#
#         acc = gen_accuracy(one_hot_eval_labels, logits_eval)
#
#         acc_val = sess.run([acc])
#         avg_acc += acc_val[0]
#
#         step += 1
#     print('Average Accuracy: ' + str(avg_acc/step))


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32  # if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        # tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
        tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def gen_norm_layer(prev, name, window_size=4):
    norm = tf.nn.lrn(prev, window_size, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                     name=name)
    _activation_summary(norm)
    return norm


def BatchNorm(inputT, is_training, scope=None):
    return tf.cond(is_training,
                   lambda: layers.batch_norm(inputT, is_training=True,
                                 center=False, updates_collections=None, scope=scope, reuse=None, trainable=True),
                   lambda: layers.batch_norm(inputT, is_training=False,
                                 updates_collections=None, center=False, scope=scope, reuse=True, trainable=True))


def gen_conv_layer(prev, name, kernel, stride, is_training, filter_img=False):
    """
    generates conv layer for 1d input e.g. signal
    Args:
      prev 3D Tensor: previous layer tensor
      name string: name to be given to the layer
      kernel: [filter_length, input_channels, output_channels]
      stride integer: number of entries by which the filter is moved right 
      is_training bool: controls training or test version of batch normalization
    Returns:
      1d conv layer
    """
    with tf.variable_scope(name) as scope:
        kernel_conv = _variable_with_weight_decay('weights',
                                                  shape=kernel,
                                                  stddev=5e-2,
                                                  wd=0.0)
        conv = tf.nn.conv1d(prev, kernel_conv, stride, padding='SAME') 
        biases = _variable_on_cpu('biases', [kernel[-1]],
                                  tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        bias = BatchNorm(bias, is_training, scope)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)
        # if filter_img:
    #	_filter_image_summary(name, kernel_conv, kernel[-1])

    return conv


def gen_pool_layer(prev, name, ksize, strides, padding='SAME'):
    """
    generates max_pool layer
    Args:
      prev : A 4-D Tensor with shape [batch, height, width, channels]
      name string: name to be given to the layer
      ksize: The size of the window for each dimension of the input tensor
      strides: The stride of the sliding window for each dimension of the input tensor
      padding: The padding algorithm
    Returns:
      max_pool layer
    """
    pool = tf.nn.max_pool(prev, ksize=ksize, strides=strides, padding=padding,
                          name=name)
    _activation_summary(pool)
    return pool


# In[9]:

def gen_fully_connected_layer(prev, name, shape, do_reshape=False):
    with tf.variable_scope(name) as scope:
        if do_reshape:
            prev = tf.reshape(prev, [batch_size, -1])
            shape[0] = prev.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=shape,
                                              stddev=0.04, wd=0.004)

        biases = _variable_on_cpu('biases', [shape[-1]],
                                  tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(prev, weights) + biases, name=scope.name)
        _activation_summary(fc)
    return fc


# In[10]:

def get_softmax_layer(prev, name, size):
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', size,
                                              stddev=1 / float(size[0]), wd=0.0)
        biases = _variable_on_cpu('biases', [size[-1]],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(prev, weights), biases,
                                name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear

def gen_model_2(gestures, is_training):
    conv1 = gen_conv_layer(gestures, 'conv1', [5, 10, 128], 2, is_training,
                           filter_img=True)
    pool1 = gen_pool_layer([conv1], 'pool1', [1, 1, 3, 3], [1, 1, 2, 1],
                           padding='VALID') # kernel combining 3 channels
    pool1 = tf.squeeze(pool1, [0])

    # need to apply average pool to reduce the number of parameters
    fcn1 = gen_fully_connected_layer(pool1, 'fc1', [None, 1024], do_reshape=True)

    # apply dropout layer

    smx = get_softmax_layer(fcn1, 'softmax', [1024, 10])
    return smx

def gen_model_1(gestures, is_training):
    # something on the lines of alexnet
    conv1 = gen_conv_layer(gestures, 'conv1', [5, 10, 128], 2, is_training,
                           filter_img=True)
    #norm1 = gen_norm_layer([conv1], 'norm1') # check without norm
    pool1 = gen_pool_layer([conv1], 'pool1', [1, 1, 3, 1], [1, 1, 2, 1],
                           padding='VALID')  # used the same ideology of conv1d to conv2d, need to confirm
    pool1 = tf.squeeze(pool1, [0])

    fcn1 = gen_fully_connected_layer(pool1, 'fc1', [None, 1024], do_reshape=True)

    # VGG like 3 * 3 conv stride 1, pad 1 and 2*2 MAX POOL stride 2
    # rather than using fully connected use avg pool across the complete image, works almost as well
    # google net inception module
    # resnet xavier/2, batch norm, l.r. 0.1, weight decay 1e-5, no dropout

    # conv2 = gen_conv_layer(pool1, 'conv2', [6, 100, 256], 4)
    # norm2 = gen_norm_layer([conv2], 'norm2')
    # pool2 = gen_pool_layer(norm2, 'pool2', [1, 1, 3, 1], [1, 1, 2, 1],
    #                       padding='VALID')

    # tf.squeeze(pool2, [0])
    # till now code modified to work with conv1d

    # conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 256, 384], [1, 1, 1, 1])
    # conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
    # conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

    # pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
    #                      padding='VALID')

    # fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
    # fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [4096, 1000], do_reshape=False)

    smx = get_softmax_layer(fcn1, 'softmax', [1024, 10])
    return smx


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def lossfnc(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def trainfnc(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad:
    #         tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def load_model(saver, sess, chkpnts_dir):
    ckpt = tf.train.get_checkpoint_state(chkpnts_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading previously trained model: {}".format(
            ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Training with fresh parameters")
        sess.run(tf.initialize_all_variables())


def _generate_image_and_label_batch(image, label,
                                    batch_size, shuffle=False):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def gen_accuracy(gt_labels, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(gt_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def convert_labels_to_one_hot(labels):
    sparse_labels = tf.reshape(labels, [-1, 1])
    derived_size = tf.shape(labels)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    indices = tf.cast(indices, tf.int64)
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, n_classes])
    outshape = tf.cast(outshape, tf.int64)
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    return labels


def train(train, test, shape, fold):
    """Train CIFAR-10 for a number of steps."""
    global NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = shape[0]
    global train_dir
    global summary_folder
    train_dir += os.sep + fold
    summary_folder += os.sep + fold


    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        '''
        gestures = train.instances
        gestures = np.array(gestures, dtype=np.float32)
        gestures = gestures.reshape(shape)

        labels = train.labels

        #gesture_batch, label_batch = tf.train.batch(
        #    [gestures, labels],
        #    batch_size=batch_size)
        gesture_batch, label_batch = _generate_image_and_label_batch(gestures[0],labels[0],batch_size)
        '''
        # Display the training images in the visualizer.
        # tf.image_summary('images', images)

        # label_batch1 =  tf.reshape(label_batch, [batch_size])
        n_input = shape[2]  # MNIST data input (img shape: 28*28)
        n_steps = shape[1]  # timesteps

        x = tf.placeholder("float", [batch_size, n_steps, n_input])
        y = tf.placeholder("float", [batch_size])
        is_training = tf.placeholder(tf.bool)

        logits = gen_model_1(x, is_training)

        loss = lossfnc(logits, y)
        train_op = trainfnc(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3 # limit on gpu usage

        checkpoints_folder = train_dir
        sess = tf.Session(config=config)
        # create summary operation
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(summary_folder, sess.graph)

        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        load_model(saver, sess, train_dir)

        step = 1
        while step * batch_size < training_iters:
            start_time = time.time()
            batch_x, batch_y = dp.next_batch(train, batch_size, step)

            batch_x = batch_x.reshape((batch_size, n_steps, n_input))



            _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y, is_training: True})

            duration = time.time() - start_time

            assert not np.isnan(
                loss_value), 'Model diverged with loss = NaN'

            if step % steps_per_checkpoint == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = (
                    '%s: train iteration %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print(format_str % (
                    datetime.datetime.now(), step * batch_size, loss_value,
                    examples_per_sec, sec_per_batch))

            if step % steps_per_checkpoint == 0:
                summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y, is_training: True})
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % steps_per_checkpoint == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir,
                                               'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1

        # evaluation on training data
        train_size = len(test.labels)
        avg_acc = 0.0
        step = 1
        while step * batch_size < train_size:
            batch_x, batch_y = dp.next_batch(train, batch_size, step)

            batch_x = batch_x.reshape((batch_size, n_steps, n_input))

            images_eval, labels_eval = batch_x, batch_y

            one_hot_eval_labels = convert_labels_to_one_hot(labels_eval)

            acc = gen_accuracy(one_hot_eval_labels, logits)

            acc_val = sess.run([acc], feed_dict={x: batch_x, y: batch_y, is_training: False})
            avg_acc += acc_val[0]

            step += 1
        print('Training Average Accuracy fold' + fold + ': ' + str(avg_acc / step))



        # evaluation on test data
        testing_iters = len(test.labels)

        avg_acc = 0.0
        step = 1
        while step * batch_size < testing_iters:
            start_time = time.time()
            batch_x, batch_y = dp.next_batch(test, batch_size, step)

            batch_x = batch_x.reshape((batch_size, n_steps, n_input))

            images_eval, labels_eval = batch_x, batch_y

            one_hot_eval_labels = convert_labels_to_one_hot(labels_eval)
            top_k_op = None

            acc = gen_accuracy(one_hot_eval_labels, logits)

            acc_val = sess.run([acc], feed_dict={x: batch_x, y: batch_y, is_training: False})
            avg_acc += acc_val[0]

            step += 1
        print('Test Average Accuracy fold' + fold + ': ' + str(avg_acc / step))
