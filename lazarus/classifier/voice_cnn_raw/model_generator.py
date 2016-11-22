import tensorflow as tf
from lazarus.utils import dataprep as dp
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import os
import time, datetime


train_dir = "/home/amit/Desktop/voice/tf_py2.7/checkpoint"

n_input = 10  # MNIST data input (img shape: 28*28)
n_steps = 68  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)
batch_size = 20

training_iters = 30000
max_steps = 3000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
steps_per_checkpoint = 100

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0

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
        dtype = tf.float32 # if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer,
                              dtype=dtype)
    return var


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
    # _activation_summary(norm)
    return norm


def gen_conv_layer(prev, name, kernel, stride, filter_img=False):
    with tf.variable_scope(name) as scope:
        kernel_conv = _variable_with_weight_decay('weights',
                                                  shape=kernel,
                                                  stddev=5e-2,
                                                  wd=0.0)
        conv = tf.nn.conv1d(prev, kernel_conv, stride, padding='SAME')
        biases = _variable_on_cpu('biases', [kernel[-1]],
                                  tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        # _activation_summary(conv)
        # if filter_img:
    #	_filter_image_summary(name, kernel_conv, kernel[-1])

    return conv


def gen_pool_layer(prev, name, ksize, strides, padding='SAME'):
    pool = tf.nn.max_pool(prev, ksize=ksize, strides=strides, padding=padding,
                          name=name)
    # _activation_summary(pool)
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
        # _activation_summary(fc)
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
        # _activation_summary(softmax_linear)

    return softmax_linear


def gen_model_1(gestures):
    conv1 = gen_conv_layer(gestures, 'conv1', [6, 10, 100], 4,
                           filter_img=True)
    norm1 = gen_norm_layer([conv1], 'norm1')
    pool1 = gen_pool_layer(norm1, 'pool1', [1, 1, 3, 1], [1, 1, 2, 1],
                           padding='VALID') # used the same ideology of conv1d to conv2d, need to confirm
    pool1 = tf.squeeze(pool1, [0])

    #conv2 = gen_conv_layer(pool1, 'conv2', [6, 100, 256], 4)
    #norm2 = gen_norm_layer([conv2], 'norm2')
    #pool2 = gen_pool_layer(norm2, 'pool2', [1, 1, 3, 1], [1, 1, 2, 1],
    #                       padding='VALID')

    #tf.squeeze(pool2, [0])
    #till now code modified to work with conv1d

    #conv3 = gen_conv_layer(pool2, 'conv3', [3, 3, 256, 384], [1, 1, 1, 1])
    #conv4 = gen_conv_layer(conv3, 'conv4', [3, 3, 384, 384], [1, 1, 1, 1])
    #conv5 = gen_conv_layer(conv4, 'conv5', [3, 3, 384, 256], [1, 1, 1, 1])

    #pool3 = gen_pool_layer(conv5, 'pool3', [1, 3, 3, 1], [1, 2, 2, 1],
    #                      padding='VALID')
    fcn1 = gen_fully_connected_layer(pool1, 'fc1', [None, 4096], do_reshape=True)
    #fcn2 = gen_fully_connected_layer(fcn1, 'fc2', [4096, 4096], do_reshape=False)
    #fcn3 = gen_fully_connected_layer(fcn2, 'fc3', [4096, 1000], do_reshape=False)

    smx = get_softmax_layer(fcn1, 'softmax', [4096, 10])
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
    tf.scalar_summary(l.op.name +' (raw)', l)
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
                                    batch_size, shuffle = False):
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

def train(train, shape):
    """Train CIFAR-10 for a number of steps."""
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = shape[0]

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

        #label_batch1 =  tf.reshape(label_batch, [batch_size])
        n_input = shape[2]  # MNIST data input (img shape: 28*28)
        n_steps = shape[1]  # timesteps

        x = tf.placeholder("float", [batch_size, n_steps, n_input])
        y = tf.placeholder("float", [batch_size])

        logits = gen_model_1(x)

        loss = lossfnc(logits, y)
        train_op = trainfnc(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        config = tf.ConfigProto(log_device_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=.4
        sess = tf.Session(config=config)
        #summary_op = tf.merge_all_summaries()
        checkpoints_folder = train_dir
        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)
        load_model(saver, sess, train_dir)

        # Start the queue runners.
        #tf.train.start_queue_runners(sess=sess)

        #summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

        step = 1
        while step * batch_size < training_iters:
            start_time = time.time()
            batch_x, batch_y = dp.next_batch(train, batch_size, step)

            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})

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
                    datetime.datetime.now(), step*batch_size, loss_value,
                    examples_per_sec, sec_per_batch))

            # if step % steps_per_checkpoint == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % steps_per_checkpoint == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir,
                                               'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            step += 1

# def generateModel(train, test, shape):
#     # Parameters
#     learning_rate = 0.001
#     training_iters = 30000
#     batch_size = 20
#     display_step = 10
#
#     # Network Parameters
#     n_input = shape[2]  # MNIST data input (img shape: 28*28)
#     n_steps = shape[1]  # timesteps
#     n_hidden = 128  # hidden layer num of features
#     n_classes = 10  # MNIST total classes (0-9 digits)
#
#     # tf Graph input
#     x = tf.placeholder("float", [None, n_steps, n_input])
#     y = tf.placeholder("float", [None, n_classes])
#
#     # Define weights
#     weights = {
#         'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
#     }
#     biases = {
#         'out': tf.Variable(tf.random_normal([n_classes]))
#     }
#
#     pred = RNN(x, weights, biases, n_input, n_steps)
#
#     # Define loss and optimizer
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#     # Evaluate model
#     correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#     # Initializing the variables
#     init = tf.initialize_all_variables()
#
#     # Launch the graph
#     with tf.Session() as sess:
#         sess.run(init)
#         step = 1
#         # Keep training until reach max iterations
#         while step * batch_size < training_iters:
#             batch_x, batch_y = train.next_batch(batch_size)
#             # Reshape data to get 28 seq of 28 elements
#             batch_x = batch_x.reshape((batch_size, n_steps, n_input))
#             # Run optimization op (backprop)
#             sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#             if step % display_step == 0:
#                 # Calculate batch accuracy
#                 acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#                 # Calculate batch loss
#                 loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
#                 print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
#                       "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                       "{:.5f}".format(acc))
#             step += 1
#         print("Optimization Finished!")
#
#         # Calculate accuracy for 128 mnist test images
#         # test_len = 128
#         test_data = test.instances.reshape((-1, n_steps, n_input))
#         test_label = test.labels
#         print("Testing Accuracy:",
#               sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
#         sess.close()
#         # print(labels)
#
#
# def RNN(x, weights, biases, n_input, n_steps):
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, n_steps, n_input)
#     # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
#
#     # Permuting batch_size and n_steps
#     x = tf.transpose(x, [1, 0, 2])
#     # Reshaping to (n_steps*batch_size, n_input)
#     x = tf.reshape(x, [-1, n_input])
#     # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#     x = tf.split(0, n_steps, x)
#
#     # Define a lstm cell with tensorflow
#     with tf.variable_scope('forward'):
#         lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
#
#     tf.get_variable_scope().reuse == True
#
#     # Get lstm cell output
#     outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
#
#     # Linear activation, using rnn inner loop last output
#     return tf.matmul(outputs[-1], weights['out']) + biases['out']
