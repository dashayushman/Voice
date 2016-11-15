import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os

from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops.rnn import bidirectional_rnn
from lazarus.utils import dataprep as dp
import numpy as np


n_input = 10 # 10 mfcc features
n_steps = 68 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
checkpoints_folder = "/home/amit/Desktop/voice/tf_py2.7/checkpoint"
steps_per_checkpoint = 100

def load_model(saver, sess, chkpnts_dir):
	ckpt = tf.train.get_checkpoint_state(chkpnts_dir)
	if ckpt and ckpt.model_checkpoint_path:
		print("Loading previously trained model: {}".format(
			ckpt.model_checkpoint_path))
		saver.restore(sess, ckpt.model_checkpoint_path)
	else:
		print("Training with fresh parameters")
		sess.run(tf.initialize_all_variables())

def generateModel(train,test):

    # learning Parameters
    learningRate = 0.001
    nEpochs = 10
    batchSize = 10
    momentum = 0.9 #for faster convergence and reduced oscillation in gradient descent

    # Network Parameters
    nFeatures = 10  # IMU = acc, gyr, ori raw features
    nHidden = 128
    nClasses = 11  # n_classes, plus the "blank" for CTC


    #batchedData, maxTimeSteps, totalN = dp.load_batched_data(rootdir, batchSize, scaler)
    batchedData, maxTimeSteps = dp.data_lists_to_batches(train[0], train[1], batchSize)
    testbatchedData, testmaxTimeSteps = dp.data_lists_to_batches(test[0], test[1], len(test[1]))
    #batchedData, maxTimeSteps = dp.data_lists_to_batches(train, test, batchSize)
    totalN = len(train[1])

    #ctc
    ####Define graph
    print('Defining graph')
    #graph = tf.Graph()
    #with graph.as_default():
    with tf.device('/gpu:0'):
        ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow

        global_step = tf.Variable(0, trainable=False)
        ####Graph input
        inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
        # Prep input data to fit requirements of rnn.bidirectional_rnn
        #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
        inputXrs = tf.reshape(inputX, [-1, nFeatures])
        #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
        inputList = tf.split(0, maxTimeSteps, inputXrs)
        targetIxs = tf.placeholder(tf.int64)
        targetVals = tf.placeholder(tf.int32)
        targetShape = tf.placeholder(tf.int64)
        targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
        seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

        ####Weights & biases
        weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                       stddev=np.sqrt(2.0 / (2 * nHidden))))
        biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
        weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                       stddev=np.sqrt(2.0 / (2 * nHidden))))
        biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
        weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                         stddev=np.sqrt(2.0 / nHidden)))
        biasesClasses = tf.Variable(tf.zeros([nClasses]))

        ####Network
        forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
        backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
        fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
        fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
        outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

        logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

        ####Optimizing
        logits3d = tf.pack(logits)
        loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
        optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

        ####Evaluating
        logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
        predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])

        # using lavenshtein distance for sequence match
        errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                    tf.to_float(tf.size(targetY.values))

        # using exact match for the sequence matching
        # correct_prediction = tf.equal(predictions, targetY)
        # errorRate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.all_variables())

        if not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)

    ####Run session
    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True)) as session:
        load_model(saver, session, checkpoints_folder)
        for epoch in range(nEpochs):
            #print('Epoch', epoch + 1, '...')
            batchErrors = np.zeros(len(batchedData))
            batchRandIxs = np.random.permutation(len(batchedData))  # randomize batch order
            for batch, batchOrigI in enumerate(batchRandIxs):
                batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                            targetShape: batchTargetShape, seqLengths: batchSeqLengths}
                _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
                #print(np.unique(
                 #   lmt))  # print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
                #if (batch % 1) == 0:
                 #   print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                  #  print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
                batchErrors[batch] = er * len(batchSeqLengths)
            epochErrorRate = batchErrors.sum() / totalN
            #print('Epoch', epoch + 1, 'error rate:', epochErrorRate)

            # Save the model checkpoint periodically.
            if (epoch + 1) % steps_per_checkpoint == 0 or (epoch + 1) == nEpochs:
                checkpoint_path = os.path.join(checkpoints_folder,
                                               'model.ckpt')
                saver.save(session, checkpoint_path, global_step=epoch)

        #  calculate accuracy on kth fold
        batchedData, _ = dp.data_lists_to_batches(test[0], test[1], len(test[1]), maxTimeSteps)
        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[0]
        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
        feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                    targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        er = session.run(errorRate, feed_dict=feedDict)

        error = er * len(batchSeqLengths)
        print('test error: ', error)

    return error
