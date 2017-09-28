#Thanks to the authors of the below repositories, from which
#most of this code was taken:

# 1.  https://github.com/igormq/ctc_tensorflow_example
# 2.  https://github.com/tbornt/phoneme_ctc
# 3.  https://github.com/mitochrome/complex-gestures-demo



import time
import logging
import os
logging.basicConfig(level=logging.INFO)

import tensorflow as tf
import numpy as np
#from RWACell import RWACell
import strokes_pb2 as strokes_pb2
from coremltools.models import MLModel
from coremltools.models.neural_network import NeuralNetworkBuilder
import coremltools.models.datatypes as datatypes

from utils import pad_sequences, sparse_tuple_from

num_classes = 64#uppercase, lowercase,space, + blank for CTC
num_features = 9


num_layers = 1 # number of layers in lstm stack
num_hidden = 100
num_epochs = 100
batch_size = 10
clip_thresh = 10000
momentum = 0.9
learning_rate = 0.001
#learning_rate = 0.0001

class Env:
    pass


def load_ipad_data(data_file_path):
    f = open(data_file_path, 'rb')
    training_set = strokes_pb2.TrainingSet()
    training_set.ParseFromString(f.read())
    f.close()
    data = [[[smpl.location_x,smpl.location_y,smpl.timeOffset,smpl.force,smpl.altitude,smpl.azimuth,smpl.strokeIndex,1 if smpl.coalesced else 0,1 if smpl.predicted else 0] for smpl in ex.strokeSamples] for ex in training_set.examples]
    labels = [ex.labels for ex in training_set.examples]
    return np.asarray(data), np.asarray(labels)

def train_model(ENV, in_file, op_file):

    graph = tf.Graph()
    with graph.as_default():
        stacked_layers = {}

        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features])

        targets = tf.sparse_placeholder(tf.int32)
        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None])

        # Weights & biases
        weight_classes = tf.Variable(tf.truncated_normal([num_hidden, num_classes],
                                                         mean=0, stddev=0.1,
                                                         dtype=tf.float32))
        bias_classes = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)


        #_activation = tf.nn.relu#this was causing the model to diverge
        _activation = None

        layers = {'forward':[],'backward':[]}
        for key in layers.keys():
            for i in range(num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_hidden,
                                               use_peepholes=True,
                                               activation=_activation,
                                               state_is_tuple=True,
                                               cell_clip=clip_thresh)
                #
                #cell = RWACell(num_units=num_hidden)
                layers[key].append(cell)
            stacked_layers[key] = tf.nn.rnn_cell.MultiRNNCell(layers[key],
                                                      state_is_tuple=True)





        outputs, bilstm_vars = tf.nn.bidirectional_dynamic_rnn(stacked_layers['forward'],
                                                               stacked_layers['backward'],
                                                               inputs,
                                                               sequence_length=seq_len,
                                                               time_major=False, # [batch_size, max_time, num_hidden]
                                                               dtype=tf.float32)


        """
        outputs_concate = tf.concat_v2(outputs, 2)
        outputs_concate = tf.reshape(outputs_concate, [-1, 2*num_hidden])
        # logits = tf.matmul(outputs_concate, weight_classes) + bias_classes
        """
        fw_output = tf.reshape(outputs[0], [-1, num_hidden])
        bw_output = tf.reshape(outputs[1], [-1, num_hidden])
        logits = tf.add(tf.add(tf.matmul(fw_output, weight_classes), tf.matmul(bw_output, weight_classes)), bias_classes)

        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        loss = tf.nn.ctc_loss(targets,logits,seq_len,time_major=False)
        error = tf.reduce_mean(loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(error)

        # Evaluating
        # decoded, log_prob = ctc_ops.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len)
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    data, labels = load_ipad_data(in_file)
    bound = ((3*len(data)/batch_size)/4)*batch_size
    train_inputs = data[0:bound]
    train_labels = labels[0:bound]
    test_data = data[bound:]
    test_labels = labels[bound:]
    num_examples = len(train_inputs)
    num_batches_per_epoch = num_examples / batch_size


    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

        ckpt = tf.train.get_checkpoint_state(op_file)
        if ckpt:
            logging.info('load', ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            logging.info("no previous session to load")

        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                # Getting the index
                indices = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]

                batch_train_inputs = train_inputs[indices]
                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparse_tuple_from(train_labels[indices])

                feed = {inputs: batch_train_inputs,
                        targets: batch_train_targets,
                        seq_len: batch_train_seq_len}
                batch_cost, _ = session.run([error, optimizer], feed)
                train_cost += batch_cost * batch_size
                train_ler += session.run(label_error_rate, feed_dict=feed) * batch_size
                log = "Epoch {}/{}, iter {}, batch_cost {}"
                logging.info(log.format(curr_epoch + 1, num_epochs, batch, batch_cost))

            saver.save(session, os.path.join(ENV.output, 'best.ckpt'), global_step=curr_epoch)



            # Shuffle the data
            shuffled_indexes = np.random.permutation(num_examples)
            train_inputs = train_inputs[shuffled_indexes]
            train_labels = train_labels[shuffled_indexes]

            # Metrics mean
            train_cost /= num_examples
            train_ler /= num_examples

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            logging.info(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler, time.time() - start))


            #run the test data through
            indices = [i % len(test_data) for i in range(batch * batch_size, (batch + 1) * batch_size)]
            test_inputs = test_data[indices]
            test_inputs, test_seq_len = pad_sequences(test_inputs)
            test_targets = sparse_tuple_from(test_labels[indices])
            feed_test = {
                inputs: test_inputs,
                targets: test_targets,
                seq_len: test_seq_len
            }
            test_cost, test_ler = session.run([error, label_error_rate], feed_dict=feed_test)
            log = "Epoch {}/{}, test_cost {}, test_ler {}"
            logging.info(log.format(curr_epoch + 1, num_epochs, test_cost, test_ler))

        input_features = [('strokeData', datatypes.Array(num_features))]
        output_features = [('labels', datatypes.Array(num_classes))]

        vars = tf.trainable_variables()
        weights = {'forward':{}, 'backward': {}}
        for _var in vars:
            name = _var.name.encode('utf-8')
            if name.startswith('bidirectional_rnn/fw'):
                key = name.replace('bidirectional_rnn/fw/', '')
                key = key.replace('multi_rnn_cell/cell_0/lstm_cell/','')
                key = key.replace(':0', '')
                weights['forward'][key] = _var.eval()
            else:
                key = name.replace('bidirectional_rnn/bw/', '')
                key = key.replace('multi_rnn_cell/cell_0/lstm_cell/', '')
                key = key.replace(':0','')
                weights['backward'][key] = _var.eval()


    builder = NeuralNetworkBuilder(input_features,output_features,mode=None)

    fw_biases = [weights['forward']['bias'][0*num_hidden:1*num_hidden],
                 weights['forward']['bias'][1*num_hidden:2*num_hidden],
                 weights['forward']['bias'][2*num_hidden:3*num_hidden],
                 weights['forward']['bias'][3*num_hidden:4*num_hidden]]

    bw_biases = [weights['backward']['bias'][0*num_hidden:1*num_hidden],
                 weights['backward']['bias'][1*num_hidden:2*num_hidden],
                 weights['backward']['bias'][2*num_hidden:3*num_hidden],
                 weights['backward']['bias'][3*num_hidden:4*num_hidden]]

    num_LSTM_gates = 5

    input_weights = {
        'forward': np.zeros((num_LSTM_gates - 1,num_hidden,num_features)),
        'backward': np.zeros((num_LSTM_gates - 1,num_hidden,num_features))
    }

    recurrent_weights = {
        'forward': np.zeros((num_LSTM_gates - 1,num_hidden,num_hidden)),
        'backward': np.zeros((num_LSTM_gates - 1,num_hidden, num_hidden))
    }


    builder.add_bidirlstm(name='bidirectional_1',
                          W_h=recurrent_weights['forward'],
                          W_x=input_weights['forward'],
                          b=fw_biases,
                          W_h_back=recurrent_weights['backward'],
                          W_x_back=input_weights['backward'],
                          b_back=bw_biases,
                          hidden_size=num_hidden,
                          input_size=num_features,

                          input_names=['strokeData',
                                       'bidirectional_1_h_in',
                                       'bidirectional_1_c_in',
                                       'bidirectional_1_h_in_rev',
                                       'bidirectional_1_c_in_rev'],
                          output_names=['y',
                                        'bidirectional_1_h_out',
                                        'bidirectional_1_c_out',
                                        'bidirectional_1_h_out_rev',
                                        'bidirectional_1_c_out_rev'],
                          peep=[weights['forward']['w_i_diag'],
                                weights['forward']['w_f_diag'],
                                weights['forward']['w_o_diag']],
                          peep_back=[weights['backward']['w_i_diag'],
                                     weights['backward']['w_f_diag'],
                                     weights['backward']['w_o_diag']],
                          cell_clip_threshold=clip_thresh)

    builder.add_softmax(name='softmax', input_name='y', output_name='labels')

    optional_inputs = [('bidirectional_1_h_in', num_hidden),
                       ('bidirectional_1_c_in', num_hidden),
                       ('bidirectional_1_h_in_rev', num_hidden),
                       ('bidirectional_1_c_in_rev', num_hidden)]
    optional_outputs = [('bidirectional_1_h_out', num_hidden),
                        ('bidirectional_1_c_out', num_hidden),
                        ('bidirectional_1_h_out_rev', num_hidden),
                        ('bidirectional_1_c_out_rev', num_hidden)]

    #not really sure what this line belowe does, just copied it from the Keras converter in coremltools,
    # and it seemed to make things work
    builder.add_optionals(optional_inputs, optional_outputs)

    model = MLModel(builder.spec)

    model.short_description = 'Model for recognizing a symbols and diagrams drawn on ipad screen with apple pencil'

    model.input_description['strokeData'] = 'A collection of strokes to classify'
    model.output_description['labels'] = 'The "probability" of each label, in a dense array'

    outfile = 'bilstm.mlmodel'
    model.save(outfile)

    print('Saved to file: %s' % outfile)


import click

@click.command()
@click.argument('in-file')
@click.option('--out-file')
def main(in_file,out_file):
    _env = Env()
    _env.output = "bilstm.checkpoints"
    #_env.output = "rwa.checkpoints"
    train_model(_env,
                in_file,
                out_file)


if __name__ == '__main__':
    main()
