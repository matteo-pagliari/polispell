import os
import time
import json
import sys
import tensorflow as tf
from seq2seq_model import Seq2SeqModel
from collections import OrderedDict
from data_iterator import BiTextIterator
from utils import prepare_train_batch

# Data loading parameters
tf.app.flags.DEFINE_string('source_vocabulary', '../text/ita_err.unique.json', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', '../text/ita.unique.json', 'Path to target vocabulary')
tf.app.flags.DEFINE_string('source_train_data', '../text/ita_err.mini', 'Path to source training data')
tf.app.flags.DEFINE_string('target_train_data', '../text/ita.mini', 'Path to target training data')

# Network parametes
tf.app.flags.DEFINE_integer('hidden_units', 20, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 30000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 30000, 'Target vocabulary size')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 5, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 5, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 1, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('max_seq_length', 50, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 100, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 11500, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1150000, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', '../model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'correction.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')


FLAGS = tf.app.flags.FLAGS


def create_model(session, FLAGS):

    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2SeqModel(config, 'train')

    '''
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print 'Reloading model parameters..'
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print 'Created new model parameters..'
        session.run(tf.global_variables_initializer())
    '''

    # To remove in case of restore training
    session.run(tf.global_variables_initializer())

    return model


def train():

    print 'Load training data'

    # TODO: insert iterator for multiple texts
    train_set = BiTextIterator(source=FLAGS.source_train_data,
                               target=FLAGS.target_train_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               maxlen=FLAGS.max_seq_length,
                               n_words_source=FLAGS.num_encoder_symbols,
                               n_words_target=FLAGS.num_decoder_symbols,
                               shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                               sort_by_length=FLAGS.sort_by_length,
                               maxibatch_size=FLAGS.max_load_batches)


    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        print 'Model created'

        # step_time, loss = 0.0, 0.0
        # words_seen, sents_seen = 0, 0
        start_time = time.time()

        print 'Start Training'
        # Training loop
        for epoch_idx in xrange(FLAGS.max_epochs):
            print 'Epoch %s' %(epoch_idx+1)
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print 'Training is already complete.', \
                    'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs)
                break

            # Debugging
            i = 0

            for source_seq, target_seq in train_set:
                # print i
                source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq,
                                                                             FLAGS.max_seq_length)

                # Debugging
                i+=1

                step_loss, summary = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                 decoder_inputs=target, decoder_inputs_length=target_len)

                print 'Batch_number: %s'%i, 'Step_loss: %s' %step_loss

                log_writer.add_summary(summary, model.global_step.eval())

            model.global_epoch_step_op.eval()

        print 'Saving the model'
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path)
        # To use in case I want the global_step
        # model.save(sess, checkpoint_path, global_step=model.global_step)
        # json.dump(model.config, open('%s.json', checkpoint_path, 'wb'), indent=2)

    print 'Training terminated'



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])










