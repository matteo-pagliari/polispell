import tensorflow as tf
import utils
import sys
import codecs
from seq2seq_model import Seq2SeqModel
from data_iterator import TextIterator, CorpusIterator
from utils import prepare_batch
from collections import OrderedDict

# Train parameters
tf.app.flags.DEFINE_string('source_vocabulary', '../text/ita_err.unique.json', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', '../text/ita.unique.json', 'Path to target vocabulary')
tf.app.flags.DEFINE_integer('num_encoder_symbols', 30000, 'Source vocabulary size')

tf.app.flags.DEFINE_integer('beam_width', 12, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 5, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 500, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', '../model/correction.ckpt', 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('decode_input', '../text/ita_err.mini.test', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', '../text/ita_err.mini.correct', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

source_vocabulary = '../text/ita_err.unique.json'
target_vocabulary = '../text/ita.unique.json'

def load_model(session, config):

    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print 'Reloading model parameters..'
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():

    # TODO: Config has to be taken from train
    config = OrderedDict(FLAGS.__flags.items())

    '''
    test_set = TextIterator(source=FLAGS.decode_input, batch_size=FLAGS.decode_batch_size,
                            source_dict=source_vocabulary, maxlen=None,
                            n_words_source=30000)
    '''

    test_set = CorpusIterator(sourcepath='../text/validation', batch_size=FLAGS.decode_batch_size,
                              source_vocabulary=source_vocabulary)
    
    # print 'CHECK', test_set.source_dict['motivazioni']


    target_inverse_dict = utils.load_inverse_dict(target_vocabulary)


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        model = load_model(sess, config)

        try:
            print 'Decoding {}..'.format(FLAGS.decode_input)

            fout = codecs.open(FLAGS.decode_output, 'w', 'utf8')


            for text in test_set:   # For multiple text
                print 'Text: ', text.source
                for idx, source_seq in enumerate(text):
                    source, source_len = prepare_batch(source_seq)
                    predicted_ids = model.predict(sess, encoder_inputs=source, encoder_inputs_length=source_len)
                    for seq in predicted_ids:
                        # print utils.seq2words(seq, target_inverse_dict)
                        # print len(utils.seq2words(seq, target_inverse_dict).split())
                
                        # TODO: Check if it writes the sentences in the correct order
                        fout.write(utils.seq2words(seq, target_inverse_dict) + '\n')
                
                    # if not FLAGS.write_n_best:
                        # break
                
                    print '  {}th line decoded'.format((idx+1) * FLAGS.decode_batch_size)
                
                print 'Decoding terminated'

        except IOError:
            pass
        finally:
            fout.close()



def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
