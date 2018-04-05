import math
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.ops import array_ops

start_token = 0
end_token = 1

class Seq2SeqModel(object):

    def __init__(self, config, mode):

        self.config = config
        self.mode = mode.lower()

        self.encoder_hidden_units = 100
        self.decoder_hidden_units = 100
        self.src_vocab_size = 30000
        self.tgt_vocab_size = 30000
        self.learning_rate = 0.02
        self.input_embedding_size = 100
        self.depth = 2
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
            tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # TODO: Check correct values
        self.attn_input_feeding = True
        self.max_gradient_norm = 1.0
        self.keep_prob = 1.0 - 0.3# - config['dropout_rate']

        if self.mode == 'decode':
            self.max_decode_step = 500


        self.build_model()

    def build_model(self):

        # Building decoder an decoder network
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):

        # [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(shape = (None,None),
                                             dtype=tf.int32, name = 'encoder_inputs')
        # [batch_size]
        self.encoder_inputs_length = tf.placeholder(shape = (None,),
                                                    dtype = tf.int32, name = 'encoder_inputs_length')

        # Dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]

        if self.mode == 'train':

            # Train Mode
            self.decoder_inputs = tf.placeholder(shape = (None,None),
                                                 dtype=tf.int32, name = 'decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(shape = (None,),
                                                        dtype = tf.int32, name = 'decoder_inputs_length')
            
            decoder_start_token = tf.ones(
                    shape=[self.batch_size, 1], dtype=tf.int32) * start_token
            
            decoder_end_token = tf.ones(
                    shape=[self.batch_size, 1], dtype=tf.int32) * end_token
            
            # [batch_size, max_time_steps + 1]
            # Insert start_token
            self.decoder_inputs_train = tf.concat([decoder_start_token, self.decoder_inputs], axis=1)
            
            # [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1
            
            # [batch_size, max_time_steps + 1]
            # Insert end_token at the end
            self.decoder_targets_train = tf.concat([self.decoder_inputs, decoder_end_token], axis=1)


    def build_encoder(self):

        print 'Building Encoder'

        with tf.variable_scope('encoder'):
            self.encoder_cell = self.build_encoder_cell()

            # Initialize encoder_embeddings to have variance=1
            initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype=tf.float32)
            self.encoder_embeddings = tf.get_variable("encoder_embeddings",
                                                      [self.src_vocab_size,self.input_embedding_size],
                                                      initializer=initializer,dtype=tf.float32)

            # [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings,
                                                                  ids=self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            input_layer = Dense(self.encoder_hidden_units, dtype=tf.float32)

            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, dtype=tf.float32,
                time_major=False)


    def build_decoder(self):

        print 'Building Decoder'

        with tf.variable_scope('decoder'):
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3), dtype=tf.float32)
            self.decoder_embeddings = tf.get_variable("decoder_embeddings",
                                                      [self.tgt_vocab_size, self.input_embedding_size],
                                                      initializer=initializer,dtype=tf.float32)

            input_layer = Dense(self.decoder_hidden_units, dtype=tf.float32, name = 'input_projection')

            # Output projection layer to convert cell_outputs to logits
            output_layer = Dense(self.tgt_vocab_size, name = 'output_projection')

            if self.mode == 'train':

                # Train Mode
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                                      ids=self.decoder_inputs_train)
                
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)
                
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')
                
                training_decoder = seq2seq.BasicDecoder(cell = self.decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=output_layer)
                
                # Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)
                
                
                (self.decoder_output_train, self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))
                
                # [batch_size, max_time_step + 1, num_decoder_symbols]
                self.decoder_logits_train = tf.identity(self.decoder_output_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=1, name='decoder_pred_train')
                
                # [batch_size, max_time_steps + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length, dtype=tf.float32, name='masks')
                
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                            targets=self.decoder_targets_train,
                                            weights=masks,
                                            average_across_timesteps=True,
                                            average_across_batch=True)
                
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                
                # Contruct graphs for minimizing loss
                self.init_optimizer()


            elif self.mode == 'decode':

                # Decode mode

                start_tokens = tf.ones([self.batch_size,], tf.int32) * start_token
                # end_token = end_token

                def embed_and_input_proj(inputs):

                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))

                # Feeds input for greedy decoding: uses argmax for the output
                decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                end_token=end_token,
                                                                embedding=embed_and_input_proj)

                inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                         helper=decoding_helper,
                                                         initial_state=self.decoder_initial_state,
                                                         output_layer=output_layer)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    maximum_iterations=self.max_decode_step))


                # To be compatible in case of use of beam search
                # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)


    def build_single_cell(self):

        cell_type = tf.contrib.rnn.LSTMCell
        cell = cell_type(self.encoder_hidden_units)

        cell = tf.contrib.rnn.DropoutWrapper(cell, dtype=tf.float32, output_keep_prob=self.keep_prob_placeholder)

        cell = tf.contrib.rnn.ResidualWrapper(cell)

        return cell


    def build_encoder_cell(self):

        return tf.contrib.rnn.MultiRNNCell([self.build_single_cell() for i in range(self.depth)])


    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # Building Attention Mechanism: Default Bahdanau
        self.attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=self.decoder_hidden_units, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length)

        self.decoder_cell_list = [self.build_single_cell() for i in range(self.depth)]
        decoder_initial_state = encoder_last_state

        def attn_decoder_input_fn(inputs, attention):

            if not self.attn_input_feeding:
                return inputs

            _input_layer= Dense(self.decoder_hidden_units, dtype=tf.float32,
                                name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.decoder_hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_Wrapper')

        batch_size = self.batch_size
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=tf.float32)
        decoder_initial_state = tuple(initial_state)

        return tf.contrib.rnn.MultiRNNCell(self.decoder_cell_list), decoder_initial_state


    def init_optimizer(self):

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients,trainable_params), global_step=self.global_step)


    def save(self, sess, path, var_list=None, global_step=None):

        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path=path, global_step=global_step)

        print ('Model saved to %s' % save_path)


    def restore(self, sess, path, var_list=None): # vat_list=None

        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print ('Model restored from %s' % path)


    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length, False)

        input_feed[self.keep_prob_placeholder] = self.keep_prob

        output_feed = [self.updates, self.loss, self.summary_op]

        outputs = sess.run(output_feed, input_feed)

        return outputs[1], outputs[2] # loss, summary


    def eval(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)

        input_feed[self.keep_prob_placeholder] = 1.0

        output_feed = [self.loss, self.summary_op]

        # TODO: maybe it's better to use tensor
        outputs = sess.run(output_feed, input_feed)

        '''
        sess.run([train_op, loss_op], feed_dict={
            input: x_input,
            y_true: y_input
        })
        '''

        return outputs[0], outputs[1]  # loss, summary


    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decode=True)

        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_pred_decode]

        outputs = sess.run(output_feed, input_feed)

        return outputs[0]


    def check_feeds(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):

        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length


        return input_feed


    def predict(self, sess, encoder_inputs, encoder_inputs_length):

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs=None, decoder_inputs_length=None,
                                      decode=True)

        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.decoder_pred_decode]

        outputs = sess.run(output_feed, input_feed)

        return outputs[0]

















