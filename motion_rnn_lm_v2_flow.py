"""RNN model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random
import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import rnn_cell_extensions # my extensions of the tf repos
import data_utils

# modified rnn func
import rnn_cell_implement # contains modified RNN cell definitions
import rnn_cell_impl # tf rnn cells + layer_norm
import deltaRNN # contains drnn cell def
import rnn_mod # contains static_rnn function and rnn_step
import rnn

import body_rnn_cell_extensions_v1

class MotionRNNModelLM(object):
  """Sequence-to-sequence model for human motion prediction"""

  def __init__(self,
               architecture,
               loop_type, 
               source_seq_len,
               target_seq_len,
               body_rnn_size, # body-part rnn (forward-rnn) size
	       body_cell_type, # cell type of body-part rnn 	
	       plan_rnn_size, # plan-rnn (backward-rnn) size 
	       plan_cell_type, # cell type of plan-rnn  
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               loss_to_use,
               number_of_actions,
               one_hot=True,
               residual_velocities=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      architecture: [basic, tied] whether to tie the decoder and decoder.
      source_seq_len: length of the input sequence.
      target_seq_len: length of the target sequence.
      body_rnn_size: number of units in BodyPart RNN
      body_cell_type: RNNcell type used for BodyPart RNN
      plan_rnn_size: number of units in Plan RNN 
      plan_cell_type: RNNcell type used for Plan RNN
      num_layers: number of rnns to stack.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      summaries_dir: where to log progress for tensorboard.
      loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
        each timestep to compute the loss after decoding, or to feed back the
        prediction from the previous time-step.
      number_of_actions: number of classes we have.
      one_hot: whether to use one_hot encoding during train/test (sup models).
      residual_velocities: whether to use a residual connection that models velocities.
      dtype: the data type to use to store internal variables.
    """

    self.HUMAN_SIZE = 54
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'train')))
    self.test_writer  = tf.summary.FileWriter(os.path.normpath(os.path.join( summaries_dir, 'test')))

    self.loop_type = loop_type
    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.body_rnn_size = body_rnn_size
    self.body_cell_type = body_cell_type
    self.plan_rnn_size = plan_rnn_size
    self.plan_cell_type = plan_cell_type
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.n_actions = number_of_actions # used as input dim to plan rnn

    print( "Number of actions %d" % self.n_actions)

    self.global_step = tf.Variable(0, trainable=False)

    # setting up the kernel and bias initializers
    k_init = tf.orthogonal_initializer()
    x_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(0.1)

    # === Create Planning RNN (Backward-RNN) ===
    if self.plan_cell_type == "elman":    
	plan_rnn_cell = rnn_cell_impl.BasicRNNCell( self.plan_rnn_size , kernel_initializer=k_init, bias_initializer=b_init)	
    
    elif self.plan_cell_type == "lstm":
	plan_rnn_cell = rnn_cell_impl.BasicLSTMCell( self.plan_rnn_size )    

    elif self.plan_cell_type == "gru":
	plan_rnn_cell = rnn_cell_impl.GRUCell( self.plan_rnn_size , kernel_initializer=k_init, bias_initializer=b_init, apply_layer_norm=False)

    #elif self.plan_cell_type == "gru" and num_layers > 1:
    #    plan_rnn_cell = rnn_cell_impl.MultiRNNCell( [rnn_cell_impl.GRUCell(self.plan_rnn_size, kernel_initializer=k_init, bias_initializer=b_init) for _ in range(num_layers)] )  

    elif self.plan_cell_type == "delta":
        plan_rnn_cell = deltaRNN.DeltaRNNCell( self.plan_rnn_size, apply_layer_norm=False )
	 	
    # === Create Body RNN (Forward-RNN) ===
    if self.body_cell_type == "elman" and num_layers==1:    
	body_rnn_cell = rnn_cell_implement.BasicRNNCell( self.body_rnn_size , kernel_initializer=k_init, bias_initializer=b_init) # using modified RNN cell def to inlcude plan RNN
    
    elif self.body_cell_type == "elman" and num_layers > 1: 
        body_rnn_cell = rnn_cell_implement.MultiRNNCell( [rnn_cell_implement.BasicRNNCell(self.body_rnn_size, kernel_initializer=k_init, bias_initializer=b_init) for _ in range(num_layers)] )    

    elif self.body_cell_type == "lstm" and num_layers==1:
	body_rnn_cell = rnn_cell_implement.BasicLSTMCell( self.body_rnn_size ) # using modified LSTM cell def to inlcude plan RNN

    elif self.body_cell_type == "lstm" and num_layers > 1: 
        body_rnn_cell = rnn_cell_implement.MultiRNNCell( [rnn_cell_implement.BasicLSTMCell(self.body_rnn_size ) for _ in range(num_layers)] ) 
    
    elif self.body_cell_type == "gru" and num_layers==1:
	body_rnn_cell = rnn_cell_implement.GRUCell( self.body_rnn_size , kernel_initializer=k_init, bias_initializer=b_init, apply_layer_norm=False)	# using modified GRU cell def

    elif self.body_cell_type == "gru" and num_layers > 1: 
        body_rnn_cell = rnn_cell_implement.MultiRNNCell( [rnn_cell_implement.GRUCell(self.body_rnn_size, kernel_initializer=k_init, bias_initializer=b_init) for _ in range(num_layers)] )

    elif self.body_cell_type == "delta" and num_layers==1:
        body_rnn_cell = deltaRNN.DeltaRNNCellBody( self.body_rnn_size, apply_layer_norm=False )

    elif self.body_cell_type == "delta" and num_layers > 1: 
        body_rnn_cell = rnn_cell_implement.MultiRNNCell( [deltaRNN.DeltaRNNCellBody(self.body_rnn_size ) for _ in range(num_layers)] )

    
    # === Transform the inputs ===
    with tf.name_scope("inputs"):

      enc_in = tf.placeholder(dtype, shape=[None, source_seq_len, 4*self.HUMAN_SIZE], name="enc_in")
      enc_out = tf.placeholder(dtype, shape=[None, source_seq_len, self.HUMAN_SIZE], name="enc_out") 
      plan_in = tf.placeholder(dtype, shape=[None, source_seq_len+target_seq_len, self.n_actions], name="plan_in") # input one_hot vector for plan_rnn
      plan_init_state = tf.placeholder(dtype, shape=[None, self.plan_rnn_size], name="plan_init_state") # init state for plan rnn sampled from Gaussian noise
      dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, 4*self.HUMAN_SIZE], name="dec_in")
      dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.HUMAN_SIZE], name="dec_out")
      is_training = tf.placeholder(dtype=tf.bool, shape=[] , name="is_training") 
      use_sample = tf.placeholder(dtype=tf.bool, shape=[] , name="use_sample")
      dropout_prob = tf.placeholder(dtype, name="dropout_prob") 
      sampling_weight = tf.placeholder(dtype, shape=[], name="sampling_weight")	

      self.encoder_inputs = enc_in
      self.encoder_outputs = enc_out
      self.plan_inputs = plan_in
      self.plan_init_state = plan_init_state
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out
      self.is_training = is_training
      self.use_sample = use_sample
      self.dropout_prob = dropout_prob
      self.sampling_weight = sampling_weight

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      enc_out = tf.transpose(enc_out, [1, 0, 2]) 
      plan_in = tf.transpose(plan_in, [1, 0, 2]) # change made
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])

      enc_in = tf.reshape(enc_in, [-1, 4*self.HUMAN_SIZE])
      enc_out = tf.reshape(enc_out, [-1, self.HUMAN_SIZE])
      plan_in = tf.reshape(plan_in, [-1, self.n_actions]) # change made
      dec_in = tf.reshape(dec_in, [-1, 4*self.HUMAN_SIZE])
      dec_out = tf.reshape(dec_out, [-1, self.HUMAN_SIZE])

      enc_in = tf.split(enc_in, source_seq_len, axis=0)
      enc_out = tf.split(enc_out, source_seq_len, axis=0)
      plan_in = tf.split(plan_in, source_seq_len+target_seq_len, axis=0) # change made
      dec_in = tf.split(dec_in, target_seq_len, axis=0)
      dec_out = tf.split(dec_out, target_seq_len, axis=0)

      #plan_init_state = tf.unstack(plan_init_state, axis=1)
      #plan_init_state = tuple([plan_init_state[idx] for idx in range(self.num_layers)]) 

    # Add Dropout Wrapper to plan-rnn cell
    #plan_rnn_cell = rnn_cell_impl.DropoutWrapper(plan_rnn_cell, input_keep_prob = 1 - self.dropout_prob, output_keep_prob = 1 - self.dropout_prob)

    # Add Dropout Wrapper to body-rnn cell
    body_rnn_cell = rnn_cell_implement.DropoutWrapper(body_rnn_cell, input_keep_prob = 1 - self.dropout_prob, output_keep_prob = 1 - self.dropout_prob)

    # === Add Linear output projection ===
    body_rnn_cell = body_rnn_cell_extensions_v1.LinearSpaceDecoderWrapper( body_rnn_cell, self.HUMAN_SIZE )

    # Finally, wrap everything in a residual layer v1 = simple interpolation (1-r)*xtp1 + r*x, v2 = autoregressive style (1-r)* xtp1 + r* W_res x
    if residual_velocities:
      body_rnn_cell = body_rnn_cell_extensions.ResidualWrapperv2( body_rnn_cell, self.HUMAN_SIZE )    

    # Re-parameterization trick to set random init state of plan RNN
    sigma = tf.get_variable("re_param_sigma", [self.plan_rnn_size], dtype=tf.float32 )
    soft_plus_sigma = tf.sqrt(tf.nn.softplus(sigma))
    mu = tf.get_variable("re_param_mu", [self.plan_rnn_size], dtype=tf.float32 )#tf.zeros_initializer())
    re_param_plan_init_state = tf.add( mu, tf.multiply( plan_init_state, soft_plus_sigma ) ) 
    #re_param_plan_init_state = plan_init_state    
    
    # run planRNN to generate sequence of planning vectors
    with vs.variable_scope("plan_rnn"):   
      plan_outputs, plan_state = rnn.static_rnn(plan_rnn_cell, plan_in, initial_state=re_param_plan_init_state, dtype=tf.float32) 
      plan_outputs = tf.stack(plan_outputs, axis=2)
      
      # reversing outputs as plan-rnn runs backwards
      plan_outputs = tf.reverse(plan_outputs, axis=[2]) # reverse along time-dim
      
      #plan_outputs = tf.reshape(plan_outputs, [-1, self.plan_rnn_size])
      past_plan_outputs, future_plan_outputs = tf.split(plan_outputs, [self.source_seq_len, self.target_seq_len] , axis=2)  

      # reshaping into list of (batch_size, hidden_units) for RNN computation 
      past_plan_outputs = tf.transpose(past_plan_outputs, [2, 0, 1]) # makes it [T, B, hidden_units]
      future_plan_outputs = tf.transpose(future_plan_outputs, [2, 0, 1]) 

      past_plan_outputs = tf.reshape(past_plan_outputs, [-1, self.plan_rnn_size])
      future_plan_outputs = tf.reshape(future_plan_outputs, [-1, self.plan_rnn_size])

      past_plan_outputs = tf.split(past_plan_outputs, source_seq_len, axis=0)
      future_plan_outputs = tf.split(future_plan_outputs, target_seq_len, axis=0)  

    # Body-RNN
    with tf.name_scope("body_rnn_past"):
      # Run Body-RNN for past frames (use gt-input at each timestep)
      past_pred_outputs, past_state = rnn_mod.static_rnn(body_rnn_cell, enc_in, past_plan_outputs, dtype=tf.float32) 
      self.past_pred_outputs = past_pred_outputs   

    def future_test_graph(target_seq_len, past_state, past_pred_outputs, body_rnn_cell, enc_out, dec_out ):
        with vs.variable_scope("body_rnn_future", reuse=tf.AUTO_REUSE):  
		# Run Body-RNN for future frames (feed model output at t as input at t+1)		
		future_state = past_state
		future_output_i = tf.concat([past_pred_outputs[-1], past_pred_outputs[-1] - past_pred_outputs[-2], past_pred_outputs[-1] - 2*past_pred_outputs[-2] + past_pred_outputs[-3], past_pred_outputs[-1] - 3*past_pred_outputs[-2] + 3*past_pred_outputs[-3] - past_pred_outputs[-4]  ], axis=1)   
		future_pred_outputs = []
		prev_time_output = past_pred_outputs[-1] # variable used to store x_t-1
                p_prev_time_output = past_pred_outputs[-2] # variable used to store x_t-2
		pp_prev_time_output = past_pred_outputs[-3] # variable used to store x_t-3

		for i in range(target_seq_len): # last future frame input ignored as gt not available
			future_output_i, future_state = body_rnn_cell(future_output_i, future_state, future_plan_outputs[i]) # using cell state at end of past frames
			future_pred_outputs.append(future_output_i)
                        # computing angle velocities to append
 			future_output_i = tf.concat([future_output_i, future_output_i - prev_time_output, future_output_i - 2*prev_time_output + p_prev_time_output, future_output_i - 3*prev_time_output + 3*p_prev_time_output - pp_prev_time_output] , axis=1)
			 
			pp_prev_time_output = p_prev_time_output
			p_prev_time_output = prev_time_output
                        prev_time_output = future_output_i[:,0:self.HUMAN_SIZE] # storing only 1st 54 dims   			

		with tf.name_scope("loss_angles"):
      			l2_past_loss_angles = tf.square(tf.subtract(enc_out, past_pred_outputs))
        		l2_future_loss_angles = tf.square(tf.subtract(dec_out, future_pred_outputs))
			norm_l2 = tf.reduce_mean(tf.concat([l2_past_loss_angles, l2_future_loss_angles], 0))
        		loss_angles = norm_l2   # train time loss has only mse_loss
        return future_pred_outputs, loss_angles   
      
    def future_train_graph(loop_type, past_state, past_pred_outputs, target_seq_len, body_rnn_cell):
        if loop_type == "closed":
	  with tf.name_scope("body_rnn_future"):  
		# Run Body-RNN for future frames (feed model output at t as input at t+1)		
		future_state = past_state
		future_output_i = tf.concat([past_pred_outputs[-1], past_pred_outputs[-1] - past_pred_outputs[-2], past_pred_outputs[-1] - 2*past_pred_outputs[-2] + past_pred_outputs[-3], past_pred_outputs[-1] - 3*past_pred_outputs[-2] + 3*past_pred_outputs[-3] - past_pred_outputs[-4]  ], axis=1)   
		future_pred_outputs = []
		prev_time_output = past_pred_outputs[-1] # variable used to store x_t-1
                p_prev_time_output = past_pred_outputs[-2] # variable used to store x_t-2
		pp_prev_time_output = past_pred_outputs[-3] # variable used to store x_t-3

		for i in range(target_seq_len): # last future frame input ignored as gt not available
			future_output_i, future_state = body_rnn_cell(future_output_i, future_state, future_plan_outputs[i]) # using cell state at end of past frames
			future_pred_outputs.append(future_output_i)
			# computing angle velocities to append
 			future_output_i = tf.concat([future_output_i, future_output_i - prev_time_output, future_output_i - 2*prev_time_output + p_prev_time_output,  future_output_i - 3*prev_time_output + 3*p_prev_time_output - pp_prev_time_output], axis=1)
	
			pp_prev_time_output = p_prev_time_output
			p_prev_time_output = prev_time_output
                        prev_time_output = future_output_i[:,0:self.HUMAN_SIZE]  

        elif loop_type == "open":
	  with tf.name_scope("body_rnn_future"):
		# Run Body-RNN for future frames (use gt-input at each timestep)
		future_pred_outputs, future_state = rnn_mod.static_rnn( body_rnn_cell, dec_in, future_plan_outputs, initial_state=past_state, dtype=tf.float32 )

        elif loop_type == "auto_cond":
	  with tf.name_scope("body_rnn_future"):
		cond_len = 5
		future_state = past_state
		future_pred_outputs  = []
		for i in range(int(target_seq_len/cond_len)):
			if i % 2 == 0: # open loop
				future_outputs, future_state = rnn_mod.static_rnn(body_rnn_cell, dec_in[i*cond_len:(i+1)*cond_len], future_plan_outputs[i*cond_len:(i+1)*cond_len], initial_state=future_state, dtype=tf.float32)
				future_pred_outputs.extend(future_outputs)
				future_output_j = tf.concat([future_outputs[-1], future_outputs[-1] - future_outputs[-2], future_outputs[-1] - 2*future_outputs[-2] + future_outputs[-3], future_outputs[-1] - 3*future_outputs[-2] + 3*future_outputs[-3] -future_outputs[-4] ], axis=1)  

				t_minus_one_output = future_outputs[-1] # variable used to store x_t-1
                                t_minus_two_output = future_outputs[-2] # variable used to store x_t-2
				t_minus_thr_output = future_outputs[-3] # variable used to store x_t-3

			elif i % 2 == 1: # closed loop 
				for j in range(cond_len):
					future_output_j, future_state = body_rnn_cell(future_output_j, future_state, future_plan_outputs[(i*cond_len)+j])
					future_pred_outputs.append(future_output_j)  
					# computing angle velocities to append by hand
 					future_output_j = tf.concat([future_output_j, future_output_j - t_minus_one_output, future_output_j - 2*t_minus_one_output + t_minus_two_output, future_output_j - 3*t_minus_one_output + 3*t_minus_two_output - t_minus_thr_output ], axis=1)  

					t_minus_thr_output = t_minus_two_output
					t_minus_two_output = t_minus_one_output
                        		t_minus_one_output = future_output_j[:,0:self.HUMAN_SIZE] # storing only 1st 54 dims      
    
        return future_pred_outputs

    def sampling(past_state, past_pred_outputs, target_seq_len, body_rnn_cell, dec_out):
        #y_hat_open = future_train_graph("open", past_state, past_pred_outputs, target_seq_len, body_rnn_cell) 
        y_hat_closed = future_train_graph("closed", past_state, past_pred_outputs, target_seq_len, body_rnn_cell)
        sampling_loss = tf.reduce_mean(tf.square(tf.subtract(y_hat_closed, dec_out)))
        return sampling_loss

    def sampler_graph(use_sample, past_state, past_pred_outputs, target_seq_len, body_rnn_cell, enc_out, dec_out):
        # use_sample = True (use closed loop for mse loss) 
        # use_sample = False (use open loop for mse loss)  
 
        future_pred_outputs = tf.cond(use_sample, lambda: future_train_graph("closed", past_state, past_pred_outputs, target_seq_len, body_rnn_cell), lambda: future_train_graph("open", past_state, past_pred_outputs, target_seq_len, body_rnn_cell))

        # compute sampling loss
        sampling_loss = sampling(past_state, past_pred_outputs, target_seq_len, body_rnn_cell, dec_out)  

        with tf.name_scope("loss_angles"):
        	l2_past_loss_angles = tf.square(tf.subtract(enc_out, past_pred_outputs))
        	l2_future_loss_angles = tf.square(tf.subtract(dec_out, future_pred_outputs))
		norm_l2 = tf.reduce_mean(tf.concat([l2_past_loss_angles, l2_future_loss_angles], 0))
        	loss_angles = norm_l2 + self.sampling_weight*sampling_loss  # train time loss has both mse_loss and sampling  

        return future_pred_outputs, loss_angles

    # conditional graph construction depending on whether it's train or test time for future frames
    future_pred_outputs, loss_angles = tf.cond(self.is_training, lambda: sampler_graph(self.use_sample, past_state, past_pred_outputs, target_seq_len, body_rnn_cell, enc_out, dec_out), lambda: future_test_graph(target_seq_len, past_state, past_pred_outputs, body_rnn_cell, enc_out, dec_out))
    
    sampling_loss = sampling(past_state, past_pred_outputs, target_seq_len, body_rnn_cell, dec_out)

    self.future_pred_outputs = future_pred_outputs
           
    self.outputs = []

    self.outputs.append(self.past_pred_outputs)
    self.outputs.append(self.future_pred_outputs)
    
    self.loss         = loss_angles
    self.sampling_loss = sampling_loss 
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    opt = tf.train.RMSPropOptimizer( self.learning_rate, decay=0.9, momentum=0.95, centered=False )
    
    # Update all the trainable parameters
    gradients = tf.gradients( self.loss, params ) 

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # === variables for loss in Euler Angles -- for each action
    with tf.name_scope( "euler_error_walking" ):
      self.walking_err80   = tf.placeholder( tf.float32, name="walking_srnn_seeds_0080" )
      self.walking_err160  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0160" )
      self.walking_err320  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0320" )
      self.walking_err400  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0400" )
      self.walking_err560  = tf.placeholder( tf.float32, name="walking_srnn_seeds_0560" )
      self.walking_err1000 = tf.placeholder( tf.float32, name="walking_srnn_seeds_1000" )

      self.walking_err80_summary   = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0080', self.walking_err80 )
      self.walking_err160_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0160', self.walking_err160 )
      self.walking_err320_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0320', self.walking_err320 )
      self.walking_err400_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0400', self.walking_err400 )
      self.walking_err560_summary  = tf.summary.scalar( 'euler_error_walking/srnn_seeds_0560', self.walking_err560 )
      self.walking_err1000_summary = tf.summary.scalar( 'euler_error_walking/srnn_seeds_1000', self.walking_err1000 )
    with tf.name_scope( "euler_error_eating" ):
      self.eating_err80   = tf.placeholder( tf.float32, name="eating_srnn_seeds_0080" )
      self.eating_err160  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0160" )
      self.eating_err320  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0320" )
      self.eating_err400  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0400" )
      self.eating_err560  = tf.placeholder( tf.float32, name="eating_srnn_seeds_0560" )
      self.eating_err1000 = tf.placeholder( tf.float32, name="eating_srnn_seeds_1000" )

      self.eating_err80_summary   = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0080', self.eating_err80 )
      self.eating_err160_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0160', self.eating_err160 )
      self.eating_err320_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0320', self.eating_err320 )
      self.eating_err400_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0400', self.eating_err400 )
      self.eating_err560_summary  = tf.summary.scalar( 'euler_error_eating/srnn_seeds_0560', self.eating_err560 )
      self.eating_err1000_summary = tf.summary.scalar( 'euler_error_eating/srnn_seeds_1000', self.eating_err1000 )
    with tf.name_scope( "euler_error_smoking" ):
      self.smoking_err80   = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0080" )
      self.smoking_err160  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0160" )
      self.smoking_err320  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0320" )
      self.smoking_err400  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0400" )
      self.smoking_err560  = tf.placeholder( tf.float32, name="smoking_srnn_seeds_0560" )
      self.smoking_err1000 = tf.placeholder( tf.float32, name="smoking_srnn_seeds_1000" )

      self.smoking_err80_summary   = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0080', self.smoking_err80 )
      self.smoking_err160_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0160', self.smoking_err160 )
      self.smoking_err320_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0320', self.smoking_err320 )
      self.smoking_err400_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0400', self.smoking_err400 )
      self.smoking_err560_summary  = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_0560', self.smoking_err560 )
      self.smoking_err1000_summary = tf.summary.scalar( 'euler_error_smoking/srnn_seeds_1000', self.smoking_err1000 )
    with tf.name_scope( "euler_error_discussion" ):
      self.discussion_err80   = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0080" )
      self.discussion_err160  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0160" )
      self.discussion_err320  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0320" )
      self.discussion_err400  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0400" )
      self.discussion_err560  = tf.placeholder( tf.float32, name="discussion_srnn_seeds_0560" )
      self.discussion_err1000 = tf.placeholder( tf.float32, name="discussion_srnn_seeds_1000" )

      self.discussion_err80_summary   = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0080', self.discussion_err80 )
      self.discussion_err160_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0160', self.discussion_err160 )
      self.discussion_err320_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0320', self.discussion_err320 )
      self.discussion_err400_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0400', self.discussion_err400 )
      self.discussion_err560_summary  = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_0560', self.discussion_err560 )
      self.discussion_err1000_summary = tf.summary.scalar( 'euler_error_discussion/srnn_seeds_1000', self.discussion_err1000 )
    with tf.name_scope( "euler_error_directions" ):
      self.directions_err80   = tf.placeholder( tf.float32, name="directions_srnn_seeds_0080" )
      self.directions_err160  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0160" )
      self.directions_err320  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0320" )
      self.directions_err400  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0400" )
      self.directions_err560  = tf.placeholder( tf.float32, name="directions_srnn_seeds_0560" )
      self.directions_err1000 = tf.placeholder( tf.float32, name="directions_srnn_seeds_1000" )

      self.directions_err80_summary   = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0080', self.directions_err80 )
      self.directions_err160_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0160', self.directions_err160 )
      self.directions_err320_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0320', self.directions_err320 )
      self.directions_err400_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0400', self.directions_err400 )
      self.directions_err560_summary  = tf.summary.scalar( 'euler_error_directions/srnn_seeds_0560', self.directions_err560 )
      self.directions_err1000_summary = tf.summary.scalar( 'euler_error_directions/srnn_seeds_1000', self.directions_err1000 )
    with tf.name_scope( "euler_error_greeting" ):
      self.greeting_err80   = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0080" )
      self.greeting_err160  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0160" )
      self.greeting_err320  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0320" )
      self.greeting_err400  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0400" )
      self.greeting_err560  = tf.placeholder( tf.float32, name="greeting_srnn_seeds_0560" )
      self.greeting_err1000 = tf.placeholder( tf.float32, name="greeting_srnn_seeds_1000" )

      self.greeting_err80_summary   = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0080', self.greeting_err80 )
      self.greeting_err160_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0160', self.greeting_err160 )
      self.greeting_err320_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0320', self.greeting_err320 )
      self.greeting_err400_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0400', self.greeting_err400 )
      self.greeting_err560_summary  = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_0560', self.greeting_err560 )
      self.greeting_err1000_summary = tf.summary.scalar( 'euler_error_greeting/srnn_seeds_1000', self.greeting_err1000 )
    with tf.name_scope( "euler_error_phoning" ):
      self.phoning_err80   = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0080" )
      self.phoning_err160  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0160" )
      self.phoning_err320  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0320" )
      self.phoning_err400  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0400" )
      self.phoning_err560  = tf.placeholder( tf.float32, name="phoning_srnn_seeds_0560" )
      self.phoning_err1000 = tf.placeholder( tf.float32, name="phoning_srnn_seeds_1000" )

      self.phoning_err80_summary   = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0080', self.phoning_err80 )
      self.phoning_err160_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0160', self.phoning_err160 )
      self.phoning_err320_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0320', self.phoning_err320 )
      self.phoning_err400_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0400', self.phoning_err400 )
      self.phoning_err560_summary  = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_0560', self.phoning_err560 )
      self.phoning_err1000_summary = tf.summary.scalar( 'euler_error_phoning/srnn_seeds_1000', self.phoning_err1000 )
    with tf.name_scope( "euler_error_posing" ):
      self.posing_err80   = tf.placeholder( tf.float32, name="posing_srnn_seeds_0080" )
      self.posing_err160  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0160" )
      self.posing_err320  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0320" )
      self.posing_err400  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0400" )
      self.posing_err560  = tf.placeholder( tf.float32, name="posing_srnn_seeds_0560" )
      self.posing_err1000 = tf.placeholder( tf.float32, name="posing_srnn_seeds_1000" )

      self.posing_err80_summary   = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0080', self.posing_err80 )
      self.posing_err160_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0160', self.posing_err160 )
      self.posing_err320_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0320', self.posing_err320 )
      self.posing_err400_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0400', self.posing_err400 )
      self.posing_err560_summary  = tf.summary.scalar( 'euler_error_posing/srnn_seeds_0560', self.posing_err560 )
      self.posing_err1000_summary = tf.summary.scalar( 'euler_error_posing/srnn_seeds_1000', self.posing_err1000 )
    with tf.name_scope( "euler_error_purchases" ):
      self.purchases_err80   = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0080" )
      self.purchases_err160  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0160" )
      self.purchases_err320  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0320" )
      self.purchases_err400  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0400" )
      self.purchases_err560  = tf.placeholder( tf.float32, name="purchases_srnn_seeds_0560" )
      self.purchases_err1000 = tf.placeholder( tf.float32, name="purchases_srnn_seeds_1000" )

      self.purchases_err80_summary   = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0080', self.purchases_err80 )
      self.purchases_err160_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0160', self.purchases_err160 )
      self.purchases_err320_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0320', self.purchases_err320 )
      self.purchases_err400_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0400', self.purchases_err400 )
      self.purchases_err560_summary  = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_0560', self.purchases_err560 )
      self.purchases_err1000_summary = tf.summary.scalar( 'euler_error_purchases/srnn_seeds_1000', self.purchases_err1000 )
    with tf.name_scope( "euler_error_sitting" ):
      self.sitting_err80   = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0080" )
      self.sitting_err160  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0160" )
      self.sitting_err320  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0320" )
      self.sitting_err400  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0400" )
      self.sitting_err560  = tf.placeholder( tf.float32, name="sitting_srnn_seeds_0560" )
      self.sitting_err1000 = tf.placeholder( tf.float32, name="sitting_srnn_seeds_1000" )

      self.sitting_err80_summary   = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0080', self.sitting_err80 )
      self.sitting_err160_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0160', self.sitting_err160 )
      self.sitting_err320_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0320', self.sitting_err320 )
      self.sitting_err400_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0400', self.sitting_err400 )
      self.sitting_err560_summary  = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_0560', self.sitting_err560 )
      self.sitting_err1000_summary = tf.summary.scalar( 'euler_error_sitting/srnn_seeds_1000', self.sitting_err1000 )
    with tf.name_scope( "euler_error_sittingdown" ):
      self.sittingdown_err80   = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0080" )
      self.sittingdown_err160  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0160" )
      self.sittingdown_err320  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0320" )
      self.sittingdown_err400  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0400" )
      self.sittingdown_err560  = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_0560" )
      self.sittingdown_err1000 = tf.placeholder( tf.float32, name="sittingdown_srnn_seeds_1000" )

      self.sittingdown_err80_summary   = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0080', self.sittingdown_err80 )
      self.sittingdown_err160_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0160', self.sittingdown_err160 )
      self.sittingdown_err320_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0320', self.sittingdown_err320 )
      self.sittingdown_err400_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0400', self.sittingdown_err400 )
      self.sittingdown_err560_summary  = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_0560', self.sittingdown_err560 )
      self.sittingdown_err1000_summary = tf.summary.scalar( 'euler_error_sittingdown/srnn_seeds_1000', self.sittingdown_err1000 )
    with tf.name_scope( "euler_error_takingphoto" ):
      self.takingphoto_err80   = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0080" )
      self.takingphoto_err160  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0160" )
      self.takingphoto_err320  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0320" )
      self.takingphoto_err400  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0400" )
      self.takingphoto_err560  = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_0560" )
      self.takingphoto_err1000 = tf.placeholder( tf.float32, name="takingphoto_srnn_seeds_1000" )

      self.takingphoto_err80_summary   = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0080', self.takingphoto_err80 )
      self.takingphoto_err160_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0160', self.takingphoto_err160 )
      self.takingphoto_err320_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0320', self.takingphoto_err320 )
      self.takingphoto_err400_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0400', self.takingphoto_err400 )
      self.takingphoto_err560_summary  = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_0560', self.takingphoto_err560 )
      self.takingphoto_err1000_summary = tf.summary.scalar( 'euler_error_takingphoto/srnn_seeds_1000', self.takingphoto_err1000 )
    with tf.name_scope( "euler_error_waiting" ):
      self.waiting_err80   = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0080" )
      self.waiting_err160  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0160" )
      self.waiting_err320  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0320" )
      self.waiting_err400  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0400" )
      self.waiting_err560  = tf.placeholder( tf.float32, name="waiting_srnn_seeds_0560" )
      self.waiting_err1000 = tf.placeholder( tf.float32, name="waiting_srnn_seeds_1000" )

      self.waiting_err80_summary   = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0080', self.waiting_err80 )
      self.waiting_err160_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0160', self.waiting_err160 )
      self.waiting_err320_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0320', self.waiting_err320 )
      self.waiting_err400_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0400', self.waiting_err400 )
      self.waiting_err560_summary  = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_0560', self.waiting_err560 )
      self.waiting_err1000_summary = tf.summary.scalar( 'euler_error_waiting/srnn_seeds_1000', self.waiting_err1000 )
    with tf.name_scope( "euler_error_walkingdog" ):
      self.walkingdog_err80   = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0080" )
      self.walkingdog_err160  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0160" )
      self.walkingdog_err320  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0320" )
      self.walkingdog_err400  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0400" )
      self.walkingdog_err560  = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_0560" )
      self.walkingdog_err1000 = tf.placeholder( tf.float32, name="walkingdog_srnn_seeds_1000" )

      self.walkingdog_err80_summary   = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0080', self.walkingdog_err80 )
      self.walkingdog_err160_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0160', self.walkingdog_err160 )
      self.walkingdog_err320_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0320', self.walkingdog_err320 )
      self.walkingdog_err400_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0400', self.walkingdog_err400 )
      self.walkingdog_err560_summary  = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_0560', self.walkingdog_err560 )
      self.walkingdog_err1000_summary = tf.summary.scalar( 'euler_error_walkingdog/srnn_seeds_1000', self.walkingdog_err1000 )
    with tf.name_scope( "euler_error_walkingtogether" ):
      self.walkingtogether_err80   = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0080" )
      self.walkingtogether_err160  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0160" )
      self.walkingtogether_err320  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0320" )
      self.walkingtogether_err400  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0400" )
      self.walkingtogether_err560  = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_0560" )
      self.walkingtogether_err1000 = tf.placeholder( tf.float32, name="walkingtogether_srnn_seeds_1000" )

      self.walkingtogether_err80_summary   = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0080', self.walkingtogether_err80 )
      self.walkingtogether_err160_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0160', self.walkingtogether_err160 )
      self.walkingtogether_err320_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0320', self.walkingtogether_err320 )
      self.walkingtogether_err400_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0400', self.walkingtogether_err400 )
      self.walkingtogether_err560_summary  = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_0560', self.walkingtogether_err560 )
      self.walkingtogether_err1000_summary = tf.summary.scalar( 'euler_error_walkingtogether/srnn_seeds_1000', self.walkingtogether_err1000 )

    self.saver = tf.train.Saver( tf.global_variables(), max_to_keep=10 )

  def step(self, session, encoder_inputs, encoder_outputs, plan_inputs, plan_init_state, decoder_inputs, decoder_outputs, is_training, use_sample, dropout_prob, sampling_weight, forward_only, srnn_seeds=False):
    """Run a step of the model feeding the given inputs.

    Args
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs (past frames).
      encoder_outputs: list of numpy vectors to feed as encoder outputs (past frames) 
      plan_inputs: list of numpy vectors to feed as planning rnn inputs 
      decoder_inputs: list of numpy vectors to feed as decoder inputs. (future frames)
      decoder_outputs: list of numpy vectors that are the expected decoder outputs. (future frames)
      forward_only: whether to do the backward step or only forward.
      is_training: bool indicating whether training (True) or testing (False) phase
      srnn_seeds: True if you want to evaluate using the sequences of SRNN
    Returns
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.
    Raises
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed = {self.encoder_inputs: encoder_inputs,
		  self.encoder_outputs: encoder_outputs,	
		  self.plan_inputs: plan_inputs,
                  self.plan_init_state: plan_init_state,
                  self.decoder_inputs: decoder_inputs,
                  self.decoder_outputs: decoder_outputs,
                  self.is_training: is_training,
                  self.use_sample: use_sample,
                  self.dropout_prob: dropout_prob, 
		  self.sampling_weight: sampling_weight	}

    # Output feed: depends on whether we do a backward step or not.
    if not srnn_seeds:
      if not forward_only:

        # Training step
        output_feed = [self.updates,         # Update Op that does SGD.
                       self.gradient_norms,  # Gradient norm.
                       self.loss,
                       self.sampling_loss, 
                       self.loss_summary,
                       self.learning_rate_summary]

        outputs = session.run( output_feed, input_feed )
        return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]  # Gradient norm, loss, summaries

      else:
        # Validation step, not on SRNN's seeds
        output_feed = [self.loss, # Loss for this batch.
		       self.sampling_loss,	  
                       self.loss_summary]

        outputs = session.run(output_feed, input_feed)
        
        return outputs[0], outputs[1], outputs[2]  # No gradient norm
    else:
      # Validation on SRNN's seeds
      output_feed = [self.loss, # Loss for this batch.
                     self.outputs,
                     self.loss_summary]
      
      outputs = session.run(output_feed, input_feed)
      
      future_prd_outputs = outputs[1][1] # [0] = past_prd_outputs [1] = future_prd_outputs	

      return outputs[0], future_prd_outputs, outputs[2]  # No gradient norm, loss, outputs.



  def get_batch( self, data, actions):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
      rng: random number gen with set seed
    Returns
      The tuple (encoder_inputs, plan_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys    = list(data.keys())
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs = np.zeros((self.batch_size, self.source_seq_len, 4*self.HUMAN_SIZE), dtype=float) # append x_t - x_t-1
    encoder_outputs = np.zeros((self.batch_size, self.source_seq_len, self.HUMAN_SIZE), dtype=float) 
    plan_inputs = np.zeros((self.batch_size, self.source_seq_len+self.target_seq_len, self.n_actions), dtype=float)
    plan_init_state = np.zeros((self.batch_size, self.plan_rnn_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, 4*self.HUMAN_SIZE), dtype=float) # append x_t - x_t-1
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.HUMAN_SIZE), dtype=float)

    first_diff_enc = np.zeros(( self.source_seq_len, self.HUMAN_SIZE), dtype=float)
    first_diff_dec = np.zeros(( self.target_seq_len, self.HUMAN_SIZE), dtype=float)

    sec_diff_enc = np.zeros(( self.source_seq_len, self.HUMAN_SIZE), dtype=float)
    sec_diff_dec = np.zeros(( self.target_seq_len, self.HUMAN_SIZE), dtype=float)

    thr_diff_enc = np.zeros(( self.source_seq_len, self.HUMAN_SIZE), dtype=float)
    thr_diff_dec = np.zeros(( self.target_seq_len, self.HUMAN_SIZE), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames+1 ,:] # modified

      # Add the data
      encoder_inputs[i,:,0:self.HUMAN_SIZE]  = data_sel[0:self.source_seq_len, 0:self.HUMAN_SIZE]
      encoder_outputs[i,:,0:self.HUMAN_SIZE] = data_sel[1:self.source_seq_len+1, 0:self.HUMAN_SIZE] 

      # Append x_t - x_t-1      
      first_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      first_diff_enc[1:self.source_seq_len,:] = data_sel[1:self.source_seq_len, 0:self.HUMAN_SIZE] - data_sel[0:self.source_seq_len-1, 0:self.HUMAN_SIZE]

      # Append x_t - 2*x_t-1 + x_t-2
      sec_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      sec_diff_enc[1,:] = data_sel[1, 0:self.HUMAN_SIZE] - 2*data_sel[0, 0:self.HUMAN_SIZE] 
      sec_diff_enc[2:self.source_seq_len,:] = data_sel[2:self.source_seq_len, 0:self.HUMAN_SIZE] - 2*data_sel[1:self.source_seq_len-1, 0:self.HUMAN_SIZE] + data_sel[0:self.source_seq_len-2, 0:self.HUMAN_SIZE]
 
      # Append x_t - 3*x_t-1 + 3*x_t-2 - x_t-3
      thr_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[1,:] = data_sel[1, 0:self.HUMAN_SIZE] - 3*data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[2,:] = data_sel[2, 0:self.HUMAN_SIZE] - 3*data_sel[1, 0:self.HUMAN_SIZE] + 3*data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[3:self.source_seq_len,:] = data_sel[3:self.source_seq_len, 0:self.HUMAN_SIZE] - 3*data_sel[2:self.source_seq_len-1, 0:self.HUMAN_SIZE] + 3*data_sel[1:self.source_seq_len-2, 0:self.HUMAN_SIZE] - data_sel[0:self.source_seq_len-3, 0:self.HUMAN_SIZE]    

      encoder_inputs[i,:,self.HUMAN_SIZE:2*self.HUMAN_SIZE] = first_diff_enc
      encoder_inputs[i,:,2*self.HUMAN_SIZE:3*self.HUMAN_SIZE] = sec_diff_enc   
      encoder_inputs[i,:,3*self.HUMAN_SIZE:4*self.HUMAN_SIZE] = thr_diff_enc 

      # add action label to plan inputs
      action_label = np.tile(data_sel[1, self.HUMAN_SIZE:self.input_size], [1, self.source_seq_len+self.target_seq_len, 1])

      plan_inputs[i,:,:] = action_label # copying action label from encoder input

      plan_init_state[i,:] = np.random.normal(loc=0.0, scale=1.0, size=(1,self.plan_rnn_size)) # plan_init_state sampled from Gaussian

      # Append x_t - x_t-1  
      first_diff_dec[0:self.target_seq_len, :] = data_sel[self.source_seq_len:total_frames, 0:self.HUMAN_SIZE] - data_sel[self.source_seq_len-1:total_frames-1, 0:self.HUMAN_SIZE]

      # Append x_t - 2*x_t-1 + x_t-2
      sec_diff_dec[0,:] = data_sel[self.source_seq_len, 0:self.HUMAN_SIZE]
      sec_diff_dec[1,:] = data_sel[self.source_seq_len+1, 0:self.HUMAN_SIZE] - 2*data_sel[self.source_seq_len, 0:self.HUMAN_SIZE] 
      sec_diff_dec[2:self.target_seq_len,:] = data_sel[self.source_seq_len+2:total_frames, 0:self.HUMAN_SIZE] - 2*data_sel[self.source_seq_len+1:total_frames-1, 0:self.HUMAN_SIZE] + data_sel[self.source_seq_len:total_frames-2, 0:self.HUMAN_SIZE] 

      # Append x_t - 3*x_t-1 + 3*x_t-2 - x_t-3
      thr_diff_dec[0,:] = data_sel[self.source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[1,:] = data_sel[self.source_seq_len+1, 0:self.HUMAN_SIZE] - 3*data_sel[self.source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[2,:] = data_sel[self.source_seq_len+2, 0:self.HUMAN_SIZE] - 3*data_sel[self.source_seq_len+1, 0:self.HUMAN_SIZE] + 3*data_sel[self.source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[3:self.target_seq_len,:] = data_sel[self.source_seq_len+3:total_frames, 0:self.HUMAN_SIZE] - 3*data_sel[self.source_seq_len+2:total_frames-1, 0:self.HUMAN_SIZE] + 3*data_sel[self.source_seq_len+1:total_frames-2, 0:self.HUMAN_SIZE] - data_sel[self.source_seq_len:total_frames-3, 0:self.HUMAN_SIZE] 
 	
      decoder_inputs[i,:,0:self.HUMAN_SIZE]  = data_sel[self.source_seq_len:total_frames, 0:self.HUMAN_SIZE]
      decoder_inputs[i,:,self.HUMAN_SIZE:2*self.HUMAN_SIZE] = first_diff_dec
      decoder_inputs[i,:,2*self.HUMAN_SIZE:3*self.HUMAN_SIZE] = sec_diff_dec
      decoder_inputs[i,:,3*self.HUMAN_SIZE:4*self.HUMAN_SIZE] = thr_diff_dec     

      decoder_outputs[i,:,:] = data_sel[self.source_seq_len+1:total_frames+1, 0:self.HUMAN_SIZE]

    return encoder_inputs, encoder_outputs, plan_inputs, plan_init_state, decoder_inputs, decoder_outputs


  def find_indices_srnn( self, data, action ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    return idx

  def get_batch_srnn(self, data, action ):
    """
    Get a random batch of data from the specified bucket, prepare for step.

    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = self.find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs = np.zeros((batch_size, source_seq_len, 4*self.HUMAN_SIZE), dtype=float) # append x_t - x_t-1
    encoder_outputs = np.zeros((batch_size, source_seq_len, self.HUMAN_SIZE), dtype=float)
    plan_inputs  = np.zeros( (batch_size, source_seq_len+target_seq_len, self.n_actions), dtype=float )
    plan_init_state = np.zeros( (batch_size, self.plan_rnn_size), dtype=float) # plan_init_state sampled from Gaussian
    decoder_inputs  = np.zeros((batch_size, target_seq_len, 4*self.HUMAN_SIZE), dtype=float) # append x_t - x_t-1
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.HUMAN_SIZE), dtype=float )
 
    first_diff_enc = np.zeros(( source_seq_len, self.HUMAN_SIZE), dtype=float)
    first_diff_dec = np.zeros(( target_seq_len, self.HUMAN_SIZE), dtype=float) 

    sec_diff_enc = np.zeros(( source_seq_len, self.HUMAN_SIZE), dtype=float)
    sec_diff_dec = np.zeros(( target_seq_len, self.HUMAN_SIZE), dtype=float)

    thr_diff_enc = np.zeros(( source_seq_len, self.HUMAN_SIZE), dtype=float)
    thr_diff_dec = np.zeros(( target_seq_len, self.HUMAN_SIZE), dtype=float)

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len+1) ,:] # modified

      encoder_inputs[i, :, 0:self.HUMAN_SIZE]  = data_sel[0:source_seq_len, 0:self.HUMAN_SIZE]

      # Append x_t - x_t-1  
      first_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      first_diff_enc[1:source_seq_len,:] = data_sel[1:source_seq_len, 0:self.HUMAN_SIZE] - data_sel[0:source_seq_len-1, 0:self.HUMAN_SIZE]

      # Append x_t - 2*x_t-1 + x_t-2
      sec_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      sec_diff_enc[1,:] = data_sel[1, 0:self.HUMAN_SIZE] - 2*data_sel[0, 0:self.HUMAN_SIZE] 
      sec_diff_enc[2:source_seq_len,:] = data_sel[2:source_seq_len, 0:self.HUMAN_SIZE] - 2*data_sel[1:source_seq_len-1, 0:self.HUMAN_SIZE] + data_sel[0:source_seq_len-2, 0:self.HUMAN_SIZE]

      # Append x_t - 3*x_t-1 + 3*x_t-2 - x_t-3
      thr_diff_enc[0,:] = data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[1,:] = data_sel[1, 0:self.HUMAN_SIZE] - 3*data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[2,:] = data_sel[2, 0:self.HUMAN_SIZE] - 3*data_sel[1, 0:self.HUMAN_SIZE] + 3*data_sel[0, 0:self.HUMAN_SIZE]
      thr_diff_enc[3:source_seq_len,:] = data_sel[3:source_seq_len, 0:self.HUMAN_SIZE] - 3*data_sel[2:source_seq_len-1, 0:self.HUMAN_SIZE] + 3*data_sel[1:source_seq_len-2, 0:self.HUMAN_SIZE] - data_sel[0:source_seq_len-3, 0:self.HUMAN_SIZE] 
 
      encoder_inputs[i,:,self.HUMAN_SIZE:2*self.HUMAN_SIZE] = first_diff_enc
      encoder_inputs[i,:,2*self.HUMAN_SIZE:3*self.HUMAN_SIZE] = sec_diff_enc
      encoder_inputs[i,:,3*self.HUMAN_SIZE:4*self.HUMAN_SIZE] = thr_diff_enc

      encoder_outputs[i, :, :] = data_sel[1:source_seq_len+1, 0:self.HUMAN_SIZE]
      
      # add action label to plan inputs 
      action_label = np.tile(data_sel[1, self.HUMAN_SIZE:self.input_size], [1, source_seq_len+target_seq_len, 1])
      plan_inputs[i,:,:] = action_label # copying action label from encoder input

      plan_init_state[i,:] = np.random.normal(loc=0.0, scale=1.0, size=(1, self.plan_rnn_size))

      decoder_inputs[i, :, 0:self.HUMAN_SIZE]  = data_sel[source_seq_len:(source_seq_len+target_seq_len), 0:self.HUMAN_SIZE]

      # Append x_t - x_t-1  
      first_diff_dec[0:target_seq_len, :] = data_sel[source_seq_len:total_frames, 0:self.HUMAN_SIZE] - data_sel[source_seq_len-1:total_frames-1, 0:self.HUMAN_SIZE]

      # Append x_t - 2*x_t-1 + x_t-2
      sec_diff_dec[0,:] = data_sel[source_seq_len, 0:self.HUMAN_SIZE]
      sec_diff_dec[1,:] = data_sel[source_seq_len+1, 0:self.HUMAN_SIZE] - 2*data_sel[source_seq_len, 0:self.HUMAN_SIZE] 
      sec_diff_dec[2:target_seq_len,:] = data_sel[source_seq_len+2:total_frames, 0:self.HUMAN_SIZE] - 2*data_sel[source_seq_len+1:total_frames-1, 0:self.HUMAN_SIZE] + data_sel[source_seq_len:total_frames-2, 0:self.HUMAN_SIZE] 

      # Append x_t - 3*x_t-1 + 3*x_t-2 - x_t-3
      thr_diff_dec[0,:] = data_sel[source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[1,:] = data_sel[source_seq_len+1, 0:self.HUMAN_SIZE] - 3*data_sel[source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[2,:] = data_sel[source_seq_len+2, 0:self.HUMAN_SIZE] - 3*data_sel[source_seq_len+1, 0:self.HUMAN_SIZE] + 3*data_sel[source_seq_len, 0:self.HUMAN_SIZE]
      thr_diff_dec[3:self.target_seq_len,:] = data_sel[source_seq_len+3:total_frames, 0:self.HUMAN_SIZE] - 3*data_sel[source_seq_len+2:total_frames-1, 0:self.HUMAN_SIZE] + 3*data_sel[source_seq_len+1:total_frames-2, 0:self.HUMAN_SIZE] - data_sel[source_seq_len:total_frames-3, 0:self.HUMAN_SIZE]

      decoder_inputs[i,:,self.HUMAN_SIZE:2*self.HUMAN_SIZE] = first_diff_dec
      decoder_inputs[i,:,2*self.HUMAN_SIZE:3*self.HUMAN_SIZE] = sec_diff_dec
      decoder_inputs[i,:,3*self.HUMAN_SIZE:4*self.HUMAN_SIZE] = thr_diff_dec
     
      decoder_outputs[i, :, :] = data_sel[source_seq_len+1:(source_seq_len+target_seq_len+1), 0:self.HUMAN_SIZE]


    return encoder_inputs, encoder_outputs, plan_inputs, plan_init_state, decoder_inputs, decoder_outputs
