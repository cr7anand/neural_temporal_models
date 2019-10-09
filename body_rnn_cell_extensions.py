
""" Extensions to TF RNN class by una_dinosaria"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
from rnn_cell_implement import RNNCell # modified body cell definitions
#from deltaRNN import RNNCell # only for delta-RNN 
#from rnn_cell_implement import MultiRNNCell
import hard_att
import queue

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv
if pv(tf.__version__) >= pv('1.2.0'):
  from tensorflow.contrib.rnn import LSTMStateTuple
else:
  from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv

from tensorflow.python.ops import variable_scope as vs

import collections
import math

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, context, scope=None): # modified
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, context, scope) # modified

    # Add the residual connection
    output = tf.add(output, inputs)

    return output, new_state

class ResidualWrapperv1(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell, output_size):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell
    self._output_size = output_size

    self.r = tf.get_variable("r_interp", [self._output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, context, scope=None): # modified
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, context, scope) # modified

    # perform residual_v1 interpolation op
    output = (1.0 - self.r) * output + self.r * inputs
 
    return output, new_state


class ResidualWrapperv2(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell, output_size):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell
    self._output_size = output_size 

    self.r = tf.get_variable("r_interp", [self._output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.W_res = tf.get_variable("W_res", [self._output_size, self._output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.b_res = tf.get_variable("b_res", [self._output_size], dtype=tf.float32, initializer=tf.constant_initializer(0.1))


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, context, scope=None): # modified
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, context, scope) # modified
   
    # perform residual_v2 interpolation op
    output = (1.0 - self.r) * output + self.r * (tf.matmul(inputs, self.W_res) + self.b_res)
 
    return output, new_state


class LinearSpaceDecoderWrapper(RNNCell): # modified
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size, is_attention, num_attn_units, num_actions, memory_length):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell): # modified
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell
    self.is_attention = is_attention
    self.num_attn_units = num_attn_units
    self.num_actions = num_actions
    self.memory_length = memory_length 

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    # output projection params
    self.w_out = tf.get_variable("proj_w_out", [insize, output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    self.b_out = tf.get_variable("proj_b_out", [output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    if self.is_attention: # flag to indicate whether we're using attention-based LM
	# init attention params
	self.num_attn_units = num_attn_units
	self.W_1_attn = tf.get_variable("W_1_attn", [insize+self.num_actions, self.num_attn_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	self.W_2_attn = tf.get_variable("W_2_attn", [insize+self.num_actions, self.num_attn_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	self.v_a = tf.get_variable("v_a_attn", [1, self.num_attn_units], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	self.memory_length = memory_length
	self.w_out_c_t = tf.get_variable("w_out_c_t", [insize+self.num_actions, output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.attn_memory = queue.Queue(self.memory_length) 
        #self.call_counter = 0 	

    self.linear_output_size = output_size


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size

  def __call__(self, inputs, state, context, scope=None):
    """Use a linear layer and pass the output to the cell."""

    #self.call_counter = self.call_counter + 1 # temp fix 
    
    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, context, scope)

    if self.is_attention and self.attn_memory.full():
        # store t-50 prev states (h_enc) 
	self.attn_memory.get()
	self.attn_memory.put(tf.concat([new_state, context], axis=1))
 
    elif self.is_attention and (not self.attn_memory.full()):
  	self.attn_memory.put(tf.concat([new_state, context], axis=1)) 
     
    if self.is_attention: 					#and self.call_counter>50: # some flag to indicate when to use attention
    	# convert attn_memory -> list
    	list_attn_memory = list(self.attn_memory.queue)
	
    	# applying attention and include c_t to decode and get y_hat	
    	alpha, c_t = hard_att.bahdanau_attention(tf.concat([state, context], axis=1), list_attn_memory, self.v_a, self.W_1_attn, self.W_2_attn, self.memory_length)
    	output = tf.matmul(output, self.w_out) + tf.matmul(c_t, self.w_out_c_t) + self.b_out	

    if not self.is_attention:					 #) or (self.is_attention and self.call_counter <= 50):
        # Apply the multiplication to everything (when no attention is used to decode)
        output = tf.matmul(output, self.w_out) + self.b_out

    # setting counter back after 150 timesteps when attention is being used
    #if self.is_attention and self.call_counter == 150:
    #    self.call_counter = 0

    return output, new_state
