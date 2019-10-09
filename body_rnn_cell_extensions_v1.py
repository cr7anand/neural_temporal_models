
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

  def __init__(self, cell, output_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell): # modified
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell 

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
    self.linear_output_size = output_size


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size

  def __call__(self, inputs, state, context, scope=None):
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, context, scope)

    output = tf.matmul(output, self.w_out) + self.b_out

    return output, new_state
