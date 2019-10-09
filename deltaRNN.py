#"""
#Author :- Ankur Mali
#"""

import os
import sys
import tensorflow as tf
import numpy as np
#from tensorflow.python.ops.rnn_cell import RNNCell
#from rnn_cell_impl import RNNCell
from rnn_cell_implement import RNNCell

class DeltaRNNCell(RNNCell):
    #"""
    #Delta RNN - Differential Framework.
    #Alexander G. Ororbia II, Tomas Mikolov and David Reitter,
    #"Learning Simpler Language Models with the
    # Delta Recurrent Neural Network Framework"
    #"""

    def __init__(self, num_units, apply_layer_norm=False):
        self._num_units = num_units
        self._apply_layer_norm = apply_layer_norm
        if self._apply_layer_norm:
           self._layer_norm = tf.contrib.layers.layer_norm

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def _outer_function(self, inner_function_output,
                        past_hidden_state, activation=tf.nn.relu,
                        wx_parameterization_gate=True, scope=None):
        #"""Check Equation 3 in Delta RNN paper
        # for basic understanding and to relate our code with papers maths.
        #"""

        assert inner_function_output.get_shape().as_list() == \
            past_hidden_state.get_shape().as_list()

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("OuterFunction"):
                r_bias = tf.get_variable(
                    "outer_function_gate",
                    [self._num_units],
                    dtype=tf.float32, initializer=tf.zeros_initializer)

                # Equation 5 in Alex(DRNN paper)
                if wx_parameterization_gate:
                    r = self._W_x_inputs + r_bias
                else:
                    r = r_bias

                gate = tf.nn.sigmoid(r)
                output = activation((1.0 - gate) * inner_function_output + gate * past_hidden_state)

        return output
        # End of outer function

    # Inner function 
    def _inner_function(self, inputs, past_hidden_state, 
                        activation=tf.nn.tanh, scope=None):
        #second order function as described equation 11 in delta rnn paper
        #This is used in inner function
        
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("InnerFunction"):
                with tf.variable_scope("Vh"):
                    V_h = _linear(past_hidden_state, self._num_units, True)

                with tf.variable_scope("Wx"):
                    self._W_x_inputs = _linear(inputs, self._num_units, True)

                alpha = tf.get_variable(
                    "alpha", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(2.0))
                    # alpha value 2.0 works better than 1.0
                beta_one = tf.get_variable(
                    "beta_one", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                beta_two = tf.get_variable(
                    "beta_two", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                z_t_bias = tf.get_variable(
                    "z_t_bias", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

                # 2nd order calculation
                #You can change activation function but before get familiar with gating operations and mathematical notations
                d_1_t = alpha * V_h * self._W_x_inputs
                d_2_t = beta_one * V_h + beta_two * self._W_x_inputs
     
                if self._apply_layer_norm:
                   d_1_t = self._layer_norm(d_1_t)
                   d_2_t = self._layer_norm(d_2_t)
   
                z_t = activation(d_1_t + d_2_t + z_t_bias)

        return z_t

    def __call__(self, inputs, state, scope=None):
        inner_function_output = self._inner_function(inputs, state)
        output = self._outer_function(inner_function_output, state)

        
        return output, output



class DeltaRNNCellBody(RNNCell):
    #
    #Delta RNN - Differential Framework.
    #Alexander G. Ororbia II, Tomas Mikolov and David Reitter,
    #"Learning Simpler Language Models with the
    # Delta Recurrent Neural Network Framework"
    #"""

    def __init__(self, num_units, apply_layer_norm=False):
        self._num_units = num_units
        self._apply_layer_norm = apply_layer_norm
        if self._apply_layer_norm: 
           self._layer_norm = tf.contrib.layers.layer_norm 

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def _outer_function(self, inner_function_output,
                        past_hidden_state, activation=tf.nn.relu,
                        wx_parameterization_gate=True, scope=None):
        #"""Check Equation 3 in Delta RNN paper
        # for basic understanding and to relate our code with papers maths.
        #"""

        assert inner_function_output.get_shape().as_list() == \
            past_hidden_state.get_shape().as_list()

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("OuterFunction"):
                r_bias = tf.get_variable(
                    "outer_function_gate",
                    [self._num_units],
                    dtype=tf.float32, initializer=tf.zeros_initializer)

                # Equation 5 in Alex(DRNN paper)
                if wx_parameterization_gate:
                    r = self._W_x_inputs + r_bias
                else:
                    r = r_bias

                gate = tf.nn.sigmoid(r)
                output = activation((1.0 - gate) * inner_function_output + gate * past_hidden_state)

        return output
        # """ End of outer function   """

    # """ Inner function """
    def _inner_function(self, inputs, past_hidden_state, context, activation=tf.nn.tanh, scope=None): # modified
        #"""second order function as described equation 11 in delta rnn paper
        #This is used in inner function
        #"""
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("InnerFunction"):
                with tf.variable_scope("Vh"):
                    V_h = _linear(past_hidden_state, self._num_units, True)

                with tf.variable_scope("Qm"): # modified
                    Q_m = _linear(context, self._num_units, True) 

                with tf.variable_scope("Wx"):
                    self._W_x_inputs = _linear(inputs, self._num_units, True)

                alpha = tf.get_variable(
                    "alpha", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(2.0))
                    #""" alpha value 2.0 works better than 1.0"""
                beta_one = tf.get_variable(
                    "beta_one", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                beta_two = tf.get_variable(
                    "beta_two", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                z_t_bias = tf.get_variable(
                    "z_t_bias", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

                # 2nd order calculation
                #You can change activation function but before get familiar with gating operations and mathematical notations
                d_1_t = alpha * V_h * ( self._W_x_inputs + Q_m ) # modified
                d_2_t = beta_one * V_h  + beta_two * ( self._W_x_inputs + Q_m ) # modified

                if self._apply_layer_norm:
                   d_1_t = self._layer_norm(d_1_t)
                   d_2_t = self._layer_norm(d_2_t)

                z_t = activation(d_1_t + d_2_t + z_t_bias)

        return z_t

    def __call__(self, inputs, state, context, scope=None):
        inner_function_output = self._inner_function(inputs, state, context)
        output = self._outer_function(inner_function_output, state)

        
        return output, output


class DeltaRNNCellBodyFlow(RNNCell):
    #
    #Delta RNN - Differential Framework.
    #Alexander G. Ororbia II, Tomas Mikolov and David Reitter,
    #"Learning Simpler Language Models with the
    # Delta Recurrent Neural Network Framework"
    #"""

    def __init__(self, num_units, apply_layer_norm=False):
        self._num_units = num_units
        self._apply_layer_norm = apply_layer_norm
        if self._apply_layer_norm: 
           self._layer_norm = tf.contrib.layers.layer_norm 

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def _outer_function(self, inputs, inner_function_output,
                        past_hidden_state, activation=tf.nn.relu,
                        wx_parameterization_gate=True, scope=None):
        #"""Check Equation 3 in Delta RNN paper
        # for basic understanding and to relate our code with papers maths.
        #"""

        assert inner_function_output.get_shape().as_list() == \
            past_hidden_state.get_shape().as_list()

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("OuterFunction"):
                r_bias = tf.get_variable("outer_function_vel_bias", [self._num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
                W_vel = tf.get_variable("outer_function_W_vel", [54, self._num_units ], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())  

                # Equation 5 in Alex(DRNN paper)
                if wx_parameterization_gate:
                    #r = self._W_x_inputs + r_bias
                    r = tf.matmul(inputs[:,54:108], W_vel) + r_bias # modified
                else:
                    r = r_bias

                gate = tf.nn.sigmoid(r)
                output = activation((1.0 - gate) * inner_function_output + gate * past_hidden_state)

        return output
        # """ End of outer function   """

    # """ Inner function """
    def _inner_function(self, inputs, past_hidden_state, context, activation=tf.nn.tanh, scope=None): # modified
        #"""second order function as described equation 11 in delta rnn paper
        #This is used in inner function
        #"""
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("InnerFunction"):
                with tf.variable_scope("Vh"):
                    V_h = _linear(past_hidden_state, self._num_units, True)

                with tf.variable_scope("Qm"): # modified
                    Q_m = _linear(context, self._num_units, True) 

                with tf.variable_scope("Wx"):
                    self._W_x_inputs = _linear(inputs[:,0:54], self._num_units, True)

                alpha = tf.get_variable(
                    "alpha", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(2.0))
                    #""" alpha value 2.0 works better than 1.0"""
                beta_one = tf.get_variable(
                    "beta_one", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                beta_two = tf.get_variable(
                    "beta_two", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(1.0))

                z_t_bias = tf.get_variable(
                    "z_t_bias", [self._num_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(0.0))

                # 2nd order calculation
                #You can change activation function but before get familiar with gating operations and mathematical notations
                d_1_t = alpha * V_h * ( self._W_x_inputs + Q_m ) # modified
                d_2_t = beta_one * V_h  + beta_two * ( self._W_x_inputs + Q_m ) # modified

                if self._apply_layer_norm:
                   d_1_t = self._layer_norm(d_1_t)
                   d_2_t = self._layer_norm(d_2_t)

                z_t = activation(d_1_t + d_2_t + z_t_bias)

        return z_t

    def __call__(self, inputs, state, context, scope=None):
        inner_function_output = self._inner_function(inputs, state, context)
        output = self._outer_function(inputs, inner_function_output, state)

        
        return output, output


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    #"""Linear mapping """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified, please check definition for input variables")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # dimension 1 cell size calculation.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2Dimensional Arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
