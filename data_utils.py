
"""Functions that help with data processing for human3.6m"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy

import itertools

def rotmat2euler( R ):
  """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r

def rotmat2quat(R):
  """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
  rotdiff = R - R.T;

  r = np.zeros(3)
  r[0] = -rotdiff[1,2]
  r[1] =  rotdiff[0,2]
  r[2] = -rotdiff[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot ):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData

  # potentially ineficient, but only done once per experiment
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
  """
  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization

  Args
    poses: The output from the TF model. A list with (seq_length) entries,
    each with a (batch_size, dim) output
  Returns
    poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
    batch is an n-by-d sequence of poses.
  """
  seq_len = len(poses)
  if seq_len == 0:
    return []

  batch_size, dim = poses[0].shape

  poses_out = np.concatenate(poses)
  poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
  poses_out = np.transpose(poses_out, [1, 0, 2])

  poses_out_list = []
  for i in xrange(poses_out.shape[0]):
    poses_out_list.append(unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

  return poses_out_list


def readCSVasFloat(filename):
  """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
  returnArray = []
  lines = open(filename).readlines()
  for line in lines:
    line = line.strip().split(',')
    if len(line) > 0:
      returnArray.append(np.array([np.float32(x) for x in line]))

  returnArray = np.array(returnArray)
  return returnArray


def load_data(path_to_dataset, subjects, actions, one_hot):
  """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

  Args
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
  Returns
    trainData: dictionary with k:v
      k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
    completeData: nxd matrix with all the data. Used to normlization stats
  """
  nactions = len( actions )

  trainData = {}
  completeData = []
  total_frames = 0
  for subj in subjects:
    for action_idx in np.arange(len(actions)):

      action = actions[ action_idx ]

      for subact in [1, 2]:  # subactions

        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

        filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
        action_sequence = readCSVasFloat(filename)

        n, d = action_sequence.shape
        even_list = range(0, n, 2)

        if one_hot:
          # Add a one-hot encoding at the end of the representation
          the_sequence = np.zeros( (len(even_list), d + nactions), dtype=float )
          the_sequence[ :, 0:d ] = action_sequence[even_list, :]
          the_sequence[ :, d+action_idx ] = 1
          trainData[(subj, action, subact, 'even')] = the_sequence
          
        else:
          trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]


        if len(completeData) == 0:
          completeData = copy.deepcopy(action_sequence)
        else:
          completeData = np.append(completeData, action_sequence, axis=0)
  
  return trainData, completeData


def normalize_data( data, data_mean, data_std, dim_to_use, actions, one_hot ):
  """
  Normalize input data by removing unused dimensions, subtracting the mean and
  dividing by the standard deviation

  Args
    data: nx99 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    data_out: the passed data matrix, but normalized
  """
  data_out = {}
  nactions = len(actions)

  if not one_hot:
    # No one-hot encoding... no need to do anything special
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ] # comment this line if you want to model all_dims 
       	
  else:
    # TODO hard-coding 99 dimensions for un-normalized human poses
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key][:, 0:99] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ] # comment this line if you want to model all_dims
      data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )

  return data_out


def normalization_stats(completeData):
  """"
  Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

  Args
    completeData: nx99 matrix with data to normalize
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0 # comment line to avoid modifying std of ignored dims

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def body_part_features():
	"""
	function to return feature_idx ranges of dims_to_use vector for different body parts ex: torso, left_arm, right_arm, left_leg
	right_leg. 
	Outputs: dict which contains start_idx:end_idx for each of above mentioned body parts
	key: 'torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg'
	value: list of idxs of dims_to_use vector relevant to that body part
	"""	
	dims_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86]
	np_dims_to_use = np.asarray(dims_to_use)
		
	# defining the ranges of features for different body parts
	node_feature_ranges = {}
	node_feature_ranges['torso'] = ((0,5),(36,50))
	node_feature_ranges['right_arm'] = ((75,98),)
	node_feature_ranges['left_arm'] = ((51,74),)
	node_feature_ranges['right_leg'] = ((6,20),)
	node_feature_ranges['left_leg'] = ((21,35),)
		
	body_part_dims = {}
	
	for key in node_feature_ranges.keys():
		# resetting list which stores idxs of body parts
		part_dims = []
		for value in node_feature_ranges[key]:   
			# find indices in dims_to_use which fall in specified body ranges
			idxs = np.where( (np_dims_to_use >= value[0]) & (np_dims_to_use <= value[1]) )
			
			# convert back to tuple
			#idxs = np.ndarray.tolist(idxs[0])
			
			# collect and store them in a list
			part_dims.append(idxs[0])
		
		merged_part_dims = list(itertools.chain(*part_dims))
		merged_part_dims = np.asarray(merged_part_dims)
		# assign dims found
		body_part_dims[key] = merged_part_dims
	
	return body_part_dims

def pearson_corr_coef(X, Y):
	"""
        function to return Pearson's Correlation Coefficient between 2 sample X,Y
        Inputs:
        X - n X D (n = samples , D = feature_dims)
        Y - same shape as X
        Outputs:
        r - Pearson's Corr Coeff  
        """  
	r = np.sum( np.mean(X - np.mean(X,0),1) * np.mean(Y - np.mean(Y,0),1) ) / ( np.mean(np.std(X,0)) * np.mean(np.std(Y,0)) ) 

	return r

def KL_multi_var(X_mean, X_cov, Y_mean, Y_cov):
	
	#function to return Multi-variate KL Divergence for Gaussian RV
	#Inputs:
	#X_mean = mean vector of 1st RV
	#X_cov = covariance matrix of 1st RV
	#Y_mean = mean vector of 2nd RV
	#Y_cov = covariance matrix of 2nd RV 
	
	inv_X_cov = np.linalg.inv(X_cov)
	inv_Y_cov = np.linalg.inv(Y_cov)
	last_term = np.matmul((X_mean - Y_mean), (X_mean - Y_mean).T) + X_cov - Y_cov

	kl_xy = 0.5*np.log(np.linalg.det(np.matmul(Y_cov, inv_X_cov))) + 0.5*np.trace(np.matmul(inv_Y_cov, last_term))
	return kl_xy

