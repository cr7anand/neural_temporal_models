from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import copy
import data_utils
import cv2
from PIL import Image

def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 99

  # Structure that indicates parents for each joint
  njoints   = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    r = angles[ expmapInd[i] ]

    thisRotation = data_utils.expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]


  return np.reshape( xyz, [-1] )

def revert_coordinate_space(channels, R0, T0):
  """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
  n, d = channels.shape

  channels_rec = copy.copy(channels)
  R_prev = R0
  T_prev = T0
  rootRotInd = np.arange(3,6)

  # Loop through the passed posses
  for ii in range(n):
    R_diff = data_utils.expmap2rotmat( channels[ii, rootRotInd] )
    R = R_diff.dot( R_prev )

    channels_rec[ii, rootRotInd] = data_utils.rotmat2expmap(R)
    T = T_prev + ((R_prev.T).dot( np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
    channels_rec[ii,:3] = T
    T_prev = T
    R_prev = R

  return channels_rec


def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                    17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

  offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
  offset = offset.reshape(-1,3)

  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55],
            [],
            [59, 60, 58],
            [],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76],
            []]

  expmapInd = np.split(np.arange(4,100)-1,32)

  return parent, offset, rotInd, expmapInd

def read_data(gt_sequences, pred_sequences):
	
	euler_gt_sequences = np.zeros((100, 99))
	euler_pred_sequences = np.zeros((100, 99))
	
	# converting back to euler angles	
	for j in np.arange( gt_sequences.shape[1] ):
	         for k in np.arange(3,97,3):
		   	euler_gt_sequences[j, k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( gt_sequences[j, k:k+3] ))
			euler_pred_sequences[j, k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( pred_sequences[j, k:k+3] )) 

	euler_gt_sequences[:,0:6] = 0
	euler_pred_sequences[:,0:6] = 0
	
	return euler_gt_sequences, euler_pred_sequences

def compute_metrics(euler_gt_sequences, euler_pred_sequences):
	
	# computing 1) fourier coeffs 2)power of fft 3) normalizing power of fft dim-wise 4) cumsum over freq. 5) EMD 
	gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape)
	pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape)
	
	# power vars
	gt_power = np.zeros((gt_fourier_coeffs.shape))
	pred_power = np.zeros((gt_fourier_coeffs.shape))
	
	# normalizing power vars
	gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
	pred_norm_power = np.zeros(gt_fourier_coeffs.shape)
	
	cdf_gt_power = np.zeros(gt_norm_power.shape)
	cdf_pred_power = np.zeros(pred_norm_power.shape)
	
	emd = np.zeros(cdf_pred_power.shape[1])
	
	# used to store powers of feature_dims and sequences used for avg later
	seq_feature_power = np.zeros(euler_gt_sequences.shape[1])
	power_weighted_emd = 0
		
	for d in range(euler_gt_sequences.shape[1]):
		gt_fourier_coeffs[:,d] = np.fft.fft(euler_gt_sequences[:,d]) # slice is 1D array
		pred_fourier_coeffs[:,d] = np.fft.fft(euler_pred_sequences[:,d])

		# computing power of fft per sequence per dim
		gt_power[:,d] = np.square(np.absolute(gt_fourier_coeffs[:,d]))
		pred_power[:,d] = np.square(np.absolute(pred_fourier_coeffs[:,d]))
			
		# matching power of gt and pred sequences
		gt_total_power = np.sum(gt_power[:,d])
		pred_total_power = np.sum(pred_power[:,d])
			
		# computing seq_power and feature_dims power 
		seq_feature_power[d] = gt_total_power
			
		# normalizing power per sequence per dim
		if gt_total_power != 0:
			gt_norm_power[:,d] = gt_power[:,d] / gt_total_power 
			
		if pred_total_power !=0:
			pred_norm_power[:,d] = pred_power[:,d] / pred_total_power
	
		# computing cumsum over freq
		cdf_gt_power[:,d] = np.cumsum(gt_norm_power[:,d]) # slice is 1D
		cdf_pred_power[:,d] = np.cumsum(pred_norm_power[:,d])
	
		# computing EMD 
		emd[d] = np.linalg.norm((cdf_pred_power[:,d] - cdf_gt_power[:,d]), ord=1)

	# computing weighted emd (by sequence and feature powers)	
	power_weighted_emd = np.average(emd, weights=seq_feature_power) 

	return power_weighted_emd


def main():

  # Load all the data
  parent, offset, rotInd, expmapInd = _some_variables()

  # short-term models
  with h5py.File( '../final_exp_samples/short-term/jul_unsup_sa/walking_samples.h5', 'r' ) as h5f5:
    jul_unsup_sa_expmap_pred = h5f5['expmap/preds/walking_6'][:]
    expmap_gt_5 = h5f5['expmap/gt/walking_6'][:]

  with h5py.File( '../final_exp_samples/short-term/pgru_skip_1/walking_samples_v2.h5', 'r' ) as h5f6:
    pgru_skip_1_expmap_pred = h5f6['expmap/preds/walking_6'][:]
    expmap_gt_6 = h5f6['expmap/gt/walking_6'][:]

  # load mocap gt and PGRU-d model predictions
  with h5py.File( '../final_exp_samples/long-term/pgru-d/walking_samples_v2.h5', 'r' ) as h5f1:
    pgru_d_expmap_pred = h5f1['expmap/preds/walking_6'][:]
    expmap_gt_1 = h5f1['expmap/gt/walking_6'][:]

  with h5py.File( '../final_exp_samples/long-term/gru-d/walking_samples.h5', 'r' ) as h5f2:
    gru_d_expmap_pred = h5f2['expmap/preds/walking_6'][:]
    expmap_gt_2 = h5f2['expmap/gt/walking_6'][:]

  with h5py.File( '../final_exp_samples/long-term/pgru-a/walking_samples.h5', 'r' ) as h5f3:
    pgru_a_expmap_pred = h5f3['expmap/preds/walking_6'][:]
    expmap_gt_3 = h5f3['expmap/gt/walking_6'][:]

  with h5py.File( '../final_exp_samples/long-term/julietta/walking_samples.h5', 'r' ) as h5f4:
    jul_long_expmap_pred = h5f4['expmap/preds/walking_6'][:]
    expmap_gt_4 = h5f4['expmap/gt/walking_6'][:]  
    
  nframes_gt, nframes_pred = expmap_gt_1.shape[0], pgru_d_expmap_pred.shape[0]

  # computing NPSS metric for all models
  #euler_gt_5_seq, euler_jul_unsup_sa_seq = read_data(jul_unsup_sa_expmap_pred, expmap_gt_5)
  #euler_gt_6_seq, euler_pgru_skip_1_seq = read_data(pgru_skip_1_expmap_pred, expmap_gt_6)

  #euler_gt_1_seq, euler_pgru_d_seq = read_data(pgru_d_expmap_pred, expmap_gt_1)
  #euler_gt_2_seq, euler_gru_d_seq = read_data(gru_d_expmap_pred, expmap_gt_2)
  #euler_gt_3_seq, euler_pgru_a_seq = read_data(pgru_a_expmap_pred, expmap_gt_3)
  #euler_gt_4_seq, euler_jul_long_seq = read_data(jul_long_expmap_pred, expmap_gt_4)

  #jul_unsup_sa_emd = compute_metrics(euler_gt_5_seq, euler_jul_unsup_sa_seq)
  #pgru_skip_1_emd = compute_metrics(euler_gt_6_seq, euler_pgru_skip_1_seq)

  #pgru_d_emd = compute_metrics(euler_gt_1_seq, euler_pgru_d_seq)
  #gru_d_emd = compute_metrics(euler_gt_2_seq, euler_gru_d_seq)
  #pgru_a_emd = compute_metrics(euler_gt_3_seq, euler_pgru_a_seq)
  #jul_long_emd = compute_metrics(euler_gt_4_seq, euler_jul_long_seq)

  # Put them together and revert the coordinate space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_1, pgru_d_expmap_pred)), np.eye(3), np.zeros(3) )
  expmap_gt   = expmap_all[:nframes_gt,:]
  pgru_d_expmap_pred = expmap_all[nframes_gt:,:]

  # gru-d revert co-ord space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_2, gru_d_expmap_pred)), np.eye(3), np.zeros(3) )
  gru_d_expmap_pred = expmap_all[nframes_gt:,:]

  # pgru-ac revert co-ord space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_3, pgru_a_expmap_pred)), np.eye(3), np.zeros(3) )
  pgru_a_expmap_pred = expmap_all[nframes_gt:,:]

  # julietta-long revert co-ord space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_4, jul_long_expmap_pred)), np.eye(3), np.zeros(3) )
  jul_long_expmap_pred = expmap_all[nframes_gt:,:]

  # jul_unsup_sa revert co-ord space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_5, jul_unsup_sa_expmap_pred)), np.eye(3), np.zeros(3) )
  jul_unsup_sa_expmap_pred = expmap_all[nframes_gt:,:]

  # pgru_skip_1 revert co-ord space
  expmap_all = revert_coordinate_space( np.vstack((expmap_gt_6, pgru_skip_1_expmap_pred)), np.eye(3), np.zeros(3) )
  pgru_skip_1_expmap_pred = expmap_all[nframes_gt:,:]

 
  # Compute 3d points for each frame
  xyz_gt, pgru_d_xyz_pred = np.zeros((nframes_gt, 96)), np.zeros((nframes_pred, 96))
  gru_d_xyz_pred = np.zeros((nframes_gt, 96))
  pgru_a_xyz_pred = np.zeros((nframes_gt, 96))
  jul_long_xyz_pred = np.zeros((nframes_gt, 96))

  jul_unsup_sa_xyz_pred = np.zeros((nframes_gt, 96))
  pgru_skip_1_xyz_pred = np.zeros((nframes_gt, 96))

  # ground-truth xyz frames
  for i in range( nframes_gt ):
    xyz_gt[i,:] = fkl( expmap_gt[i,:], parent, offset, rotInd, expmapInd )

  # pgru-d xyz frames
  for i in range( nframes_pred ):
    pgru_d_xyz_pred[i,:] = fkl( pgru_d_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # gru-d xyz frames
  for i in range( nframes_pred ):
    gru_d_xyz_pred[i,:] = fkl( gru_d_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # gru-ac xyz frames
  for i in range( nframes_pred ):
    pgru_a_xyz_pred[i,:] = fkl( pgru_a_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # jul-long xyz frames
  for i in range( nframes_pred ):
    jul_long_xyz_pred[i,:] = fkl( jul_long_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # jul-unsup-sa xyz frames
  for i in range( nframes_pred ):
    jul_unsup_sa_xyz_pred[i,:] = fkl( jul_unsup_sa_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # pgru-skip-1 xyz frames
  for i in range( nframes_pred ):
    pgru_skip_1_xyz_pred[i,:] = fkl( pgru_skip_1_expmap_pred[i,:], parent, offset, rotInd, expmapInd )

  # setting up stuff to save video
  FFMpegWriter = animation.writers['ffmpeg']
  metadata = dict(title='Walking Sequence 6', artist='Matplotlib', comment='Movie support!')
  writer = FFMpegWriter(fps=12, codec="libx264", bitrate=-1, metadata=metadata)

  # === Plot and animate ===
  fig = plt.figure(figsize=(22.0,11.0))
  fig.suptitle("Walking Sequence 6") 
  fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=None, hspace=None)
  gt_ax = fig.add_subplot(3, 3, 2, projection='3d')  
  jul_unsup_sa_pred_ax = fig.add_subplot(3, 3, 4, projection='3d')
  pgru_skip_1_pred_ax = fig.add_subplot(3, 3, 5, projection='3d')
  jul_long_pred_ax = fig.add_subplot(3, 3, 6, projection='3d')
  pgru_a_pred_ax = fig.add_subplot(3, 3, 7, projection='3d')
  gru_d_pred_ax = fig.add_subplot(3, 3, 8, projection='3d')
  pgru_d_pred_ax = fig.add_subplot(3, 3, 9, projection='3d')

  # setting viewing angle
  gt_ax.view_init(azim=135)
  jul_unsup_sa_pred_ax.view_init(azim=45)
  pgru_skip_1_pred_ax.view_init(azim=45)
  jul_long_pred_ax.view_init(azim=45)
  pgru_a_pred_ax.view_init(azim=45)
  gru_d_pred_ax.view_init(azim=45)
  pgru_d_pred_ax.view_init(azim=45)

  font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
         }

  # titles and legends for subplots
  gt_ax.set_title("Ground-Truth")
  
  #jul_unsup_sa_emd_str = '$\mathrm{NPSS}=%.3f$'%(jul_unsup_sa_emd)
  jul_unsup_sa_pred_ax.set_title("A")
  #jul_unsup_sa_pred_ax.text2D(0.35,0.80, jul_unsup_sa_emd_str, fontdict=font, transform=jul_unsup_sa_pred_ax.transAxes)

  #pgru_skip_1_emd_str = '$\mathrm{NPSS}=%.3f$'%(pgru_skip_1_emd)
  pgru_skip_1_pred_ax.set_title("B")
  #pgru_skip_1_pred_ax.text2D(0.35,0.80, pgru_skip_1_emd_str, fontdict=font, transform=pgru_skip_1_pred_ax.transAxes)

  #jul_long_emd_str = '$\mathrm{NPSS}=%.3f$'%(jul_long_emd)
  jul_long_pred_ax.set_title("C")
  #jul_long_pred_ax.text2D(0.35,0.80, jul_long_emd_str, fontdict=font, transform=jul_long_pred_ax.transAxes)

  #pgru_a_emd_str = '$\mathrm{NPSS}=%.3f$'%(pgru_a_emd)
  pgru_a_pred_ax.set_title("D")
  #pgru_a_pred_ax.text2D(0.35,0.80, pgru_a_emd_str, fontdict=font, transform=pgru_a_pred_ax.transAxes)

  #gru_d_emd_str = '$\mathrm{NPSS}=%.3f$'%(gru_d_emd)
  gru_d_pred_ax.set_title("E")
  #gru_d_pred_ax.text2D(0.35, 0.80, gru_d_emd_str, fontdict=font, transform=gru_d_pred_ax.transAxes)

  #pgru_d_emd_str = '$\mathrm{NPSS}=%.3f$'%(pgru_d_emd)
  pgru_d_pred_ax.set_title("F")
  #pgru_d_pred_ax.text2D(0.35, 0.80, pgru_d_emd_str, fontdict=font, transform=pgru_d_pred_ax.transAxes)

  ob_gt = viz.Ax3DPose(gt_ax)
  jul_unsup_sa_ob_pred = viz.Ax3DPose(jul_unsup_sa_pred_ax)
  pgru_skip_1_ob_pred = viz.Ax3DPose(pgru_skip_1_pred_ax) 
  jul_long_ob_pred = viz.Ax3DPose(jul_long_pred_ax)
  pgru_a_ob_pred = viz.Ax3DPose(pgru_a_pred_ax)
  gru_d_ob_pred = viz.Ax3DPose(gru_d_pred_ax)
  pgru_d_ob_pred = viz.Ax3DPose(pgru_d_pred_ax)
  
  with writer.saving(fig, "walking_seq_6.mp4", 100):
    
    for i in range(nframes_gt):
      # Plot the conditioning ground truth
      ob_gt.update( xyz_gt[i,:] )
      fig.canvas.draw()

      # Plot the jul-unsup-sa prediction
      jul_unsup_sa_ob_pred.update( jul_unsup_sa_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()	 
      
      # Plot the pgru-skip-1 prediction
      pgru_skip_1_ob_pred.update( pgru_skip_1_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()

      # Plot the jul-long prediction
      jul_long_ob_pred.update( jul_long_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()

      # Plot the pgru-ac prediction
      pgru_a_ob_pred.update( pgru_a_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()

      # Plot the gru-d prediction
      gru_d_ob_pred.update( gru_d_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()
      
      # Plot the pgru-ac prediction
      pgru_d_ob_pred.update( pgru_d_xyz_pred[i,:], lcolor="#9b59b6", rcolor="#2ecc71" )
      plt.show(block=False)
      fig.canvas.draw()
      writer.grab_frame()
      
if __name__ == '__main__':
  main()
