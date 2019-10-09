#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 01:41:56 2018

@author: anand
"""

import h5py
import numpy as np
import sklearn.metrics.pairwise as metrics
import matplotlib.pyplot as plt
import matplotlib.style as style
import data_utils
import forward_kinematics

def read_data(fname):
	
	hf = h5py.File(fname,'r')
	action = 'discussion'
	
	gt_sequences = np.zeros((8, 100, 99))
	pred_sequences = np.zeros((8, 100, 99))

	euler_gt_sequences = np.zeros((8, 100, 99))
	euler_pred_sequences = np.zeros((8, 100, 99))
	
	error_hf = hf.get('mean_'+ action + '_error/')
	errors = np.array(error_hf)

	for i in range(8):
		gt_fname = 'expmap/gt/' + action + '_' + str(i)
		n1 = np.array(hf.get(gt_fname))
		gt_sequences[i,:,:] = n1
		
		pred_fname = 'expmap/preds/' + action + '_' + str(i) 
		n2 = np.array(hf.get(pred_fname))
		pred_sequences[i,:,:] = n2

		# converting back to euler angles	
		for j in np.arange( gt_sequences.shape[1] ):
		         for k in np.arange(3,97,3):
			   	euler_gt_sequences[i, j, k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( gt_sequences[i, j, k:k+3] ))
				euler_pred_sequences[i, j, k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( pred_sequences[i, j, k:k+3] )) 

		euler_gt_sequences[i,:,0:6] = 0
		euler_pred_sequences[i,:,0:6] = 0
	
	return euler_gt_sequences, euler_pred_sequences, errors
	
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
	
	emd = np.zeros(cdf_pred_power.shape[0:3:2])
	
	# used to store powers of feature_dims and sequences used for avg later
	seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
	power_weighted_emd = 0
	
	for s in range(euler_gt_sequences.shape[0]):
		
		for d in range(euler_gt_sequences.shape[2]):
			gt_fourier_coeffs[s,:,d] = np.fft.fft(euler_gt_sequences[s,:,d]) # slice is 1D array
			pred_fourier_coeffs[s,:,d] = np.fft.fft(euler_pred_sequences[s,:,d])

			# computing power of fft per sequence per dim
			gt_power[s,:,d] = np.square(np.absolute(gt_fourier_coeffs[s,:,d]))
			pred_power[s,:,d] = np.square(np.absolute(pred_fourier_coeffs[s,:,d]))
			
			# matching power of gt and pred sequences
			gt_total_power = np.sum(gt_power[s,:,d])
			pred_total_power = np.sum(pred_power[s,:,d])
			#power_diff = gt_total_power - pred_total_power
			
			# adding power diff to zero freq of pred seq
			#pred_power[s,0,d] = pred_power[s,0,d] + power_diff
			
			# computing seq_power and feature_dims power 
			seq_feature_power[s,d] = gt_total_power
			
			# normalizing power per sequence per dim
			if gt_total_power != 0:
				gt_norm_power[s,:,d] = gt_power[s,:,d] / gt_total_power 
			
			if pred_total_power !=0:
				pred_norm_power[s,:,d] = pred_power[s,:,d] / pred_total_power
	
			# computing cumsum over freq
			cdf_gt_power[s,:,d] = np.cumsum(gt_norm_power[s,:,d]) # slice is 1D
			cdf_pred_power[s,:,d] = np.cumsum(pred_norm_power[s,:,d])
	
			# computing EMD 
			emd[s,d] = np.linalg.norm((cdf_pred_power[s,:,d] - cdf_gt_power[s,:,d]), ord=1)

	# computing weighted emd (by sequence and feature powers)	
	power_weighted_emd = np.average(emd, weights=seq_feature_power) 

	return power_weighted_emd


# read data from all models	
#gru_nl_nd_gt_sequence, gru_nl_nd_pred_sequence, gru_nl_nd_errors = read_data('../multi_exp_samples/long-term/simple_gru_no_plan_no_deriv/discussion_samples_v2.h5')	
#pgru_d_nl_gt_sequence, pgru_d_nl_pred_sequence, pgru_d_nl_errors = read_data('../final_exp_samples/long-term/full_pgru_no_loss/discussion_samples_v2.h5')	
#no_plan_gt_sequence, no_plan_pred_sequence, no_plan_errors = read_data('../final_exp_samples/long-term/gru-d/discussion_samples.h5')
#plan_gt_sequence, plan_pred_sequence, plan_errors = read_data('../final_exp_samples/long-term/pgru-d/discussion_samples.h5')
#multi_base_gt, multi_base_pred, multi_base_errors = read_data('../multi_action_samples/samples_1_layer_attn_drop_0.2_5k.h5')

# compute metrics for all models
multi_base_npss = compute_metrics(multi_base_gt, multi_base_pred)
#gru_nl_nd_emd = compute_metrics(gru_nl_nd_gt_sequence, gru_nl_nd_pred_sequence)
#pgru_d_nl_emd = compute_metrics(pgru_d_nl_gt_sequence, pgru_d_nl_pred_sequence)
#no_plan_emd  = compute_metrics(no_plan_gt_sequence, no_plan_pred_sequence)
#plan_emd = compute_metrics(plan_gt_sequence, plan_pred_sequence)
#auto_cond_emd = compute_metrics(auto_cond_gt_sequence, auto_cond_pred_sequence)
#jul_emd = compute_metrics(jul_gt_sequence, jul_pred_sequence)
#skip_1_emd = compute_metrics(skip_1_gt_sequence, skip_1_pred_sequence)
#jul_unsup_emd = compute_metrics(jul_unsup_gt_sequence, jul_unsup_pred_sequence)

