import os
import cv2
import json
import argparse
import numpy as np
import cupy as cp
from tqdm import tqdm
from distutils.util import strtobool
from utils.scatter import phase_function
from utils.tof import light_wave

def value2index(value, value_max, value_min, index_size):
    if value >= value_max:
        return 0
    if value <= value_min:
        return index_size - 1
    else:
        return int(np.round(value / (value_max - value_min) * index_size))

    
def index2value(index, value_max, value_min, index_size):
    assert (index <= index_size) and (index >= 0), 'invalid index'
    return index * (value_max - value_min) / index_size + value_min
    
    
def pred_drs(Q_input, Q1_LUT, Q2_LUT, Q3_LUT, args, nan_th=np.inf):
    """
    観測Qから
    d: 物体距離
    r: 物体反射率
    s: 霧の消滅係数
    を推定する．
    各推定値の範囲はargs参照
    
    Parameters
    ==========
    Q_input: cp.array([row*col, 3])
        観測値Q1, Q2, Q3
        
    Q1_LUT: cp.array(args.sigma_size)
        sとQ1の対応を保持したLookup table
        
    Q2_LUT, Q3_LUT: cp.array([args.dist_size, args.ref_size, args.sigma_size])
        それぞれdrsとQ2, drsとQ3の対応を保持したLookup table
        
    args: Namespace
        LUTを作るのに用いた設定 (引数)
        
        
    Returns
    =======
    drs_pred: np.array([row*col, 3])
        推定したdrs
    
    """
    drs_pred = cp.zeros(Q_input.shape)
    intensity_defogged = cp.zeros(Q_input.shape[0])
    Q_pred = cp.zeros(Q_input.shape)
    Q_error = cp.zeros(Q_input.shape)

    for i, (Q1_obs, Q2_obs, Q3_obs) in enumerate(tqdm(Q_input)):

        s_error = (Q1_LUT-Q1_obs)**2
        s_index = cp.argmin(s_error)
        drs_pred[i, 2] = index2value(s_index, 
                                     args.sigma_max, 
                                     args.sigma_min, 
                                     args.sigma_size)
        
        Q_pred[i, 0] = Q1_LUT[s_index] 
        Q_error[i, 0] = s_error[s_index]
        
        Q2Q3_obs = cp.array([Q2_obs, Q3_obs])
        Q2Q3_LUT = cp.stack([Q2_LUT[:, :, s_index], Q3_LUT[:, :, s_index]], axis=2).reshape(args.dist_size*args.ref_size,-1)

        dr_error_vec = (Q2Q3_LUT-Q2Q3_obs)**2
        dr_error = cp.sum(dr_error_vec, axis=1)
        
        
        if dr_error.min() < nan_th:
            dr_index = cp.argmin(dr_error)
            d_index = dr_index // args.ref_size
            r_index = dr_index % args.ref_size 
            
            Q_error[i, 1], Q_error[i, 2] = dr_error_vec[dr_index]

            drs_pred[i, 0] = index2value(d_index, 
                                         args.dist_max * args.scene_scale, 
                                         args.dist_min * args.scene_scale, 
                                         args.dist_size)
            drs_pred[i, 1] = index2value(r_index, 
                                         args.ref_max, 
                                         args.ref_min, 
                                         args.ref_size)
            

            Q_pred[i, 1] = Q2Q3_LUT[dr_index][0] 
            Q_pred[i, 2] = Q2Q3_LUT[dr_index][1]

            intensity_defogged[i] = Q2_LUT[d_index, r_index, 0] + Q3_LUT[d_index, r_index, 0]
            
        else:
            drs_pred[i, 0] = nan_th
            drs_pred[i, 1] = nan_th
            
            Q_pred[i, 1] = nan_th
            Q_pred[i, 2] = nan_th
            
            intensity_defogged[i] = nan_th

    return cp.asnumpy(drs_pred), cp.asnumpy(intensity_defogged), cp.asnumpy(Q_pred), cp.asnumpy(Q_error)

def pred_drs_cpu(Q_input, Q1_LUT, Q2_LUT, Q3_LUT, args, nan_th=np.inf):

    drs_pred = np.zeros(Q_input.shape)
    intensity_defogged = np.zeros(Q_input.shape[0])
    Q_pred = np.zeros(Q_input.shape)
    Q_error = np.zeros(Q_input.shape)

    for i, (Q1_obs, Q2_obs, Q3_obs) in enumerate(tqdm(Q_input)):

        s_error = (Q1_LUT-Q1_obs)**2
        s_index = np.argmin(s_error)
        drs_pred[i, 2] = index2value(s_index, 
                                     args.sigma_max, 
                                     args.sigma_min, 
                                     args.sigma_size)
        
        Q_pred[i, 0] = Q1_LUT[s_index] 
        Q_error[i, 0] = s_error[s_index]
        
        Q2Q3_obs = np.array([Q2_obs, Q3_obs])
        Q2Q3_LUT = np.stack([Q2_LUT[:, :, s_index], Q3_LUT[:, :, s_index]], axis=2).reshape(args.dist_size*args.ref_size,-1)

        dr_error_vec = (Q2Q3_LUT-Q2Q3_obs)**2
        dr_error = np.sum(dr_error_vec, axis=1)
        
        
        if dr_error.min() < nan_th:
            dr_index = np.argmin(dr_error)
            d_index = dr_index // args.ref_size
            r_index = dr_index % args.ref_size 
            
            Q_error[i, 1], Q_error[i, 2] = dr_error_vec[dr_index]

            drs_pred[i, 0] = index2value(d_index, 
                                         args.dist_max * args.scene_scale, 
                                         args.dist_min * args.scene_scale, 
                                         args.dist_size)
            drs_pred[i, 1] = index2value(r_index, 
                                         args.ref_max, 
                                         args.ref_min, 
                                         args.ref_size)
            

            Q_pred[i, 1] = Q2Q3_LUT[dr_index][0] 
            Q_pred[i, 2] = Q2Q3_LUT[dr_index][1]

            intensity_defogged[i] = Q2_LUT[d_index, r_index, 0] + Q3_LUT[d_index, r_index, 0]
            
        else:
            drs_pred[i, 0] = nan_th
            drs_pred[i, 1] = nan_th
            
            Q_pred[i, 1] = nan_th
            Q_pred[i, 2] = nan_th
            
            intensity_defogged[i] = nan_th

    return drs_pred, intensity_defogged, Q_pred, Q_error

def input_RevLUT(Q_input, RevLUT, 
                 Q1_max=250, Q1_min=0, Q1_size=50, 
                 Q2_max=3000, Q2_min=0, Q2_size=600, 
                 Q3_max=2000, Q3_min=0, Q3_size=400):
    
    pixels = Q_input.shape[0]
    dist = np.zeros(pixels)
    ref = np.zeros(pixels)
    sigma = np.zeros(pixels)
    intensity = np.zeros(pixels)
    dist, ref, sigma, intensity = RevLUT[value2index(Q_input[:, 0], Q1_max, Q1_min, Q1_size),
                                                         value2index(Q_input[:, 1], Q2_max, Q2_min, Q2_size),
                                                         value2index(Q_input[:, 2], Q3_max, Q3_min, Q3_size)]
    return dist, ref, sigma, intensity

def input_RevLUT_slow(Q_input, RevLUT, 
                 Q1_max=250, Q1_min=0, Q1_size=50, 
                 Q2_max=3000, Q2_min=0, Q2_size=600, 
                 Q3_max=2000, Q3_min=0, Q3_size=400):
    
    pixels = Q_input.shape[0]
    dist = np.zeros(pixels)
    ref = np.zeros(pixels)
    sigma = np.zeros(pixels)
    intensity = np.zeros(pixels)
    for i, Q in enumerate(Q_input):
        dist[i], ref[i], sigma[i], intensity[i] = RevLUT[value2index(Q[0], Q1_max, Q1_min, Q1_size),
                                                         value2index(Q[1], Q2_max, Q2_min, Q2_size),
                                                         value2index(Q[2], Q3_max, Q3_min, Q3_size)]
        
    return dist, ref, sigma, intensity
        
def const_reverse_LUT(Q1_LUT, Q2_LUT, Q3_LUT, args, 
                      Q1_max=250, Q1_min=0, Q1_size=25, 
                      Q2_max=3000, Q2_min=0, Q2_size=60, 
                      Q3_max=2000, Q3_min=0, Q3_size=40, 
                      nan_th=np.inf):
    
    RevLUT = cp.zeros([Q1_size, Q2_size, Q3_size, 4])
    Q2Q3_vec = cp.array([[int(index2value(Q2_index, Q2_max, Q2_min, Q2_size)),
                          int(index2value(Q3_index, Q3_max, Q3_min, Q3_size))] 
                         for Q2_index in tqdm(range(Q2_size)) for Q3_index in range(Q3_size)])
    
    for Q1 in tqdm(range(Q1_size)):
        s_error = (Q1_LUT-index2value(Q1, Q1_max, Q1_min, Q1_size))**2
        s_index = cp.argmin(s_error)
        RevLUT[Q1, :, :, 2] = index2value(s_index, args.sigma_max, args.sigma_min, args.sigma_size)

        Q2Q3_LUT = cp.stack([Q2_LUT[:, :, s_index],
                             Q3_LUT[:, :, s_index]], 
                             axis=2).reshape(args.dist_size*args.ref_size,-1)
    
        for Q2Q3_real_scale in Q2Q3_vec:
        
            Q2_revlut_scale = value2index(Q2Q3_real_scale[0], Q2_max, Q2_min, Q2_size)
            Q3_revlut_scale = value2index(Q2Q3_real_scale[1], Q3_max, Q3_min, Q3_size)
            dr_error_vec = (Q2Q3_LUT-Q2Q3_real_scale)**2
            dr_error = cp.sum(dr_error_vec, axis=1)
    
            if dr_error.min() < nan_th:
                dr_index = cp.argmin(dr_error)
                d_index = dr_index // args.ref_size
                r_index = dr_index % args.ref_size 

                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 0] = index2value(d_index, 
                                                                              args.dist_max * args.scene_scale, 
                                                                              args.dist_min * args.scene_scale, 
                                                                              args.dist_size)
                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 1] = index2value(r_index, 
                                                                              args.ref_max, 
                                                                              args.ref_min, 
                                                                              args.ref_size)
                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 3] = Q2_LUT[d_index, r_index, 0] + Q3_LUT[d_index, r_index, 0]

            else:
                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 0] = cp.nan
                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 1] = cp.nan
                RevLUT[Q1, Q2_revlut_scale, Q3_revlut_scale, 3] = cp.nan

    return cp.asnumpy(RevLUT)
