import os
import json
import time
import argparse
import numpy as np
import cupy as cp
from tqdm import tqdm
from distutils.util import strtobool
from utils.tof import light_wave
from utils.scatter import phase_function


def gen_sim_gpu(drs_gt_vec, args):
    setup_start = time.time()
    ## if simulation, use 0-1 values as emitted light waveform
    ## if real, use real waveform observed by APD

    # drs_gt_vec = cp.load('/data/daiki-kij/scene_data/scale100/drs_gt_vec.npy')
    drs_gt_vec = cp.array(drs_gt_vec)
    dist_vec = drs_gt_vec[:, 0]
    dist_max = float(cp.max(dist_vec))
    min_time_index = int(np.round(args.disparity / args.c * 1e9 / args.temporal_unit_time_ns) + 1)

    TtoD = args.c * args.temporal_unit_time_ns * 1e-9 / 2
    before_dists = cp.arange(min_time_index, int(dist_max  / TtoD), 1) * TtoD
    args.temporal_size = 2 * len(before_dists)
    light_time = int(args.light_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)
    light_m = args.amp * light_wave(light_time, args.temporal_size)
    AtoQ = args.integration_count * args.temporal_unit_time_ns

    g1_width_index = int(args.g1_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)
    g2_width_index = int(args.g2_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)
    g3_width_index = int(args.g3_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)
    interval_width_index = int(args.interval_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)


    g1_start_index = 0
    g1_end_index = g1_start_index + g1_width_index
    g2_start_index = g1_end_index + interval_width_index
    g2_end_index = g2_start_index + g2_width_index
    g3_start_index = g2_end_index + interval_width_index
    g3_end_index = g3_start_index + g3_width_index

    phase = phase_function(args.g/100, phase_angle=np.pi)

    ## Observed light is returned from a certain distance, so index [0:min_time_index] is 0
    min_time_index = int(np.round(args.disparity / args.c * 1e9 / args.temporal_unit_time_ns) + 1)
    setup_end = time.time()
    print('- Setup is completed. :', str(setup_end-setup_start), '[s]')

    cal_start = time.time()

    num_pixels = len(drs_gt_vec)
    obj_index_vec = cp.asnumpy((dist_vec / TtoD).astype(cp.int))
    ref_vec = drs_gt_vec[:, 1]
    sigma_t_vec = drs_gt_vec[:, 2]
    sigma_s_vec = (args.rho / 100) * sigma_t_vec
    direct_w_ref = ref_vec * cp.exp(- 2 *  sigma_t_vec * dist_vec) / (dist_vec**2)

    gateMat = np.zeros((args.temporal_size,3))
    gateMat[g1_start_index:g1_end_index,0] = 1 #Q1
    gateMat[g2_start_index:g2_end_index,1] = 1 #Q2
    gateMat[g3_start_index:g3_end_index,2] = 1 #Q3
    gateConv = light_m.T.dot(gateMat)

    gateConv_obj = np.zeros([num_pixels, 3])
    for pix in range(num_pixels):
        gateConv_obj[pix] = gateConv[obj_index_vec[pix]]

    gateConv_obj = cp.array(gateConv_obj)    
    gateConv = cp.array(gateConv)

    Q_ours_fog = AtoQ * direct_w_ref[:,cp.newaxis]*gateConv_obj

    for dIdx, before_dist in enumerate(tqdm(before_dists)):
        mask = (dist_vec >= float(before_dist))*1
        impulse_response_fog = mask * phase * sigma_s_vec * cp.exp(-2 * sigma_t_vec * before_dist) / (before_dist**2)
        Q_ours_fog += AtoQ * impulse_response_fog[:,cp.newaxis]*gateConv[cp.newaxis,dIdx]


    Q_ours_fogfree = AtoQ * (ref_vec / (dist_vec**2))[:,cp.newaxis] * gateConv_obj

    return Q_ours_fog, Q_ours_fogfree

def gen_drs_gt(MOR, args, scene_root='', is_save=True, scene_org_root='/data/daiki-kij/scene_data'):
    phase = phase_function(g=0.9, phase_angle=np.pi)
    depth_raw = cv2.imread(os.path.join(scene_root, 'depth.png'), flags=2)/ 1000 
    ref_raw = cv2.imread(os.path.join(scene_root, 'ref.png'), flags=2)

    row, col = depth_raw.shape
    depth_gt = ((depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())**(1/1.)) *  300 + 4 
    ref_gt = (ref_raw - ref_raw.min()) / (ref_raw.max() - ref_raw.min()) + 0.005
    ref_gt = ref_gt / ref_gt.max()
    sigma_gt = np.random.normal(3/MOR, 0.001, (480, 640))
    # sigma_t = cv2.blur(sigma_t, (3, 3))
    drs_gt_vec = np.stack([depth_gt.reshape(-1), ref_gt.reshape(-1), sigma_gt.reshape(-1)], axis=1)
    
    return drs_gt_vec