import os
import time
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from utils.args import const_LUT_args
from utils.analysis import index2value
from utils.scatter import phase_function
from utils.tof import light_wave

def main(args):
    
    setup_start = time.time()
    
    if args.is_save:
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
          
    light_time = int(args.light_clock * args.scene_scale * args.clock_ns / args.temporal_unit_time_ns)
    light_m = args.amp * light_wave(light_time, args.temporal_size)
    
    Q1 = np.zeros((args.dist_size, args.ref_size, args.sigma_size))
    Q2 = np.zeros((args.dist_size, args.ref_size, args.sigma_size))
    Q3 = np.zeros((args.dist_size, args.ref_size, args.sigma_size))
#     if is_debug:
#         returned_light = np.zeros((args.dist_size, args.ref_size, args.sigma_size))
    
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
    
    phase = phase_function(args.g / 100, phase_angle=np.pi)
    
    ## dist, ref, sigma's index list
#     ds = [[d, s] for d in range(args.dist_size) for s in range(args.sigma_size)]
#     r = np.array([index2value(r+1, args.ref_max, args.ref_min, args.ref_size) for r in range(args.ref_size)])
    
    ## Observed light is returned from a certain distance, so index [0:min_time_index] is 0
    min_time_index = int(np.round(args.disparity / args.c * 1e9 / args.temporal_unit_time_ns) + 1)
    setup_end = time.time()
    print('- Setup is completed. :', str(setup_end-setup_start), '[s]')
    
    cal_start = time.time()

    TtoD = args.c * args.temporal_unit_time_ns * 1e-9 / 2
    gateMat = np.zeros((args.temporal_size,3))
    gateMat[g1_start_index:g1_end_index,0] = 1 #Q1
    gateMat[g2_start_index:g2_end_index,1] = 1 #Q2
    gateMat[g3_start_index:g3_end_index,2] = 1 #Q3
    
    
    d_cpu = np.array(range(args.dist_size))[np.newaxis,:]
    s_cpu = np.array(range(args.sigma_size))[:,np.newaxis]
    r_cpu = np.array([index2value(r+1, args.ref_max, args.ref_min, args.ref_size) for r in range(args.ref_size)])[:,np.newaxis]
    t_cpu = np.arange(min_time_index, np.ceil(args.dist_max * args.scene_scale * 2/TtoD), 1)[np.newaxis,:] * TtoD
        
    gateConv = np.array(light_m.T.dot(gateMat))

    ts = np.ones((s_cpu.shape[0],t_cpu.shape[1]))
    ds = np.ones((s_cpu.shape[0],d_cpu.shape[1]))

        # Scattering property
    sigma_t = index2value(s_cpu, args.sigma_max, args.sigma_min, args.sigma_size)
    sigma_s = (args.rho / 100) * sigma_t

        # Direct Component
    obj_dist = index2value(d_cpu, args.dist_max * args.scene_scale, args.dist_min * args.scene_scale, args.dist_size)
    direct_wo_ref = np.exp(- 2 * sigma_t * obj_dist) / (obj_dist**2)

        # Scattering Component
        ## dist list between object and min_time_index
    impulse_response_fog = phase * sigma_s * np.exp(-2 * sigma_t * t_cpu) / (t_cpu**2)

        # Modeling Temporal Response
        ## before obj_index, tenporal_response_fog
        ## At obj_index, r * direct_wo_ref
    obj_index = (obj_dist / TtoD).astype(np.int)
    AtoQ = args.integration_count * args.temporal_unit_time_ns

    for dIdx, objIdx in enumerate(tqdm(list(obj_index.flatten()))):
        temporal_response_fog = np.zeros((args.sigma_size, args.temporal_size))
        temporal_response_fog[:, min_time_index:objIdx] = impulse_response_fog[:,:(int(objIdx)-min_time_index)]
            # temporal_response_fog = temporal_response_fog.dot(np.array(light_m.T))
        gate_observation_fog = temporal_response_fog.dot(gateConv)

            # Store light to each gate as Q1 - Q3
        Q1[dIdx] = AtoQ * (gate_observation_fog[np.newaxis,:,0] + (r_cpu * direct_wo_ref[:,dIdx])*gateConv[objIdx,0])
        Q2[dIdx] = AtoQ * (gate_observation_fog[np.newaxis,:,1] + (r_cpu * direct_wo_ref[:,dIdx])*gateConv[objIdx,1])
        Q3[dIdx] = AtoQ * (gate_observation_fog[np.newaxis,:,2] + (r_cpu * direct_wo_ref[:,dIdx])*gateConv[objIdx,2])

    if args.is_save:
        print('Start saving')
        # Q1 integration value does not depend on reflectance of objects. Q1's 0 means ref=0. 
        save_start = time.time()
        np.save(os.path.join(args.result_dir, 'Q1.npy'), Q1[0, 0])
        del Q1
        np.save(os.path.join(args.result_dir, 'Q2.npy'), Q2)
        del Q2
        np.save(os.path.join(args.result_dir, 'Q3.npy'), Q3)
        del Q3
        save_end = time.time()
        print('Save is completed', save_end-save_start)
        
        # Save configs
        with open(os.path.join(args.result_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
            
        print('- Saved to:', args.result_dir)
        
    
if __name__ == '__main__':
    args = const_LUT_args()
    main(args)