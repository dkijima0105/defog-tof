import os
import json
import argparse
from distutils.util import strtobool


def const_LUT_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float, help='speed of light',
            default=3e8)
    parser.add_argument('--g', type=int, 
                        help='Forward Scattering coefficient of HG function. acutual g = g/100', 
                        default=90)
    parser.add_argument('--rho', type=int, help="Single Scattering Albedo", default=98)
    parser.add_argument('--amp', type=float, help='Amplitude of emitted light', default=1.0)
    
    parser.add_argument('--scene_scale', type=int, help='fog scene to real scene scale', default=20)
    parser.add_argument('--dist_size', type=int, help='The number of distance grid', default=2000)
    parser.add_argument('--ref_size', type=int, help='The number of reflectance grid', default=500)
    parser.add_argument('--sigma_size', type=int, help='The number of sigima_t grid', default=500)
    parser.add_argument('--dist_min', type=float, help='minimum distance', default=0.8)
    parser.add_argument('--dist_max', type=float, help='maximum distance', default=4.8)
    parser.add_argument('--ref_min', type=float, help='minimum reflectance', default=0.0)
    parser.add_argument('--ref_max', type=float, help='maximum reflectance', default=1.0)
    parser.add_argument('--sigma_min', type=float, help='minimum extinction coefficient', default=0.0)
    parser.add_argument('--sigma_max', type=float, help='maximum extinction coefficient', default=0.40)
    
    parser.add_argument('--temporal_size', type=int, help='The length of lightwave', default=3000)
    parser.add_argument('--temporal_unit_time_ns', type=float, help='temporal unit time when args.scene_scale = 1.0: [ns]', default=0.1)
    parser.add_argument('--disparity', type=float, help='The distance between camera and light source [m]', default=0.03)
    parser.add_argument('--clock_ns', type=float, help='1 clock is 2.65 ns', default=2.65)
    parser.add_argument('--g1_clock', type=float, help='G1 width [clock]', default=2.0)
    parser.add_argument('--g2_clock', type=float, help='G2 width [clock]', default=10.0)
    parser.add_argument('--g3_clock', type=float, help='G3 width [clock]', default=10.0)
    parser.add_argument('--light_clock', type=float, help='Light width [clock]', default=11.0)
    parser.add_argument('--interval_clock', type=float, help='interval clock width [clock]', default=0.0)
    parser.add_argument('--integration_count', type=int, help='The number of integration', default=60000)
    
    parser.add_argument('--gpu_id', type=int, help='GPU device id = [0, 1]', default=1)
    parser.add_argument('--result_dir', type=str, help='path to data dir', default="~/Research/defog/LUT/debug")
    parser.add_argument('--is_save', type=strtobool, default=True)
    
    return parser.parse_args()

def decode_LUT_args(LUT_root):
    
    json_open = open(os.path.join(LUT_root, 'config.json'), "r")
    args_dict = json.load(json_open)

    parser = argparse.ArgumentParser()

    parser.add_argument('--c', type=float, help='speed of light',
            default=args_dict['c'])
    parser.add_argument('--amp', type=float, help='Radiant Intensity', 
                       default=args_dict['amp'])
    parser.add_argument('--g', type=float, help='Forward Scattering coefficient of Henyey Greenstein pahse function', 
                        default=args_dict['g'])
    parser.add_argument('--rho', type=float, help="Single Scattering Albedo", 
                        default=args_dict['rho'])

    parser.add_argument('--scene_scale', type=int, help='fog scene to real scene scale',
                        default=args_dict['scene_scale'])
    parser.add_argument('--dist_size', type=int, help='The number of distance grid',
                        default=args_dict['dist_size'])
    parser.add_argument('--ref_size', type=int, help='The number of reflectance grid',
                        default=args_dict['ref_size'])
    parser.add_argument('--sigma_size', type=int, help='The number of sigima_t grid',
                        default=args_dict['sigma_size'])
    parser.add_argument('--dist_min', type=float, help='minimum distance', 
                        default=args_dict['dist_min'])
    parser.add_argument('--dist_max', type=float, help='maximum distance', 
                        default=args_dict['dist_max'])
    parser.add_argument('--ref_min', type=float, help='minimum reflectance', 
                        default=args_dict['ref_min'])
    parser.add_argument('--ref_max', type=float, help='maximum reflectance', 
                        default=args_dict['ref_max'])
    parser.add_argument('--sigma_min', type=float, help='minimum extinction coefficient',
                        default=args_dict['sigma_min'])
    parser.add_argument('--sigma_max', type=float, help='maximum extinction coefficient',
                        default=args_dict['sigma_max'])

    parser.add_argument('--temporal_size', type=int, help='The length of lightwave',
                        default=args_dict['temporal_size'])
    parser.add_argument('--temporal_unit_time_ns', type=float, help='temporal unit time when args.scene_scale = 1.0: [ns]', 
                        default=args_dict['temporal_unit_time_ns'])
    parser.add_argument('--disparity', type=float, help='The distance between camera and light source [m]', 
                        default=args_dict['disparity'])
    parser.add_argument('--clock_ns', type=float, help='1 clock is 2.65 ns', 
                        default=args_dict['clock_ns'])
    parser.add_argument('--g1_clock', type=float, help='G1 width [clock]', 
                        default=args_dict['g1_clock'])
    parser.add_argument('--g2_clock', type=float, help='G2 width [clock]', 
                        default=args_dict['g2_clock'])
    parser.add_argument('--g3_clock', type=float, help='G3 width [clock]', 
                        default=args_dict['g3_clock'])
    parser.add_argument('--light_clock', type=float, help='Light width [clock]', 
                        default=args_dict['light_clock'])
    parser.add_argument('--interval_clock', type=float, help='interval clock width [clock]', 
                        default=args_dict['interval_clock'])
    parser.add_argument('--integration_count', type=int, help='The number of integration',
                        default=args_dict['integration_count'])
    

    args = parser.parse_args(args=[])

    return args

def decode_LUT_args_old(LUT_root):
    
    json_open = open(os.path.join(LUT_root, 'config.json'), "r")
    args_dict = json.load(json_open)

    parser = argparse.ArgumentParser()

    parser.add_argument('--c', type=float, help='speed of light',
            default=args_dict['c'])
    parser.add_argument('--amp', type=float, help='Radiant Intensity', 
                       default=args_dict['amp'])
    parser.add_argument('--g', type=float, help='Forward Scattering coefficient of Henyey Greenstein pahse function', 
                        default=args_dict['g'])
    parser.add_argument('--rho', type=float, help="Single Scattering Albedo", 
                        default=args_dict['rho'])

    parser.add_argument('--scene_scale', type=int, help='fog scene to real scene scale',
                        default=args_dict['scene_scale'])
    parser.add_argument('--dist_size', type=int, help='The number of distance grid',
                        default=args_dict['dist_size'])
    parser.add_argument('--ref_size', type=int, help='The number of reflectance grid',
                        default=args_dict['ref_size'])
    parser.add_argument('--sigma_size', type=int, help='The number of sigima_t grid',
                        default=args_dict['sigma_size'])
    parser.add_argument('--dist_min', type=float, help='minimum distance',
                        default=args_dict['dist_min'])
    parser.add_argument('--dist_max', type=float, help='maximum distance', 
                        default=args_dict['dist_max'])
    parser.add_argument('--ref_min', type=float, help='minimum reflectance', 
                        default=args_dict['ref_min'])
    parser.add_argument('--ref_max', type=float, help='maximum reflectance', 
                        default=args_dict['ref_max'])
    parser.add_argument('--sigma_min', type=float, help='minimum extinction coefficient',
                        default=args_dict['sigma_min'])
    parser.add_argument('--sigma_max', type=float, help='maximum extinction coefficient',
                        default=args_dict['sigma_max'])

    parser.add_argument('--temporal_size', type=int, help='The length of lightwave',
                        default=args_dict['temporal_size'])
    parser.add_argument('--temporal_unit_time_ns', type=float, help='temporal unit time when args.scene_scale = 1.0: [ns]', 
                        default=args_dict['temporal_unit_time_ns'])
    parser.add_argument('--disparity', type=float, help='The distance between camera and light source [m]', 
                        default=args_dict['disparity'])
    parser.add_argument('--clock_ns', type=float, help='1 clock is 2.65 ns', 
                        default=args_dict['clock_ns'])
    parser.add_argument('--g1_clock', type=float, help='G1 width [clock]', 
                        default=args_dict['g1_clock'])
    parser.add_argument('--g2_clock', type=float, help='G2 width [clock]', 
                        default=args_dict['g2_clock'])
    parser.add_argument('--g3_clock', type=float, help='G3 width [clock]', 
                        default=args_dict['g3_clock'])
    parser.add_argument('--light_clock', type=float, help='Light width [clock]', 
                        default=args_dict['light_clock'])
    parser.add_argument('--interval_clock', type=float, help='interval clock width [clock]', 
                        default=args_dict['interval_clock'])
    parser.add_argument('--integration_count', type=int, help='The number of integration',
                        default=args_dict['integration_count'])
    
    args = parser.parse_args(args=[])

    return args

def const_RevLUT_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--I_0', type=float, help='Amplitude of emitted light', default=0.008)
    parser.add_argument('--Q1_max', type=int, help='The maximum value of Q1', default=150)
    parser.add_argument('--Q1_min', type=int, help='The minimum value of Q1', default=0)
    parser.add_argument('--Q1_size', type=int, help='The size of Q1', default=150)
    
    parser.add_argument('--Q2_max', type=int, help='The maximum value of Q2', default=3000)
    parser.add_argument('--Q2_min', type=int, help='The minimum value of Q2', default=0)
    parser.add_argument('--Q2_size', type=int, help='The size of Q2', default=600)
    
    parser.add_argument('--Q3_max', type=int, help='The maximum value of Q3', default=2000)
    parser.add_argument('--Q3_min', type=int, help='The minimum value of Q3', default=0)
    parser.add_argument('--Q3_size', type=int, help='The size of Q3', default=400)
    
    parser.add_argument('--gpu_id', type=int, help='GPU', default=1)
    parser.add_argument('--LUT_dir', type=str, help='path to lookup table', default='/data/daiki-kij/LUT/demo/wide/')
    parser.add_argument('--result_dir', type=str, help='path to data dir', default='/data/daiki-kij/defog/RevLUT/')
    parser.add_argument('--is_save', type=strtobool, default=True)
    
    return parser.parse_args()