import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz


def light_wave(light_time, temporal_size):
    lrow = np.zeros(temporal_size)
    lcol = np.zeros(temporal_size)
    lrow[0] = 1
    lcol[0:0+light_time] = 1
    light_m = toeplitz(lrow, lcol).T

    return light_m

def light_wave_real(lightwave, temporal_size):
    lrow = np.zeros(temporal_size)
    lcol = np.zeros(temporal_size)
    lrow[0] = lightwave[0]
    lcol[0:len(lightwave)] = lightwave
    light_m = toeplitz(lrow, lcol).T

    return light_m

def return_Q(root, x_start=0, x_range=328, y_start=0, y_range=239, offset=250):
        return np.stack([np.asarray(pd.read_csv(os.path.join(root, 'Q1Ave.csv'), delimiter=",")),
                         np.asarray(pd.read_csv(os.path.join(root, 'Q2Ave.csv'), delimiter=",")),
                         np.asarray(pd.read_csv(os.path.join(root, 'Q3Ave.csv'), delimiter=","))], 
                        axis=2)[:, y_start:y_start+y_range, x_start:x_start+x_range] - offset
    
def rotate_Q(root, rot=1, x_start=40, x_range=160, y_start=208, y_range=120, offset=250):
    return np.rot90([np.asarray(pd.read_csv(os.path.join(root, 'Q1Ave.csv'), delimiter=",")),
                        np.asarray(pd.read_csv(os.path.join(root, 'Q2Ave.csv'), delimiter=",")),
                        np.asarray(pd.read_csv(os.path.join(root, 'Q3Ave.csv'), delimiter=","))
                     ],
                     rot, (1,2))[:, y_start:y_start+y_range, x_start:x_start+x_range].transpose(1, 2, 0) - offset

def rotate_Q_pgm(root, rot=1, x_start=40, x_range=160, y_start=208, y_range=120, offset=250):
    return np.rot90([cv2.imread(os.path.join(root, 'G1_000.pgm'), cv2.IMREAD_UNCHANGED),
                     cv2.imread(os.path.join(root, 'G2_000.pgm'), cv2.IMREAD_UNCHANGED),
                     cv2.imread(os.path.join(root, 'G3_000.pgm'), cv2.IMREAD_UNCHANGED)
                    ],
                    rot, (1,2))[:, y_start:y_start+y_range, x_start:x_start+x_range].transpose(1, 2, 0) - offset

def obs_data_path(exp_data_root, exp_type, reg_type):
    
    path_dict = {}
    for exp in exp_type:
        for reg in reg_type:
            exp_path = os.path.join(exp_data_root, exp)
            path = os.path.join(exp_path, reg)
            path_dict[(exp, reg)] = path
        
    return path_dict

def obs_data_loader(exp_data_root="/data/daiki-kij/defog/obseravation/20200422/defog",
                    exp_type=None,
                    reg_type=['ours'], 
                    reg_amb_type=['ambient_ours'], 
                    rot=1,
                    x_start=40, x_range=160, y_start=208, y_range=120, ch=3):
    row = y_range
    col = x_range

    if exp_type is None:
        exp_type = np.sort(os.listdir(exp_data_root))

    path_dict = obs_data_path(exp_data_root, exp_type, reg_type)
    path_amb_dict = obs_data_path(exp_data_root, exp_type, reg_amb_type)

    Qs = np.zeros([row, col, ch, len(exp_type), len(reg_type)])

    for e, e_type in enumerate(exp_type):
        for r, r_type in enumerate(reg_type):
            try:
                Qs[:, :, :, e, r] = (rotate_Q(path_dict[(str(e_type), str(r_type))], rot=rot, x_start=x_start, x_range=x_range, y_start=y_start, y_range=y_range)) - rotate_Q(path_amb_dict[(str(e_type), reg_amb_type[r])],rot=rot, x_start=x_start, x_range=x_range, y_start=y_start, y_range=y_range)
            except FileNotFoundError:
                print(exp_type[e], reg_type[r])
                
    return Qs

def cal_raw_depth_form_Q(Q, light_clock=10, clock_ns=2.65, c=3e8):
    if Q.shape[2] == 2:
        return (Q[:, :, 1] / (Q[:, :, 0] + Q[:, :, 1])) * c * light_clock * clock_ns * 1e-9 / 2 
    if Q.shape[2] == 3:
        return (Q[:, :, 2]-Q[:, :, 0]) / (Q[:, :, 1]-Q[:, :, 0] + Q[:, :, 2]-Q[:, :, 0]) * c * light_clock * clock_ns * 1e-9 / 2 
    
def cal_intensity_form_Q(Q):
    if Q.shape[2] == 2:
        return Q[:, :, 0] + Q[:, :, 1]
    if Q.shape[2] == 3:
        return Q[:, :, 1] + Q[:, :, 2] - 2 * Q[:, :, 0]

def gate_calib(Q_path,
               Q_1m_path='/data/daiki-kij/defog/obseravation/20200417/calib/exp01/default',
               Q_2m_path='/data/daiki-kij/defog/obseravation/20200417/calib/exp11/default', 
               light_clock=13):
    
    Q = return_Q(Q_path)
    row, col, _ = Q.shape
    Q = Q.reshape(row*col, -1)
    Q_1m = return_Q(Q_1m_path).reshape(row*col, -1)
    Q_2m = return_Q(Q_2m_path).reshape(row*col, -1)
    
    Dist_raw = cal_raw_depth_form_Q(Q, light_clock=light_clock)
    Dist_raw_1m = cal_raw_depth_form_Q(Q_1m, light_clock=light_clock)
    Dist_raw_2m = cal_raw_depth_form_Q(Q_2m, light_clock=light_clock)
    
    delta = Dist_raw_2m - Dist_raw_1m
    SurCoeff = delta / np.median(delta)
    dist_gcalib = (Dist_raw - Dist_raw_1m) / SurCoeff + 0.97
    
    return dist_gcalib.reshape(row, col)

def dist_calib(dist_gcalibs):
    row, col = 239, 328
    a = np.zeros([row, col])
    b = np.zeros([row, col])

    x = dist_gcalibs
    y = np.zeros([row, col, 13])
    for i in range(13):
        y[:, :, i] = 0.87 + 0.1 * i

    for i in range(row):
        for j in range(col): 
            a[i,j], b[i,j] = np.polyfit(x[i,j], y[i,j], 1)
            
    return a, b

def return_settings(exp_name, 
                    gate_start=17, Light_start=20, 
                    offset=5, 
                    G1_width=10, G2_width=10, G3_width=10, Light_width=11, 
                    interval = 0):
    
    G1_start = gate_start
    G2_start = G1_start + G1_width + interval
    G3_start = G2_start + G2_width + interval
    GD_high = G3_start + G3_width + 2

    GD_low = gate_start - 2
    GD_width = GD_high - GD_low
    Light_end = Light_start + Light_width
    
    return G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD_low, GD_width

def set_reg(exp_name, reg_root,
            gate_start=17, Light_start=20, 
            offset=5, 
            G1_width=10, G2_width=10, G3_width=10, Light_width=11, 
            interval = 0):
    
    exp_path = os.path.join(reg_root, "reg_" + exp_name + ".txt")
    
    G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD_low, GD_width \
    = return_settings(exp_name, 
                      gate_start=gate_start, Light_start=Light_start, 
                      offset=offset, 
                      G1_width=G1_width, G2_width=G2_width, G3_width=G3_width, Light_width=Light_width, 
                      interval=interval)
    reg_dict = return_reg( 
                   G1_start, G2_start, G3_start, 
                   G1_width, G2_width, G3_width, 
                   Light_start, Light_end, 
                   GD_low, GD_width)
    write_reg(reg_dict, exp_path)
    print("saved to ", exp_path)
    G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD1_low, GD1_width=decode_reg(exp_path)
    drawClocks([(int(GD1_low), int(GD1_low) + int(GD1_width)), 
                (int(G3_start), int(G3_start) + int(G3_width)), 
                (int(G2_start), int(G2_start) + int(G2_width)), 
                (int(G1_start), int(G1_start) + int(G1_width)), 
                (int(Light_start)+5.0, int(Light_end)+5.0), 
                (int(Light_start), int(Light_end))
                ],
                [0, 100], 
                exp_name
              )
        
def return_reg(G1_start, G2_start, G3_start, 
               G1_width, G2_width, G3_width, Light_start, Light_end, 
               GD1_low, GD1_width, GD2_low=255, GD2_width=39, GD3_low=255, GD3_width=39, 
               UIC_ck=111, 
              ):
    
    reg_dict = {
            "04": "CF",
            "2A": "00",
            "2B": format(UIC_ck, '02x'), # Time budget of unit integration cycle (1 UIC [ck]), set 111 ('0x6f')!
            "34": "00",
            "35": format(G1_start, '02x'),
            "36": "00",
            "37": format(G1_width, "02x"),
            "38": "00",
            "39": format(G2_start, "02x"),
            "3A": "00",
            "3B": format(G2_width, '02x'),
            "3C": "00",
            "3D": format(G3_start, '02x'),
            "3E": "00",
            "3F": format(G3_width, "02x"),
            "40": "00",
            "41": format(Light_start, '02x'),
            "42": "00",
            "43": format(Light_end, "02x"),
            "30": "00",
            "31": format(GD1_low, "02x"),
            "32": "00",
            "33": format(GD1_width, "02x"),
            "B9": "83",
            "BA": format(GD2_low, "02x"), # if don't use GD2, set 255 ('0xff')!
            "BB": "00",
            "BC": format(GD2_width, "02x"),# if don't use GD2, set 39 ('0x27')!
            "BD": "83",
            "BE": format(GD3_low, "02x"), # if don't use GD3, set 255 ('0xff')!
            "BF": "00",
            "C0": format(GD3_width, "02x"),# if don't use GD3, set 39 ('0x27')!
            "A2": "38",
            "A6": "6A",
            "A7": "95",
            "A8": "3C",
            "B4": "10", #Reserved
    }

    return reg_dict
     
def decode_reg(path, is_print=False):
    reg_dict = reg2dict(path)
    G1_start = format(int(reg_dict["35"], 16))
    G2_start = format(int(reg_dict["39"], 16))
    G3_start = format(int(reg_dict["3D"], 16))
    G1_width = format(int(reg_dict["37"], 16))
    G2_width = format(int(reg_dict["3B"], 16))
    G3_width = format(int(reg_dict["3F"], 16))

    Light_start = format(int(reg_dict["41"], 16))
    Light_end = format(int(reg_dict["43"], 16))
    GD1_low = format(int(reg_dict["31"], 16))
    GD1_width = format(int(reg_dict["33"], 16))
    GD2_low = format(int(reg_dict["BA"], 16))
    GD2_width = format(int(reg_dict["BC"], 16))

    if is_print:
        print("G1_start:", G1_start)
        print("G2_start:", G2_start)
        print("G3_start:", G3_start)
        print("G1_width:", G1_width)
        print("G2_width:", G2_width)
        print("G3_width:", G3_width)
        print("Light_start:", Light_start)
        print("Light_end:", Light_end)
        print("GD1_low:", GD1_low)
        print("GD1_width:", GD1_width)
        
    return G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD1_low, GD1_width
        
def write_reg(reg_dict, path):
    with open(path, mode='w') as f:
        for k, v in reg_dict.items():
            f.write(k + " " + v + "\n")

def reg2dict(path):
    reg_dict = {}
    with open(path) as f:
        file = f.read()
        line = file.split('\n')
        for l in line:
            s = l.split(" ")
            if s[0] != "":
                reg_dict[s[0]] = s[1]
            else:
                break
    return reg_dict
    
def drawClocks(signals, xrange=[0, 111], title=''):
    signals = np.array(signals)
    if xrange == 'auto':
        drawRange = (np.min(signals[:,0])-1,np.max(signals[:,1])+1)
    else:
        drawRange = xrange
    assert(drawRange[0]>=0)

    clockPlots = [[
        [drawRange[0],s[0],s[0],s[1],s[1],drawRange[1]],
        [0,0,1,1,0,0]
    ] for s in signals]

    for n in range(len(clockPlots)):
        offsetY = n*2
        plt.plot(clockPlots[n][0],np.array(clockPlots[n][1])+offsetY)
        plt.xlabel('ck')
            
    
    plt.title(title)
    plt.show()
    plt.close()
    
def return_settings(exp_name, 
                    gate_start=17, Light_start=20, 
                    offset=5, 
                    G1_width=10, G2_width=10, G3_width=10, Light_width=11, 
                    interval = 0):
    
    G1_start = gate_start
    G2_start = G1_start + G1_width + interval
    G3_start = G2_start + G2_width + interval
    GD_high = G3_start + G3_width + 2

    GD_low = gate_start - 2
    GD_width = GD_high - GD_low
    Light_end = Light_start + Light_width
    
    return G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD_low, GD_width

def set_reg(exp_name, reg_root,
            gate_start=17, Light_start=20, 
            offset=5, 
            G1_width=10, G2_width=10, G3_width=10, Light_width=11, 
            interval = 0):
    
    exp_path = os.path.join(reg_root, "reg_" + exp_name + ".txt")
    
    G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD_low, GD_width \
    = return_settings(exp_name, 
                      gate_start=gate_start, Light_start=Light_start, 
                      offset=offset, 
                      G1_width=G1_width, G2_width=G2_width, G3_width=G3_width, Light_width=Light_width, 
                      interval=interval)
    reg_dict = return_reg( 
                   G1_start, G2_start, G3_start, 
                   G1_width, G2_width, G3_width, 
                   Light_start, Light_end, 
                   GD_low, GD_width)
    write_reg(reg_dict, exp_path)
    print("saved to ", exp_path)
    G1_start, G2_start, G3_start, G1_width, G2_width, G3_width, Light_start, Light_end, GD1_low, GD1_width=decode_reg(exp_path)
    drawClocks([(int(GD1_low), int(GD1_low) + int(GD1_width)), 
                (int(G3_start), int(G3_start) + int(G3_width)), 
                (int(G2_start), int(G2_start) + int(G2_width)), 
                (int(G1_start), int(G1_start) + int(G1_width)), 
                (int(Light_start)+5.0, int(Light_end)+5.0), 
                (int(Light_start), int(Light_end))
                ],
                [0, 100], 
                exp_name
              )