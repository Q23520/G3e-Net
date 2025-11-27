import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from configs import Config
from utils.G3e_Net_utils import (
    EndPoint_Distance,
    EndPoint_Distance_CarotidArtery,
    compute_space_distances,
    compute_space_distances_CaritidArtery,
    Avg_Centerline_Graph,
    Point_CarotidArtery_Centerline,
    compute_smoothed_normals_multiscale,
    compute_curvatures_multiscale,
)


def Point_Centerline_Graph(file, folderpath, centerlinepath, Node, mode, Nodenumber, Timenumber, iter):
    """
    Load raw centerline-aligned samples and resample them into fixed-length batches.

    Args:
        file: path to the raw simulation folder
        folderpath: Excel file that lists case folders
        centerlinepath: directory containing centerline files
        Node: either 'observation' or 'boundary'
        mode: 'train' or 'valid'
        Nodenumber: number of points sub-sampled per iteration
        Timenumber: temporal index to read
        iter: number of resampling rounds per case

    Returns:
        Tuple[np.ndarray, np.ndarray]: stacked samples and centerline tensors
    """
    if mode == 'train':
        data_folder = pd.read_excel(folderpath, sheet_name=0)
    else:
        data_folder = pd.read_excel(folderpath, sheet_name=1)
    
    for j in range(0, data_folder.size):
        folderlist = data_folder.values.tolist()
        foldername = str(folderlist[j]).replace("[", "").replace("]", "").replace("'", "")
        i = Timenumber
        step = str(i)
        
        if i < 10:
            path = os.path.join(file, foldername, 'step00' + step + '_' + Node + '_nodes.txt')
        elif i > 100:
            path = os.path.join(file, foldername, 'step' + step + '_' + Node + '_nodes.txt')
        else:
            path = os.path.join(file, foldername, 'step0' + step + '_' + Node + '_nodes.txt')
        data_step = np.loadtxt(path)
        
        spaces = data_step[:, :3]
        # Standardize xyz
        mean_space = np.mean(spaces, axis=0)
        std_space = np.std(spaces, axis=0)
        spaces_ = (spaces - mean_space) / std_space
        
        times = data_step[:, 3]
        times = times.reshape(len(times), 1)
        
        # Centerline sampling
        path_centerline = os.path.join(centerlinepath, foldername + '_Centerline_pro.txt')
        centerline = np.loadtxt(path_centerline)
        centerline_parameter, lengths = EndPoint_Distance(
            centerline, num_points=3000, distance_threshold=0.01, k=5, threshold_ratio=1.5
        )
        
        if centerline_parameter.shape[0] < 3000:
            print(f"Warning: insufficient centerline samples in {path_centerline}")
        
        segment_indices, segment_distances = compute_space_distances(spaces, centerline_parameter, lengths)
        
        N, A = centerline_parameter.shape
        cline_resample = centerline_parameter.reshape(-1, N, A)
        
        if Node == 'observation':
            # Observation nodes carry velocity + pressure
            pressure = data_step[:, 8]
            pressure = pressure.reshape(len(pressure), 1)
            mean_pressure = np.mean(pressure, axis=0)
            std_pressure = np.std(pressure, axis=0)
            pressure_ = (pressure - mean_pressure) / std_pressure
            
            velocity = data_step[:, 4:7]
            mean_velocity = np.mean(velocity, axis=0)
            std_velocity = np.std(velocity, axis=0)
            velocity_ = (velocity - mean_velocity) / std_velocity
            
            data_array = np.concatenate((spaces_, segment_indices, segment_distances, times, velocity_, pressure_), 1)
        else:
            # Boundary nodes require curvature descriptors
            normals, strengths = compute_smoothed_normals_multiscale(spaces, scales=[10, 30, 60])
            k1, k2, H, G = compute_curvatures_multiscale(spaces, normals, scales=[100, 200, 300])
            
            pressure = data_step[:, 8]
            pressure = pressure.reshape(len(pressure), 1)
            mean_pressure = np.mean(pressure, axis=0)
            std_pressure = np.std(pressure, axis=0)
            pressure_ = (pressure - mean_pressure) / std_pressure
            
            velocity = data_step[:, 4:7]
            data_array = np.concatenate((spaces_, segment_indices, segment_distances, times, H, velocity, pressure_), 1)
        
        # Resample and expand
        N, A = data_array.shape
        data_array = data_array.reshape(-1, N, A)
        itnumber = iter
        cline_resample = np.concatenate(([cline_resample] * itnumber), 0)
        
        _, Len, _ = data_array.shape
        idx = np.random.choice(Len, (Nodenumber * itnumber), replace=True)
        data_sample = data_array[:, idx, :]
        
        for i in range(0, itnumber):
            if i == 0:
                data_Node = data_sample[:, 0:Nodenumber, :]
            else:
                data_Node = np.concatenate((data_Node, data_sample[:, (i * Nodenumber):((i + 1) * Nodenumber), :]), axis=0)
        
        if j == 0:
            data_total = data_Node
            centerline_total = cline_resample
        else:
            data_total = np.concatenate((data_total, data_Node), axis=0)
            centerline_total = np.concatenate((centerline_total, cline_resample), axis=0)
    
    return data_total, centerline_total


class Data_Centerline_Graph(Dataset):
    """Dataset wrapper for the abdominal (generic) centerline training regime."""
    
    def __init__(self, config, mode, type, Nodenumber, Timenumber, angle):
        self.data_file = config.dataset['data_file_txt']
        self.folderpath = config.dataset['data_folder_path']
        self.centerlinepath = config.dataset['centerline_file']
        self.iter = config.dataset['iter']
        self.type = type
        self.mode = mode
        self.dims = config.dataset['space_dims']
        
        self.sim_data, self.centerline_parameter = Avg_Centerline_Graph(
            self.data_file, self.folderpath, self.centerlinepath,
            self.type, self.mode, Nodenumber, Timenumber, self.iter
        )
        self.obv_data = self.sim_data
        self.cparameter = self.centerline_parameter
    
    def __getitem__(self, index):
        obv = self.obv_data[index % len(self.obv_data)]
        cparameter = self.cparameter[index % len(self.cparameter)]
        
        if self.type == 'observation':
            input_idx_ob = 12
            output_idx_ob = 15
            inputs = np.array(obv[:, :input_idx_ob])
            targets = np.array(obv[:, input_idx_ob:output_idx_ob])
            hiddens = np.array(obv[:, output_idx_ob:])
            inputs = [torch.FloatTensor(f) for f in np.hsplit(inputs, np.arange(11) + 1)]
            targets = [torch.FloatTensor(f) for f in np.hsplit(targets, np.arange(2) + 1)]
            hiddens = torch.FloatTensor(hiddens)
            centerline = [torch.FloatTensor(f) for f in np.hsplit(cparameter, np.arange(4) + 1)]
            return inputs, targets, hiddens, centerline
        else:
            input_idx_bo = 13
            output_idx_bo = 16
            inputs = np.array(obv[:, :input_idx_bo])
            targets = np.array(obv[:, input_idx_bo:output_idx_bo])
            hiddens = np.array(obv[:, output_idx_bo:])
            inputs = [torch.FloatTensor(f) for f in np.hsplit(inputs, np.arange(12) + 1)]
            targets = [torch.FloatTensor(f) for f in np.hsplit(targets, np.arange(2) + 1)]
            hiddens = torch.FloatTensor(hiddens)
            centerline = [torch.FloatTensor(f) for f in np.hsplit(cparameter, np.arange(4) + 1)]
            return inputs, targets, hiddens, centerline
    
    def __len__(self):
        return len(self.obv_data)


class CarotidArtery_Centerline(Dataset):
    """Dataset wrapper for the carotid-specific training configuration."""
    
    def __init__(self, config, mode, type, Nodenumber, Timenumber, angle):
        self.data_file = config.dataset['data_file_txt']
        self.folderpath = config.dataset['data_folder_path']
        self.centerlinepath = config.dataset['centerline_file']
        self.iter = config.dataset['iter']
        self.type = type
        self.mode = mode
        self.dims = config.dataset['space_dims']
        
        self.sim_data, self.centerline_parameter = Point_CarotidArtery_Centerline(
            self.data_file, self.folderpath, self.centerlinepath,
            self.type, self.mode, Nodenumber, Timenumber, self.iter
        )
        self.obv_data = self.sim_data
        self.cparameter = self.centerline_parameter
    
    def __getitem__(self, index):
        obv = self.obv_data[index % len(self.obv_data)]
        cparameter = self.cparameter[index % len(self.cparameter)]
        
        if self.type == 'observation':
            input_idx_ob = 9
            output_idx_ob = 12
            inputs = np.array(obv[:, :input_idx_ob])
            targets = np.array(obv[:, input_idx_ob:output_idx_ob])
            hiddens = np.array(obv[:, output_idx_ob:])
            inputs = [torch.FloatTensor(f) for f in np.hsplit(inputs, np.arange(8) + 1)]
            targets = [torch.FloatTensor(f) for f in np.hsplit(targets, np.arange(2) + 1)]
            hiddens = torch.FloatTensor(hiddens)
            centerline = [torch.FloatTensor(f) for f in np.hsplit(cparameter, np.arange(4) + 1)]
            return inputs, targets, hiddens, centerline
        else:
            input_idx_bo = 10
            output_idx_bo = 13
            inputs = np.array(obv[:, :input_idx_bo])
            targets = np.array(obv[:, input_idx_bo:output_idx_bo])
            hiddens = np.array(obv[:, output_idx_bo:])
            inputs = [torch.FloatTensor(f) for f in np.hsplit(inputs, np.arange(9) + 1)]
            targets = [torch.FloatTensor(f) for f in np.hsplit(targets, np.arange(2) + 1)]
            hiddens = torch.FloatTensor(hiddens)
            centerline = [torch.FloatTensor(f) for f in np.hsplit(cparameter, np.arange(4) + 1)]
            return inputs, targets, hiddens, centerline
    
    def __len__(self):
        return len(self.obv_data)

