from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from configs import Config
from nn import DGCNN_CenterGraph, CarotidArtery_CenterGraph
from utils.G3e_Net_utils import (
    Centerline_sampling,
    CarotidArtery_Centerline_Sample,
)

def write_log(directory: str, message: str) -> None:
    """Append a single line to `log.txt` inside `directory`."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "log.txt"), "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def ThreetoTow(threedim: np.ndarray) -> np.ndarray:
    """Flatten a (B, N, C) array into (N, B*C) to match legacy expectations."""
    _, length, _ = threedim.shape
    return np.reshape(threedim, (length, -1))


def data_ori(orig: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    """Inverse the z-score normalization using the statistics of `orig`."""
    mean_space = np.mean(orig, axis=0)
    std_space = np.std(orig, axis=0)
    return normalized * std_space + mean_space


def L2_Nrom(result: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs(result - target)


def Data_MAE(result: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(result - target)))


def Data_NMAE(result: np.ndarray, target: np.ndarray) -> float:
    denom = np.max(result) - np.min(result)
    return float(np.mean(np.abs(result - target) / (denom + 1e-8)))


def Data_RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def Data_NRMSE(y_true: np.ndarray, y_pred: np.ndarray, method: str = "min-max") -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if method == "min-max":
        norm_factor = np.max(y_true) - np.min(y_true)
    elif method == "mean":
        norm_factor = np.mean(y_true)
    elif method == "std":
        norm_factor = np.std(y_true)
    else:
        raise ValueError("method must be one of {'min-max','mean','std'}")
    return float(rmse / (norm_factor + 1e-8))


def compute_wall_shear_stress(
    surface_points: np.ndarray,
    internal_points: np.ndarray,
    velocities: np.ndarray,
    mu: float = 0.365,
    k_neighbors: int = 500,
) -> np.ndarray:
    """Compute wall shear stress magnitude following the legacy implementation."""
    tau_w = []
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(internal_points)

    for surface_point in surface_points:
        _, indices = neigh.kneighbors([surface_point])
        nearest_points = internal_points[indices[0]]
        nearest_velocities = velocities[indices[0]]

        pca = PCA(n_components=3)
        pca.fit(nearest_points - surface_point)
        normal = pca.components_[-1]

        A = nearest_points - surface_point
        B = nearest_velocities - nearest_velocities.mean(axis=0)
        try:
            grad_v, *_ = np.linalg.lstsq(A, B, rcond=None)
        except np.linalg.LinAlgError:
            grad_v = np.zeros((3, 3))

        tau_vec = mu * (grad_v @ normal)
        tau_w.append(np.linalg.norm(tau_vec))

    return np.array(tau_w).reshape(-1, 1)


def save_solution_FT(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    t_x: np.ndarray,
    t_y: np.ndarray,
    t_z: np.ndarray,
    p_u: np.ndarray,
    p_v: np.ndarray,
    p_w: np.ndarray,
    p_p: np.ndarray,
    t_u: np.ndarray,
    t_v: np.ndarray,
    t_w: np.ndarray,
    t_p: np.ndarray,
    num: int,
    log: str,
    foldername: str,
    boundary: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    """
    Persist evaluation outputs (predictions, targets, RMSE) alongside summary
    metrics. Matches the behaviour of utils.save_solution_FT.
    """
    r_x = data_ori(x, t_x)
    r_y = data_ori(y, t_y)
    r_z = data_ori(z, t_z)
    r_tu = data_ori(u, t_u)
    r_tv = data_ori(v, t_v)
    r_tw = data_ori(w, t_w)
    r_tp = data_ori(p, t_p)
    r_pu = data_ori(u, p_u)
    r_pv = data_ori(v, p_v)
    r_pw = data_ori(w, p_w)
    r_p = data_ori(p, p_p)

    v_magnitude_p = np.sqrt(np.square(r_pu) + np.square(r_pv) + np.square(r_pw))
    v_magnitude_r = np.sqrt(np.square(r_tu) + np.square(r_tv) + np.square(r_tw))

    tau_w_result = compute_wall_shear_stress(boundary, np.concatenate((r_x, r_y, r_z), axis=1),
                                             np.concatenate((r_pu, r_pv, r_pw), axis=1))
    tau_w_target = compute_wall_shear_stress(boundary, np.concatenate((r_x, r_y, r_z), axis=1),
                                             np.concatenate((r_tu, r_tv, r_tw), axis=1))

    result = np.concatenate((r_x, r_y, r_z, r_pu, r_pv, r_pw, v_magnitude_p, r_p), axis=1)
    target = np.concatenate((r_x, r_y, r_z, r_tu, r_tv, r_tw, v_magnitude_r, r_tp), axis=1)
    rmse = np.concatenate(
        (
            r_x,
            r_y,
            r_z,
            L2_Nrom(r_pu, r_tu),
            L2_Nrom(r_pv, r_tv),
            L2_Nrom(r_pw, r_tw),
            L2_Nrom(v_magnitude_p, v_magnitude_r),
            L2_Nrom(r_p, r_tp),
        ),
        axis=1,
    )

    result_boundary = np.concatenate((boundary[:, 0:1], boundary[:, 1:2], boundary[:, 2:3], tau_w_result), axis=1)
    target_boundary = np.concatenate((boundary[:, 0:1], boundary[:, 1:2], boundary[:, 2:3], tau_w_target), axis=1)
    rmse_boundary = np.concatenate(
        (boundary[:, 0:1], boundary[:, 1:2], boundary[:, 2:3], L2_Nrom(tau_w_result, tau_w_target)), axis=1
    )

    save_path = f"/home/qly/FixedT/DGCNN_SDP/Result/Data/{foldername}/"
    os.makedirs(save_path, exist_ok=True)
    write_log(save_path, log)

    pd.DataFrame(result, columns=["X", "Y", "Z", "U", "V", "W", "Velocity Magnitude", "Pressure"]).to_csv(
        os.path.join(save_path, f"Result_{num}.csv"), index=False
    )
    pd.DataFrame(target, columns=["X", "Y", "Z", "U", "V", "W", "Velocity Magnitude", "Pressure"]).to_csv(
        os.path.join(save_path, f"Tar_{num}.csv"), index=False
    )
    pd.DataFrame(rmse, columns=["X", "Y", "Z", "U", "V", "W", "Velocity Magnitude", "Pressure"]).to_csv(
        os.path.join(save_path, f"Rmse_{num}.csv"), index=False
    )

    pd.DataFrame(result_boundary, columns=["X", "Y", "Z", "Wall Shear Stress Magnitude"]).to_csv(
        os.path.join(save_path, f"Boundary_Result_{num}.csv"), index=False
    )
    pd.DataFrame(target_boundary, columns=["X", "Y", "Z", "Wall Shear Stress Magnitude"]).to_csv(
        os.path.join(save_path, f"Boundary_Tar_{num}.csv"), index=False
    )
    pd.DataFrame(rmse_boundary, columns=["X", "Y", "Z", "Wall Shear Stress Magnitude"]).to_csv(
        os.path.join(save_path, f"Boundary_Rmse_{num}.csv"), index=False
    )

    mae_vm = Data_MAE(v_magnitude_p, v_magnitude_r)
    mae_p = Data_MAE(r_p, r_tp)
    mae_wm = Data_MAE(tau_w_result, tau_w_target)
    nmae_vm = Data_NMAE(v_magnitude_p, v_magnitude_r)
    nmae_p = Data_NMAE(r_p, r_tp)
    nmae_wm = Data_NMAE(tau_w_result, tau_w_target)
    rmse_v = Data_RMSE(v_magnitude_r, v_magnitude_p)
    rmse_p = Data_RMSE(r_tp, r_p)
    rmse_wm = Data_RMSE(tau_w_result, tau_w_target)
    nrmse_v = Data_NRMSE(v_magnitude_r, v_magnitude_p)
    nrmse_p = Data_NRMSE(r_tp, r_p)
    nrmse_wm = Data_NRMSE(tau_w_result, tau_w_target)

    return (
        mae_vm,
        mae_p,
        mae_wm,
        nmae_vm,
        nmae_p,
        nmae_wm,
        rmse_v,
        rmse_p,
        rmse_wm,
        nrmse_v,
        nrmse_p,
        nrmse_wm,
    )


def split_and_filter_data(data_sample: np.ndarray, centerline_parameter: np.ndarray) -> np.ndarray:
    """Match each observation with the relevant centerline segments."""
    _, num_points, num_attrs = data_sample.shape
    flattened = data_sample.reshape(num_points, num_attrs)
    groups = {label: [] for label in range(4)}

    for row in centerline_parameter:
        group_id = int(row[3])
        groups[group_id].append(row)

    for key in groups:
        groups[key] = np.array(groups[key]) if groups[key] else np.empty((0, centerline_parameter.shape[1]))

    result = []
    for group_id in range(4):
        for row in flattened:
            index = int(row[group_id + 3])
            if 0 <= index < len(groups[group_id]):
                selected = groups[group_id][index]
                result.append(np.concatenate((selected, row[:3])))

    if not result:
        return np.empty((0, num_points, centerline_parameter.shape[1] + 3))

    result = np.array(result)
    return result.reshape(-1, result.shape[0], result.shape[1])


# NOTE: The geometry-heavy helpers (centerline sampling, curvature analysis,
#       etc.) remain imported from `utils.utils` to avoid duplicating several
#       hundred lines of numerical routines. The interfaces are wrapped above
#       so the calling code can stay clean.


# --------------------------------------------------------------------------------------
# Evaluation pipeline
# --------------------------------------------------------------------------------------


def chunk_observations(data: np.ndarray, interval: int) -> Iterator[np.ndarray]:
    """Yield contiguous slices of `data` using `interval` as the window size."""
    _, length, _ = data.shape
    for start in range(0, length, interval):
        end = min(length, start + interval)
        yield data[:, start:end, :]


def sample_boundary_points(boundary: np.ndarray, sample_size: int) -> np.ndarray:
    """Randomly sample boundary points with replacement."""
    _, count, _ = boundary.shape
    indices = np.random.choice(count, sample_size, replace=True)
    return boundary[:, indices, :]


@dataclass
class EvalCase:
    folder: str
    observation: np.ndarray
    boundary: np.ndarray
    centerline_ob: np.ndarray
    centerline_bo: np.ndarray
    space_ob: np.ndarray
    boundary_space: np.ndarray
    velocity_ob: np.ndarray
    pressure_ob: np.ndarray


class BaseEvaluator:
    """Shared evaluation logic for both historical test scripts."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = Config.load(args.config)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.interval = args.interval
        self.boundary_samples = args.boundary_samples

    def run(self) -> None:
        folders = pd.read_excel(self.config.dataset["data_folder_path"], sheet_name=2).values.tolist()
        save_root = os.path.join(self.args.output_dir, "Plot")
        os.makedirs(save_root, exist_ok=True)
        writer = SummaryWriter(os.path.join(save_root, "Board"))

        aggregate_metrics = self._init_metric_accumulator()

        with torch.no_grad():
            for idx, entry in enumerate(folders):
                folder = str(entry).replace("[", "").replace("]", "").replace("'", "")
                case = self._load_case(folder)

                model = self._build_model().to(self.device)
                self._load_checkpoint(model)
                model.eval()
                torch.cuda.empty_cache()

                metrics = self._evaluate_case(model, case)
                self._log_case(folder, metrics, idx, writer)
                self._update_aggregate(aggregate_metrics, metrics)

        self._summarize_aggregate(aggregate_metrics, len(folders))
        writer.close()

    # ----- hooks -----
    def _load_case(self, folder: str) -> EvalCase:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _build_model(self) -> torch.nn.Module:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _evaluate_case(self, model: torch.nn.Module, case: EvalCase) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError

    # ----- shared helpers -----
    def _load_checkpoint(self, model: torch.nn.Module) -> None:
        checkpoint = torch.load(self.config.training["load_path"], map_location=self.device)
        model.load_state_dict(checkpoint["state"])
        if self.args.verbose:
            print(f"Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']}, LR {checkpoint['learningrate']}")
            print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    def _init_metric_accumulator(self) -> Dict[str, float]:
        return {}

    def _update_aggregate(self, aggregate: Dict[str, float], metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + value

    def _summarize_aggregate(self, aggregate: Dict[str, float], count: int) -> None:
        if not aggregate:
            return
        summary = ", ".join(f"{key}={value / count:.6f}" for key, value in aggregate.items())
        print(f"[AVERAGE] {summary}")

    def _log_case(self, folder: str, metrics: Dict[str, float], step: int, writer: SummaryWriter) -> None:
        message = f"{folder} - " + ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        print(message)
        write_log(os.path.join(self.args.output_dir, "Plot"), message)
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)


class CenterlineEvaluator(BaseEvaluator):
    """Replacement for the legacy `Test_DGCNN_Centerline.py` script."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.input_idx = 12

    def _load_case(self, folder: str) -> EvalCase:
        data_file = self.config.dataset["data_file_txt"]
        centerline_file = self.config.dataset["centerline_file"]

        obs, space_ob, vel_ob, p_ob, centerline_ob = Centerline_sampling(
            data_file, folder, "observation", self.config.dataset["Time_step"], centerline_file
        )
        bound, space_bo, vel_bo, p_bo, centerline_bo = Centerline_sampling(
            data_file, folder, "boundary", self.config.dataset["Time_step"], centerline_file
        )

        return EvalCase(
            folder=folder,
            observation=obs,
            boundary=bound,
            centerline_ob=centerline_ob,
            centerline_bo=centerline_bo,
            space_ob=space_ob,
            boundary_space=space_bo,
            velocity_ob=vel_ob,
            pressure_ob=p_ob,
        )

    def _build_model(self) -> torch.nn.Module:
        return DGCNN_CenterGraph(
            k=self.config.net["k"],
            dropout=self.config.net["dropout"],
            output_channels=self.config.net["output_channels"],
            layer_size_trunk=self.config.net["layer_size_trunk"],
            emb_dims=self.config.net["emb_dims"],
        )

    def _evaluate_case(self, model: torch.nn.Module, case: EvalCase) -> Dict[str, float]:
        res_u, res_v, res_w, res_p = [], [], [], []
        inputs = case.observation
        boundary = case.boundary

        for chunk in chunk_observations(inputs, self.interval):
            boundary_sample = sample_boundary_points(boundary, self.boundary_samples)
            data_centerline_ob = split_and_filter_data(chunk, case.centerline_ob)
            data_centerline_bo = split_and_filter_data(boundary_sample, case.centerline_bo)

            tensor_ob = torch.tensor(chunk).float().to(self.device)
            tensor_bo = torch.tensor(boundary_sample).float().to(self.device)
            tensor_center_ob = torch.tensor(data_centerline_ob).float().to(self.device)
            tensor_center_bo = torch.tensor(data_centerline_bo).float().to(self.device)

            pred_u, pred_v, pred_w, pred_p = model.predict(tensor_bo, tensor_ob, tensor_center_bo, tensor_center_ob)
            res_u.append(pred_u.detach().cpu().numpy())
            res_v.append(pred_v.detach().cpu().numpy())
            res_w.append(pred_w.detach().cpu().numpy())
            res_p.append(pred_p.detach().cpu().numpy())

        predictions = {
            "u": ThreetoTow(np.concatenate(res_u, axis=1)),
            "v": ThreetoTow(np.concatenate(res_v, axis=1)),
            "w": ThreetoTow(np.concatenate(res_w, axis=1)),
            "p": ThreetoTow(np.concatenate(res_p, axis=1)),
        }

        targets = {
            "u": ThreetoTow(np.array(case.observation[:, :, self.input_idx : self.input_idx + 1])),
            "v": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 1 : self.input_idx + 2])),
            "w": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 2 : self.input_idx + 3])),
            "p": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 3 :])),
        }

        metrics = {
            "MSE": float(
                np.mean(np.square(predictions["u"] - targets["u"]))
                + np.mean(np.square(predictions["v"] - targets["v"]))
                + np.mean(np.square(predictions["w"] - targets["w"]))
                + np.mean(np.square(predictions["p"] - targets["p"]))
            ),
            "MAE": float(
                np.mean(np.abs(predictions["u"] - targets["u"]))
                + np.mean(np.abs(predictions["v"] - targets["v"]))
                + np.mean(np.abs(predictions["w"] - targets["w"]))
                + np.mean(np.abs(predictions["p"] - targets["p"]))
            ),
        }

        log = (
            f"U -- {Data_MAE(predictions['u'], targets['u']):.6f}, "
            f"V -- {Data_MAE(predictions['v'], targets['v']):.6f}, "
            f"W -- {Data_MAE(predictions['w'], targets['w']):.6f}, "
            f"P -- {Data_MAE(predictions['p'], targets['p']):.6f}, "
            f"MAE -- {metrics['MAE']:.6f}, MSE -- {metrics['MSE']:.6f}"
        )
        write_log(os.path.join(self.args.output_dir, "Plot"), f"{case.folder}: {log}")

        mae_vm, mae_p, _, nmae_vm, nmae_p, _, *_ = save_solution_FT(
            x=np.array(case.space_ob[:, :1]),
            y=np.array(case.space_ob[:, 1:2]),
            z=np.array(case.space_ob[:, 2:3]),
            u=np.array(case.velocity_ob[:, :1]),
            v=np.array(case.velocity_ob[:, 1:2]),
            w=np.array(case.velocity_ob[:, 2:3]),
            p=case.pressure_ob,
            t_x=ThreetoTow(np.array(case.observation[:, :, 0:1])),
            t_y=ThreetoTow(np.array(case.observation[:, :, 1:2])),
            t_z=ThreetoTow(np.array(case.observation[:, :, 2:3])),
            p_u=predictions["u"],
            p_v=predictions["v"],
            p_w=predictions["w"],
            p_p=predictions["p"],
            t_u=targets["u"],
            t_v=targets["v"],
            t_w=targets["w"],
            t_p=targets["p"],
            num=self.config.dataset["Time_step"] + 1,
            log=log,
            foldername=case.folder,
            boundary=case.boundary_space,
        )

        metrics.update({"MAE_VM": mae_vm, "MAE_P": mae_p, "NMAE_VM": nmae_vm, "NMAE_P": nmae_p})
        return metrics


class CarotidEvaluator(BaseEvaluator):
    """Replacement for the legacy `Test_CarotidArtery.py` script."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.input_idx = 9

    def _load_case(self, folder: str) -> EvalCase:
        data_file = self.config.dataset["data_file_txt"]
        centerline_file = self.config.dataset["centerline_file"]

        obs, space_ob, vel_ob, p_ob, centerline_ob = CarotidArtery_Centerline_Sample(
            data_file, folder, "observation", self.config.dataset["Time_step"], centerline_file
        )
        bound, space_bo, vel_bo, p_bo, centerline_bo = CarotidArtery_Centerline_Sample(
            data_file, folder, "boundary", self.config.dataset["Time_step"], centerline_file
        )

        return EvalCase(
            folder=folder,
            observation=obs,
            boundary=bound,
            centerline_ob=centerline_ob,
            centerline_bo=centerline_bo,
            space_ob=space_ob,
            boundary_space=space_bo,
            velocity_ob=vel_ob,
            pressure_ob=p_ob,
        )

    def _build_model(self) -> torch.nn.Module:
        return CarotidArtery_CenterGraph(
            k=self.config.net["k"],
            dropout=self.config.net["dropout"],
            output_channels=self.config.net["output_channels"],
            n_features=self.config.net["n_features"],
            batch_size=self.config.net["batch_size"],
            hyper_parameters=self.config.net["hyper_parameters"],
            emb_dims=self.config.net["emb_dims"],
        )

    def _evaluate_case(self, model: torch.nn.Module, case: EvalCase) -> Dict[str, float]:
        res_u, res_v, res_w, res_p = [], [], [], []
        for chunk in chunk_observations(case.observation, self.interval):
            boundary_sample = sample_boundary_points(case.boundary, self.boundary_samples)
            tensor_ob = torch.tensor(chunk).float().to(self.device)
            tensor_bo = torch.tensor(boundary_sample).float().to(self.device)
            tensor_center_ob = torch.tensor(case.centerline_ob).float().to(self.device)
            tensor_center_bo = torch.tensor(case.centerline_bo).float().to(self.device)

            pred_u, pred_v, pred_w, pred_p = model.predict(tensor_bo, tensor_ob, tensor_center_bo, tensor_center_ob)
            res_u.append(pred_u.detach().cpu().numpy())
            res_v.append(pred_v.detach().cpu().numpy())
            res_w.append(pred_w.detach().cpu().numpy())
            res_p.append(pred_p.detach().cpu().numpy())

        predictions = {
            "u": ThreetoTow(np.concatenate(res_u, axis=1)),
            "v": ThreetoTow(np.concatenate(res_v, axis=1)),
            "w": ThreetoTow(np.concatenate(res_w, axis=1)),
            "p": ThreetoTow(np.concatenate(res_p, axis=1)),
        }

        targets = {
            "u": ThreetoTow(np.array(case.observation[:, :, self.input_idx : self.input_idx + 1])),
            "v": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 1 : self.input_idx + 2])),
            "w": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 2 : self.input_idx + 3])),
            "p": ThreetoTow(np.array(case.observation[:, :, self.input_idx + 3 :])),
        }

        mae_u = Data_MAE(predictions["u"], targets["u"])
        mae_v = Data_MAE(predictions["v"], targets["v"])
        mae_w = Data_MAE(predictions["w"], targets["w"])
        mae_p = Data_MAE(predictions["p"], targets["p"])
        mse = (
            np.mean(np.square(predictions["u"] - targets["u"]))
            + np.mean(np.square(predictions["v"] - targets["v"]))
            + np.mean(np.square(predictions["w"] - targets["w"]))
            + np.mean(np.square(predictions["p"] - targets["p"]))
        )
        log = (
            f"U -- {mae_u:.6f}, V -- {mae_v:.6f}, W -- {mae_w:.6f}, "
            f"P -- {mae_p:.6f}, MAE -- {(mae_u + mae_v + mae_w + mae_p):.6f}, MSE -- {mse:.6f}"
        )
        write_log(os.path.join(self.args.output_dir, "Plot"), f"{case.folder}: {log}")

        metrics_tuple = save_solution_FT(
            x=np.array(case.space_ob[:, :1]),
            y=np.array(case.space_ob[:, 1:2]),
            z=np.array(case.space_ob[:, 2:3]),
            u=np.array(case.velocity_ob[:, :1]),
            v=np.array(case.velocity_ob[:, 1:2]),
            w=np.array(case.velocity_ob[:, 2:3]),
            p=case.pressure_ob,
            t_x=ThreetoTow(np.array(case.observation[:, :, 0:1])),
            t_y=ThreetoTow(np.array(case.observation[:, :, 1:2])),
            t_z=ThreetoTow(np.array(case.observation[:, :, 2:3])),
            p_u=predictions["u"],
            p_v=predictions["v"],
            p_w=predictions["w"],
            p_p=predictions["p"],
            t_u=targets["u"],
            t_v=targets["v"],
            t_w=targets["w"],
            t_p=targets["p"],
            num=self.config.dataset["Time_step"] + 1,
            log=log,
            foldername=case.folder,
            boundary=case.boundary_space,
        )

        keys = [
            "MAE_VM",
            "MAE_P",
            "MAE_WM",
            "NMAE_VM",
            "NMAE_P",
            "NMAE_WM",
            "RMSE_VM",
            "RMSE_P",
            "RMSE_WM",
            "NRMSE_VM",
            "NRMSE_P",
            "NRMSE_WM",
        ]
        return dict(zip(keys, metrics_tuple))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation script for centerline models.")
    parser.add_argument("--mode", choices=["centerline", "carotid"], required=True, help="Which evaluation pipeline to run.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--interval", type=int, default=20000, help="Observation chunk size.")
    parser.add_argument("--boundary-samples", type=int, default=10000, help="Boundary sample count per chunk.")
    parser.add_argument("--output-dir", default="/home/qly/FixedT/DGCNN_SDP/Result", help="Directory for logs and boards.")
    parser.add_argument("--verbose", action="store_true", help="Print checkpoint metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator_cls = CenterlineEvaluator if args.mode == "centerline" else CarotidEvaluator
    evaluator = evaluator_cls(args)
    evaluator.run()


if __name__ == "__main__":
    main()

