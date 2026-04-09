# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

"""
Compute comprehensive evaluation metrics for station predictions.

This script parallelizes metric computation across multiple CPU cores
for efficient processing of rollout predictions.

Usage:
    python evaluate_station_metrics.py --p_preds /path/to/predictions.zarr [--max_cores 96]
"""

import numpy as np
import os
import pickle
import xarray as xr
import pandas as pd
from collections import defaultdict
import time
from joblib import Parallel, delayed
import multiprocessing
import argparse
import datetime


# Variable name mapping from abbreviation to full name
d_srf_abr2full = {
    "2t": "2m_temperature", 
    "10u": "10m_u_component_of_wind", 
    "10v": "10m_v_component_of_wind", 
    "msl": "mean_sea_level_pressure", 
    "tp": "total_precipitation",
    "tp_log": "total_precipitation_log",
    "tp_mswep": "total_precipitation_MSWEP",
    "tp_mswep_log": "total_precipitation_MSWEP_log",
    "pe": "potential_evaporation",
    "e": "evaporation",
    "r": "runoff",
    "swvl": "volumetric_soil_water_layer",
    "swvl_1": "volumetric_soil_water_layer_1",
    "swc": "soil_water_content",
    "tws": "terrestrial_water_storage",
    "tws_gou": "terrestrial_water_storage_Gou",
    "tws_itsg": "terrestrial_water_storage_ITSG",
    "slhf": "surface_latent_heat_flux",
    "sshf": "surface_sensible_heat_flux",
    "ssr": "surface_net_solar_radiation", 
    "str": "surface_net_thermal_radiation",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "tsr": "top_net_solar_radiation",
    "ttr": "top_net_thermal_radiation",
    "tisr": "toa_incident_solar_radiation",
    "sst": "sea_surface_temperature",
    "ci": "sea_ice_cover",
    "co2": "global_CO2",
    "pe_6hr": "potential_evaporation_6hr",
    "e_6hr": "evaporation_6hr",
    "r_6hr": "runoff_6hr",
    "tp_6hr": "total_precipitation_6hr",
    "slhf_6hr": "surface_latent_heat_flux_6hr",
    "sshf_6hr": "surface_sensible_heat_flux_6hr",
    "ssr_6hr": "surface_net_solar_radiation_6hr", 
    "str_6hr": "surface_net_thermal_radiation_6hr",
    "ssrd_6hr": "surface_solar_radiation_downwards_6hr",
    "strd_6hr": "surface_thermal_radiation_downwards_6hr",
    "tsr_6hr": "top_net_solar_radiation_6hr",
    "ttr_6hr": "top_net_thermal_radiation_6hr",
    "sf_6hr": "snowfall_6hr",
    "ir_mod": "MOD05_L2_avg_IR", 
    "nir_mod": "MOD05_L2_avg_NIR",
    "ir_myd": "MYD05_L2_avg_IR",
    "nir_myd": "MYD05_L2_avg_NIR",
    "u_10m_cosmo": "U_10M",
    "v_10m_cosmo": "V_10M",
    "vmax_10m": "VMAX_10M",
    "2d": "TD_2M",
    "sp": "PS",
    "clct_cosmo": "CLCT",
    "alhfl_s": "ALHFL_S",
    "ashfl_s": "ASHFL_S",
    "asob_s": "ASOB_S",
    "athb_s": "ATHB_S",
    "tp_cosmo": "TOT_PREC",
    "tcwv": "TQV",
    "w_snow_cosmo": "W_SNOW",
    "2q": "2m_specific_humidity",
    "pa": "air_pressure",
    "dt": "dew_point_temperature",
    "wd": "wind_from_direction",
    "ws": "wind_speed",
    "ts": "time_shift_minutes"
}


def get_every_nth_index(indices, n, shift_n_every_m=None):
    """
    Sample every nth index with optional shifting to avoid regular temporal patterns.
    
    Args:
        indices: List or array of indices/timestamps to sample from
        n: Take every nth element (e.g., n=100 for ~1% sampling)
        shift_n_every_m: Shift the starting index by 1 every m samples to avoid
                        always sampling the same time of day. None = no shifting.
    
    Returns:
        List of sampled indices
    
    Example:
        # Sample every 100th, shifting by 1 every 3 samples
        # Result: [0, 100, 200, 301, 401, 501, 602, ...]
        get_every_nth_index(range(1000), n=100, shift_n_every_m=3)
    """
    if n <= 0:
        return list(indices)
    
    sampled = []
    shift_offset = 0
    
    for i in range(0, len(indices), n):
        idx = i + shift_offset
        if idx < len(indices):
            sampled.append(indices[idx])
        
        # Apply shift every m samples
        if shift_n_every_m is not None and (len(sampled) % shift_n_every_m == 0):
            shift_offset += 1
    
    return sampled


class StationGTReader:
    """
    Memory-mapped reader for station ground truth data.
    Provides efficient access to GT data for given timestamps.
    """
    def __init__(self, data_path, meta_path):
        """
        Args:
            data_path: Path to .npy file with data array (T, V, H, W)
            meta_path: Path to .npz file with metadata (timestamps, variables, etc.)
        """
        # Load data with memory mapping (efficient - doesn't load full array to RAM)
        print(f'Loading data from {data_path} with memory mapping...')
        self.data = np.load(data_path, mmap_mode='r')
        print(f'Data shape: {self.data.shape} (T={self.data.shape[0]} timestamps, '
              f'V={self.data.shape[1]} vars, H={self.data.shape[2]}, W={self.data.shape[3]})')
        
        # Load metadata (small, fits in RAM)
        print(f'Loading metadata from {meta_path}...')
        meta = np.load(meta_path, allow_pickle=True)
        
        # Extract and normalize timestamps
        times_raw = meta["timestamps"]
        self.timestamps = np.array([np.datetime64(t, 'h') for t in times_raw])
        
        # Extract variables
        vars_raw = meta["variables"]
        self.variables = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
        
        # Build time -> index lookup dict for O(1) access
        self.time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(self.timestamps)}
        
        # Store other metadata
        self.lon_grid = meta.get("lon_grid", None)
        self.lat_grid = meta.get("lat_grid", None)
        self.locations = meta.get("locations", None)
        self.scales = meta.get("scales", None)
        
        print(f'Loaded {len(self.timestamps)} timestamps and {len(self.variables)} variables')
        print(f'Time range: {self.timestamps[0]} to {self.timestamps[-1]}')
        print(f'Variables: {self.variables}')
    
    def get_data_at_time(self, timestamp, variables=None):
        """
        Get ground truth data for a specific timestamp.
        
        Args:
            timestamp: datetime64, string, or datetime object
            variables: list of variable names to retrieve (None = all variables)
        
        Returns:
            dict with 'time', 'data', 'variables', and metadata
            data shape: (V, H, W) where V is number of requested variables
        """
        # Normalize timestamp to datetime64[h]
        t = np.datetime64(timestamp, 'h')
        t_key = int(t.astype('int64'))
        
        if t_key not in self.time_to_idx:
            raise KeyError(f"Timestamp {t} not found in dataset. Available range: "
                         f"{self.timestamps[0]} to {self.timestamps[-1]}")
        
        idx = self.time_to_idx[t_key]
        
        # Get variable indices
        if variables is None:
            var_indices = list(range(len(self.variables)))
            var_names = self.variables
        else:
            var_indices = [self.variables.index(v) for v in variables]
            var_names = variables
        
        # Extract data (memory-mapped, only reads what's needed)
        data = self.data[idx, var_indices, :, :]
        
        return {
            'time': t,
            'data': data,
            'variables': var_names,
        }
    
    def __repr__(self):
        return (f"StationGTReader(timestamps={len(self.timestamps)}, "
                f"variables={len(self.variables)}, "
                f"shape={self.data.shape})")


class StationGTReaderW5K:
    """
    Ground truth reader for W5K station data stored in NPZ format.
    
    This reader handles W5K station data where all information is stored
    in a single NPZ file, with separate CSV metadata.
    """
    def __init__(self, data_path, meta_path):
        """
        Initialize reader with paths to data and metadata files.
        
        Args:
            data_path: path to .npz file containing station data
            meta_path: path to .csv file containing station metadata (not used in computation)
        """
        print(f'Loading W5K data from {data_path}...')
        
        # Load NPZ file with memory mapping (efficient - doesn't load full arrays to RAM)
        npz_data = np.load(data_path, allow_pickle=True, mmap_mode='r')
        
        # Extract data arrays (memory-mapped)
        self.data = npz_data['data']  # Shape: (T, V, H, W) - memory mapped
        print(f'Data shape: {self.data.shape} (T={self.data.shape[0]} timestamps, '
              f'V={self.data.shape[1]} vars, H={self.data.shape[2]}, W={self.data.shape[3]})')
        
        # Extract and normalize timestamps
        times_raw = npz_data['timestamps']
        self.timestamps = np.array([np.datetime64(t, 'h') for t in times_raw])
        
        # Extract variables
        vars_raw = npz_data['variables']
        self.variables = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
        
        # Build time -> index lookup dict for O(1) access
        self.time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(self.timestamps)}
        
        # Store grid and mapping information
        self.lon_grid = npz_data.get('lon_grid', None)
        self.lat_grid = npz_data.get('lat_grid', None)
        self.mapping = npz_data.get('mapping', None)
        self.locations = npz_data.get('locations', None)
        self.scales = npz_data.get('scales', None)
        
        print(f'Loaded {len(self.timestamps)} timestamps and {len(self.variables)} variables')
        print(f'Time range: {self.timestamps[0]} to {self.timestamps[-1]}')
        print(f'Variables: {self.variables}')
    
    def get_data_at_time(self, timestamp, variables=None, warn_missing=True):
        """
        Get ground truth data for a specific timestamp.
        
        Args:
            timestamp: datetime64, string, or datetime object
            variables: list of variable names to retrieve (None = all variables)
            warn_missing: if True, print a warning for missing timestamps instead of raising error
        
        Returns:
            dict with 'time', 'data', 'variables', and metadata, or None if timestamp not found
            data shape: (V, H, W) where V is number of requested variables
        """
        # Normalize timestamp to datetime64[h]
        t = np.datetime64(timestamp, 'h')
        t_key = int(t.astype('int64'))
        
        if t_key not in self.time_to_idx:
            if warn_missing:
                return None
            else:
                raise KeyError(f"Timestamp {t} not found in dataset. Available range: {self.timestamps[0]} to {self.timestamps[-1]}")
        
        idx = self.time_to_idx[t_key]
        
        # Get variable indices
        if variables is None:
            var_indices = list(range(len(self.variables)))
            var_names = self.variables
        else:
            var_indices = [self.variables.index(v) for v in variables]
            var_names = variables
        
        # Extract data
        data = self.data[idx, var_indices, :, :]
        
        return {
            'time': t,
            'data': data,
            'variables': var_names,
        }
    
    def __repr__(self):
        return (f"StationGTReaderW5K(timestamps={len(self.timestamps)}, "
                f"variables={len(self.variables)}, "
                f"shape={self.data.shape})")


def compute_wind_speed_and_direction(u, v):
    """
    Compute wind speed and direction from U (zonal) and V (meridional) components.
    
    Args:
        u: U component (zonal wind, positive = eastward)
        v: V component (meridional wind, positive = northward)
    
    Returns:
        speed: Wind speed (magnitude)
        direction: Wind direction in degrees (0-360, meteorological convention:
                   0° = wind from north, 90° = wind from east, etc.)
    """
    speed = np.sqrt(u**2 + v**2)
    # Meteorological convention: direction wind is coming FROM
    # atan2(v, u) gives mathematical angle, we need to convert to meteorological
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360
    return speed, direction


def compute_metrics_for_init_time(it, lt, var_abr, var_a2f, p_preds, station_type, p_station, ensemble, delta_time, pred_data_preloaded=None):
    """
    Compute metrics for a single init_time at a given lead_time.
    Each worker opens its own file handles to avoid serialization issues.
    Returns: (it, metrics_dict, skip_reason) where:
        - metrics_dict[var] = {metric: value} if successful
        - metrics_dict = None and skip_reason = string describing why if skipped
    
    Args:
        station_type: 'w5k' or '11k' - determines data file structure
        delta_time: The first lead_time value (time step), used to check GT at init_time - delta_time
        pred_data_preloaded: Optional dict with pre-loaded prediction data {init_time: xarray.Dataset}
    """
    tt = it + lt  # target time
    
    # Get prediction data - either from pre-loaded dict or open zarr file
    try:
        if pred_data_preloaded is not None:
            # Use pre-loaded data (much faster for w5k)
            pred = pred_data_preloaded[it]
        else:
            # Open zarr file for each task (slower, but needed for 11k to avoid OOM)
            ds_preds = xr.open_zarr(p_preds, consolidated=False)
            if hasattr(ds_preds, 'ensemble'):
                pred = ds_preds.sel(init_time=it, lead_time=lt, ensemble=ensemble)
            else:
                pred = ds_preds.sel(init_time=it, lead_time=lt)
            ds_preds.close()
    except (KeyError, ValueError) as e:
        return (it, None, 'prediction_not_found')  # Skip if prediction not found
    
    # Open GT file (each worker gets its own handle)
    try:
        if station_type == '11k':
            p_data = os.path.join(p_station, 'stations_11k_training_data.npy')
            p_meta = os.path.join(p_station, 'stations_11k_training_meta.npz')
            
            # Load with memory mapping (efficient)
            data_mmap = np.load(p_data, mmap_mode='r')
            meta = np.load(p_meta, allow_pickle=True)
            
            # Extract timestamp info
            times_raw = meta["timestamps"]
            timestamps = np.array([np.datetime64(t, 'h') for t in times_raw])
            time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(timestamps)}
            
            # Get variable indices
            vars_raw = meta["variables"]
            variables = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
            var_indices = [variables.index(v) for v in var_abr]
        
        elif station_type == '11k_holdout':    
            p_data = os.path.join(p_station, 'stations_11k_holdout_data.npy')
            p_meta = os.path.join(p_station, 'stations_11k_holdout_meta.npz')
            # p_mapping = os.path.join(p_station, 'final_holdout_mapping_90x180.json')
            
            # Load with memory mapping (efficient)
            data_mmap = np.load(p_data, mmap_mode='r')
            meta = np.load(p_meta, allow_pickle=True)
            
            # Extract timestamp info
            times_raw = meta["timestamps"]
            timestamps = np.array([np.datetime64(t, 'h') for t in times_raw])
            time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(timestamps)}
            
            # Get variable indices
            vars_raw = meta["variables"]
            variables = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
            var_indices = [variables.index(v) for v in var_abr]
        
        elif station_type == 'w5k':
            p_data = os.path.join(p_station, 'stations_data.npz')
            
            # Load NPZ file with memory mapping (efficient - doesn't load full arrays to RAM)
            npz_data = np.load(p_data, allow_pickle=True, mmap_mode='r')
            data_mmap = npz_data['data']  # Memory-mapped array
            
            # Extract timestamp info (small arrays, load into memory)
            times_raw = npz_data['timestamps']
            timestamps = np.array([np.datetime64(t, 'h') for t in times_raw])
            time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(timestamps)}
            
            # Get variable indices
            vars_raw = npz_data['variables']
            variables = [v.decode() if isinstance(v, bytes) else v for v in vars_raw]
            var_indices = [variables.index(v) for v in var_abr]
        else:
            raise ValueError(f"Unknown station_type: {station_type}")
        
        # Get GT data for target time (tt)
        t = np.datetime64(tt, 'h')
        t_key = int(t.astype('int64'))
        
        if t_key not in time_to_idx:
            return (it, None, f'gt_target_time_missing:{t}')
        
        idx_tt = time_to_idx[t_key]
        
        # Get GT data for initialization time (it)
        t_init = np.datetime64(it, 'h')
        t_init_key = int(t_init.astype('int64'))
        
        if t_init_key not in time_to_idx:
            return (it, None, f'gt_init_time_missing:{t_init}')
        
        idx_it = time_to_idx[t_init_key]
        
        # Get GT data for initialization time - delta_time (previous time step)
        t_prev = np.datetime64(it - delta_time, 'h')
        t_prev_key = int(t_prev.astype('int64'))
        
        if t_prev_key not in time_to_idx:
            return (it, None, f'gt_prev_time_missing:{t_prev}')  # Skip if previous time GT not available
        
        idx_prev = time_to_idx[t_prev_key]
        
        # Extract GT data for all three times (use views first, copy only when needed)
        gt_tt_view = data_mmap[idx_tt, var_indices, :, :]
        gt_init = data_mmap[idx_it, var_indices, :, :]
        gt_prev = data_mmap[idx_prev, var_indices, :, :]
        
        # Create combined mask first, then copy only the slices we need
        # This reduces memory by not copying until after masking
        combined_masks = []
        for v_idx in range(len(var_abr)):
            nan_mask_init = np.isnan(gt_init[v_idx])
            nan_mask_prev = np.isnan(gt_prev[v_idx])
            combined_masks.append(nan_mask_init | nan_mask_prev)
        
        # Now copy only once with masks applied
        gt = gt_tt_view.copy()
        for v_idx in range(len(var_abr)):
            gt[v_idx][combined_masks[v_idx]] = np.nan
        
    except (KeyError, ValueError, IndexError) as e:
        return (it, None, f'gt_loading_error:{str(e)}')  # Skip if GT not found
    
    # Extract prediction data for each variable
    pred_array = np.stack([pred[var_a2f[k]].values for k in var_abr], axis=0)  # shape (V, H, W)
    
    # For w5k dataset, also compute wind speed and direction from U/V components
    var_abr_extended = list(var_abr)
    if station_type == 'w5k':
        # Check if both U and V components are present
        if '10u' in var_abr and '10v' in var_abr:
            u_idx = var_abr.index('10u')
            v_idx = var_abr.index('10v')
            
            # Compute wind speed and direction for predictions
            pred_u = pred_array[u_idx]
            pred_v = pred_array[v_idx]
            pred_ws, pred_wd = compute_wind_speed_and_direction(pred_u, pred_v)
            
            # Compute wind speed and direction for ground truth
            gt_u = gt[u_idx]
            gt_v = gt[v_idx]
            gt_ws, gt_wd = compute_wind_speed_and_direction(gt_u, gt_v)
            
            # Append to arrays
            pred_array = np.concatenate([pred_array, pred_ws[np.newaxis, :, :], pred_wd[np.newaxis, :, :]], axis=0)
            gt = np.concatenate([gt, gt_ws[np.newaxis, :, :], gt_wd[np.newaxis, :, :]], axis=0)
            
            # Add synthetic variable names
            var_abr_extended.extend(['ws', 'wd'])
    
    # Compute metrics for each variable
    metrics_dict = {}
    for v_idx, var in enumerate(var_abr_extended):
        pred_var = pred_array[v_idx]  # (H, W)
        gt_var = gt[v_idx]  # (H, W)
        
        # Mask for valid GT values (non-NaN)
        valid_mask = ~np.isnan(gt_var)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            continue
        
        # Extract valid values
        pred_valid = pred_var[valid_mask]
        gt_valid = gt_var[valid_mask]
        
        # Compute metrics
        # Special handling for wind direction (circular variable)
        if var == 'wd':
            # Circular error: take the minimum of clockwise and counter-clockwise differences
            error_raw = pred_valid - gt_valid
            error = np.where(np.abs(error_raw) > 180, 
                           error_raw - np.sign(error_raw) * 360, 
                           error_raw)
            abs_error = np.abs(error)
            sq_error = error ** 2
        else:
            error = pred_valid - gt_valid
            abs_error = np.abs(error)
            sq_error = error ** 2
        
        # MAPE calculation
        gt_threshold = 0.1 * np.abs(gt_valid).mean()
        mape_mask = np.abs(gt_valid) > gt_threshold
        if np.sum(mape_mask) > 0:
            mape_value = np.mean(abs_error[mape_mask] / np.abs(gt_valid[mape_mask])) * 100
        else:
            mape_value = np.nan
        
        # Pearson correlation
        if n_valid >= 3:
            corr_matrix = np.corrcoef(pred_valid, gt_valid)
            pearson_r = corr_matrix[0, 1]
        else:
            pearson_r = np.nan
        
        # Percentile-clipped MAE (to handle outliers)
        percentiles = [90, 95, 98, 99]
        mae_percentile = {}
        for pct in percentiles:
            pct_value = np.percentile(abs_error, pct)
            clipped_mask = abs_error <= pct_value
            if np.sum(clipped_mask) > 0:
                mae_percentile[f'mae_p{pct}'] = np.mean(abs_error[clipped_mask])
            else:
                mae_percentile[f'mae_p{pct}'] = np.nan
        
        # Normalized MAE and normalized MAE percentiles
        gt_mean = np.abs(gt_valid).mean()
        nmae = np.mean(abs_error) / (gt_mean + 1e-10)
        nmae_percentile = {}
        for pct in percentiles:
            pct_value = np.percentile(abs_error, pct)
            clipped_mask = abs_error <= pct_value
            if np.sum(clipped_mask) > 0:
                nmae_percentile[f'nmae_p{pct}'] = np.mean(abs_error[clipped_mask]) / (gt_mean + 1e-10)
            else:
                nmae_percentile[f'nmae_p{pct}'] = np.nan
        
        metrics_dict[var] = {
            'mae': np.mean(abs_error),
            'nmae': nmae,
            'rmse': np.sqrt(np.mean(sq_error)),
            'bias': np.mean(error),
            'std_error': np.std(error),
            'max_abs_error': np.max(abs_error),
            'n_valid_points': n_valid,
            'mape': mape_value,
            'nrmse': np.sqrt(np.mean(sq_error)) / (np.abs(gt_valid).mean() + 1e-10) * 100,
            'pearson_r': pearson_r,
            **mae_percentile,
            **nmae_percentile,
        }
    
    return (it, metrics_dict, None)


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics for station predictions')
    parser.add_argument('--p_preds', type=str, required=True,
                       help='Path to predictions zarr file')
    parser.add_argument('--station_type', type=str, required=True, choices=['11k', 'w5k', '11k_holdout'],
                       help='Station dataset type: 11k, w5k, or 11k_holdout')
    parser.add_argument('--p_station', type=str, 
                       default='/capstor/store/cscs/swissai/a122/ESFM_ExtraDataset',
                       help='Path to station dataset directory')
    parser.add_argument('--max_cores', type=int, default=96,
                       help='Maximum number of CPU cores to use')
    parser.add_argument('--ensemble', type=int, default=0,
                       help='Ensemble member to evaluate (default: 0)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--subsample_init_times', type=int, default=None,
                       help='Sample every Nth init_time to reduce computation (e.g., 100 for ~1%% sampling). None = use all.')
    parser.add_argument('--subsample_shift_every', type=int, default=None,
                       help='When subsampling, shift index by 1 every M samples to avoid regular temporal patterns. None = no shifting.')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STATION PREDICTION EVALUATION")
    print("=" * 80)
    print(f"Station type: {args.station_type}")
    print(f"Predictions path: {args.p_preds}")
    print(f"Station data path: {args.p_station}")
    print(f"Max CPU cores: {args.max_cores}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Setup checkpoint directory for incremental saving
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Scan for existing checkpoints to enable resume capability
    existing_checkpoints = {}
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_lt_') and f.endswith('.pkl')])
    if checkpoint_files:
        print(f"\n{'='*80}")
        print(f"FOUND {len(checkpoint_files)} EXISTING CHECKPOINTS - RESUME MODE ENABLED")
        print(f"{'='*80}")
        for ckpt_file in checkpoint_files:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
            try:
                with open(ckpt_path, 'rb') as f:
                    ckpt_data = pickle.load(f)
                    lt_hours = ckpt_data['lead_time_hours']
                    existing_checkpoints[lt_hours] = ckpt_path
                    print(f"  ✓ Found checkpoint for lead_time {lt_hours:.0f}h")
            except Exception as e:
                print(f"  ⚠️  Failed to load checkpoint {ckpt_file}: {e}")
        print(f"Will skip {len(existing_checkpoints)} already-completed lead_times\n")
    else:
        print(f"\nNo existing checkpoints found - starting fresh evaluation\n")
    
    # Initialize station GT reader based on station type
    if args.station_type == '11k':
        p_data = os.path.join(args.p_station, 'stations_11k_training_data.npy')
        p_meta = os.path.join(args.p_station, 'stations_11k_training_meta.npz')
        gt_reader = StationGTReader(p_data, p_meta)
    elif args.station_type == '11k_holdout':    
        p_data = os.path.join(args.p_station, 'stations_11k_holdout_data.npy')
        p_meta = os.path.join(args.p_station, 'stations_11k_holdout_meta.npz')
        gt_reader = StationGTReader(p_data, p_meta)
    elif args.station_type == 'w5k':
        p_data = os.path.join(args.p_station, 'stations_data.npz')
        p_meta = os.path.join(args.p_station, 'meta.csv')
        gt_reader = StationGTReaderW5K(p_data, p_meta)
    else:  
        raise ValueError(f"Unknown station_type: {args.station_type}")
    
    var_abr = gt_reader.variables
    ## drop ts if it's in var_abr:
    if 'ts' in var_abr:
        var_abr.remove('ts')
    var_full = [d_srf_abr2full[a] for a in var_abr]
    var_a2f = {a: f for a, f in zip(var_abr, var_full)}
    
    print(f"\nVariables to evaluate: {var_abr}")
    
    # Load predictions
    print(f"\nLoading predictions from {args.p_preds}...")
    ds_preds = xr.open_zarr(args.p_preds, consolidated=False)
    init_times = ds_preds['init_time'].values
    lead_times = ds_preds['lead_time'].values
    
    # Apply subsampling if requested
    if args.subsample_init_times is not None and args.subsample_init_times > 1:
        original_count = len(init_times)
        init_times = get_every_nth_index(
            init_times, 
            n=args.subsample_init_times,
            shift_n_every_m=args.subsample_shift_every
        )
        print(f"\n  SUBSAMPLING ENABLED: Using {len(init_times)}/{original_count} init_times ")
        print(f"   (every {args.subsample_init_times}th sample")
        if args.subsample_shift_every:
            print(f"    with shift by 1 every {args.subsample_shift_every} samples)")
        else:
            print(f")")
        print(f"   This reduces evaluation to ~{len(init_times)/original_count*100:.1f}% of full dataset")
    
    print(f"Found {len(init_times)} init times and {len(lead_times)} lead times")
    print(f"Total (init_time × lead_time) pairs: {len(init_times) * len(lead_times)}")
    
    # Close the zarr file - each worker will open its own
    ds_preds.close()
    
    # Initialize storage for metrics and missing GT tracking
    ensemble = args.ensemble
    metrics_per_timestep = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    missing_gt_log = defaultdict(lambda: defaultdict(list))  # {reason: {lead_time: [init_times]}}
    
    # Determine number of CPU cores to use
    # For w5k, use fewer workers to avoid OOM (each worker loads zarr data)
    max_cores = args.max_cores
    if args.station_type == 'w5k':
        # Limit to 8-12 workers for w5k to reduce memory pressure
        n_jobs = min(12, max_cores, len(init_times))
        backend = 'threading'  # Threading shares memory better than loky
    else:
        n_jobs = min(max(1, multiprocessing.cpu_count() - 2), max_cores, len(init_times))
        backend = 'loky'
    
    print("=" * 80)
    print(f"PARALLEL COMPUTATION using {n_jobs} workers with '{backend}' backend")
    print(f"(out of {multiprocessing.cpu_count()} CPU cores available)")
    print("=" * 80)
    
    total_iterations = len(lead_times) * len(init_times)
    completed_iterations = 0
    start_time = time.time()
    
    # Get delta_time (first lead_time value) to check GT at init_time - delta_time
    delta_time = lead_times[0]
    print(f"\nDelta time (first lead_time): {delta_time / np.timedelta64(1, 'h'):.0f} hours")
    print(f"Will verify GT availability at both init_time and init_time - delta_time for each variable")
    
    for lt_idx, lt in enumerate(lead_times):
        lt_hours = lt / np.timedelta64(1, 'h')
        lt_start_time = time.time()
        
        # Check if this lead_time is already checkpointed
        if lt_hours in existing_checkpoints:
            print(f"\n[Lead Time {lt_idx+1}/{len(lead_times)}] ⏭️  SKIPPING {lt_hours:.0f}h (checkpoint exists)")
            # Load the checkpoint data into metrics_per_timestep
            with open(existing_checkpoints[lt_hours], 'rb') as f:
                ckpt_data = pickle.load(f)
                metrics_per_timestep[lt] = ckpt_data['metrics_for_this_leadtime']
                # Merge missing_gt_log
                for reason, lead_time_data in ckpt_data.get('missing_gt_log', {}).items():
                    for lt_key, times in lead_time_data.items():
                        missing_gt_log[reason][lt_key].extend(times)
            continue
        
        print(f"\n[Lead Time {lt_idx+1}/{len(lead_times)}] Processing {lt_hours:.0f} hours with {n_jobs} parallel workers...")
        
        # For w5k, pre-load all predictions for this lead_time to avoid repeated zarr openings
        pred_data_preloaded = None
        if args.station_type == 'w5k':
            print(f"  Pre-loading predictions for lead_time {lt_hours:.0f}h into memory...")
            ds_preds_lt = xr.open_zarr(args.p_preds, consolidated=False)
            
            # Load all data for this lead_time
            pred_data_preloaded = {}
            if hasattr(ds_preds_lt, 'ensemble'):
                for it in init_times:
                    try:
                        pred_data_preloaded[it] = ds_preds_lt.sel(init_time=it, lead_time=lt, ensemble=ensemble).load()
                    except (KeyError, ValueError):
                        pass  # Skip missing init times
            else:
                for it in init_times:
                    try:
                        pred_data_preloaded[it] = ds_preds_lt.sel(init_time=it, lead_time=lt).load()
                    except (KeyError, ValueError):
                        pass  # Skip missing init times
            
            ds_preds_lt.close()
            print(f"  Loaded {len(pred_data_preloaded)} predictions into memory")
        
        # Process all init_times for this lead_time
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(compute_metrics_for_init_time)(it, lt, var_abr, var_a2f, args.p_preds, args.station_type, args.p_station, ensemble, delta_time, pred_data_preloaded)
            for it in init_times
        )
        
        # Collect results
        n_skipped = 0
        for it, metrics_dict, skip_reason in results:
            if metrics_dict is not None:
                for var in metrics_dict:
                    metrics_per_timestep[lt][it][var] = metrics_dict[var]
            else:
                n_skipped += 1
                if skip_reason:
                    missing_gt_log[skip_reason][lt].append(it)
            completed_iterations += 1
        
        lt_elapsed = time.time() - lt_start_time
        
        # Save checkpoint immediately after processing this lead_time
        checkpoint_filename = f"checkpoint_lt_{lt_hours:.0f}h.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'lead_time': lt,
                    'lead_time_hours': lt_hours,
                    'metrics_for_this_leadtime': dict(metrics_per_timestep[lt]),  # Convert defaultdict to dict
                    'missing_gt_log': {k: dict(v) for k, v in missing_gt_log.items()},  # Current state
                    'n_skipped': n_skipped,
                    'n_processed': len(init_times) - n_skipped,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'elapsed_seconds': lt_elapsed,
                }, f)
            print(f"  💾 Checkpoint saved: {checkpoint_filename}")
        except Exception as e:
            print(f"  ⚠️  Failed to save checkpoint: {e}")
        
        print(f"  ✓ Completed lead_time {lt_hours:.0f}h in {lt_elapsed:.1f}s ({len(init_times)} samples, {n_skipped} skipped)")
        
        # Progress update with ETA
        elapsed = time.time() - start_time
        progress_pct = (completed_iterations / total_iterations) * 100
        
        # ETA based on lead_times completed (more accurate than per-iteration)
        lead_times_completed = lt_idx + 1
        avg_time_per_leadtime = elapsed / lead_times_completed
        remaining_leadtimes = len(lead_times) - lead_times_completed
        eta_seconds = avg_time_per_leadtime * remaining_leadtimes
        
        if eta_seconds > 3600:
            eta_str = f"{eta_seconds/3600:.1f} hours"
        elif eta_seconds > 60:
            eta_str = f"{eta_seconds/60:.1f} min"
        else:
            eta_str = f"{eta_seconds:.0f}s"
        
        print(f"    Overall Progress: {lead_times_completed}/{len(lead_times)} lead_times ({progress_pct:.1f}%) | "
              f"Elapsed: {elapsed/60:.1f} min | ETA: {eta_str}")
    
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✓ All metrics computed in {total_elapsed/60:.2f} minutes ({completed_iterations} iterations)")
    print("=" * 80)
    
    # Report missing GT timestamps
    if missing_gt_log:
        print("\n" + "!" * 80)
        print("WARNING: Missing Ground Truth Data")
        print("!" * 80)
        
        total_missing = sum(len(times) for reason_data in missing_gt_log.values() for times in reason_data.values())
        total_attempted = len(init_times) * len(lead_times)
        print(f"\nTotal timesteps skipped due to missing GT: {total_missing} out of {total_attempted} ({total_missing/total_attempted*100:.1f}%)")
        if args.subsample_init_times is not None:
            print(f"(Note: Evaluation used subsampling with every {args.subsample_init_times}th init_time)\n")
        else:
            print()
        
        for reason, lead_time_data in sorted(missing_gt_log.items()):
            # Parse reason to make it more readable
            if reason.startswith('gt_target_time_missing:'):
                reason_display = "GT missing for target time (init_time + lead_time)"
            elif reason.startswith('gt_init_time_missing:'):
                reason_display = "GT missing for initialization time"
            elif reason.startswith('gt_prev_time_missing:'):
                reason_display = "GT missing for previous time (init_time - delta_time)"
            elif reason == 'prediction_not_found':
                reason_display = "Prediction not found in zarr file"
            elif reason.startswith('gt_loading_error:'):
                reason_display = f"GT loading error: {reason.split(':', 1)[1]}"
            else:
                reason_display = reason
            
            total_for_reason = sum(len(times) for times in lead_time_data.values())
            print(f"\n{reason_display}:")
            print(f"  Total affected: {total_for_reason} timesteps")
            
            # Group by lead_time and show statistics
            for lt in sorted(lead_time_data.keys(), key=lambda x: x.astype('timedelta64[h]').astype(int)):
                lt_hours = lt / np.timedelta64(1, 'h')
                missing_times = lead_time_data[lt]
                print(f"  - Lead time {lt_hours:>6.0f}h: {len(missing_times):>4} missing init times")
                
                # Show first few missing timestamps as examples
                if len(missing_times) <= 5:
                    for t in missing_times:
                        print(f"      {t}")
                else:
                    print(f"      Examples: {missing_times[0]}, {missing_times[1]}, ... {missing_times[-1]}")
        
        print("\n" + "!" * 80)
        print("Metrics calculated on available data only. Review warnings above.")
        print("!" * 80 + "\n")
    
    # Aggregate metrics by lead_time
    print("\nAggregating metrics across init_times for each lead_time...")
    metrics_by_leadtime = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Collect all unique variables from metrics (including synthetic ws/wd for w5k)
    all_vars = set(var_abr)
    for lt in lead_times:
        for it in init_times:
            if it in metrics_per_timestep[lt]:
                all_vars.update(metrics_per_timestep[lt][it].keys())
    all_vars = sorted(list(all_vars))  # Sort for consistent ordering
    
    for lt in lead_times:
        for var in all_vars:
            metric_names = ['mae', 'nmae', 'rmse', 'bias', 'std_error', 'max_abs_error', 'mape', 'nrmse', 'pearson_r', 'n_valid_points', 'mae_p90', 'mae_p95', 'mae_p98', 'mae_p99', 'nmae_p90', 'nmae_p95', 'nmae_p98', 'nmae_p99']
            metric_values = {mn: [] for mn in metric_names}
            
            for it in init_times:
                if it in metrics_per_timestep[lt] and var in metrics_per_timestep[lt][it]:
                    for mn in metric_names:
                        val = metrics_per_timestep[lt][it][var][mn]
                        if (mn in ['mape', 'pearson_r']) and np.isnan(val):
                            continue
                        metric_values[mn].append(val)
            
            for mn in metric_names:
                if len(metric_values[mn]) > 0:
                    values = np.array(metric_values[mn])
                    metrics_by_leadtime[lt][var][mn] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'n_samples': len(values),
                    }
    
    # Compute overall metrics
    for lt in lead_times:
        all_mae = []
        all_rmse = []
        all_bias = []
        all_pearson = []
        
        for it in init_times:
            if it in metrics_per_timestep[lt]:
                for var in var_abr:
                    if var in metrics_per_timestep[lt][it]:
                        all_mae.append(metrics_per_timestep[lt][it][var]['mae'])
                        all_rmse.append(metrics_per_timestep[lt][it][var]['rmse'])
                        all_bias.append(metrics_per_timestep[lt][it][var]['bias'])
                        r_val = metrics_per_timestep[lt][it][var]['pearson_r']
                        if not np.isnan(r_val):
                            all_pearson.append(r_val)
        
        if len(all_mae) > 0:
            metrics_by_leadtime[lt]['_overall'] = {
                'mae': {'mean': np.mean(all_mae), 'std': np.std(all_mae)},
                'rmse': {'mean': np.mean(all_rmse), 'std': np.std(all_rmse)},
                'bias': {'mean': np.mean(all_bias), 'std': np.std(all_bias)},
            }
            if len(all_pearson) > 0:
                metrics_by_leadtime[lt]['_overall']['pearson_r'] = {
                    'mean': np.mean(all_pearson), 'std': np.std(all_pearson)
                }
    
    print(f"\n✓ Metrics aggregation complete!")
    
    # Create detailed DataFrame
    print("\nCreating detailed metrics table...")
    detailed_rows = []
    for lt in lead_times:
        lt_hours = lt / np.timedelta64(1, 'h')
        for var in all_vars:
            if var in metrics_by_leadtime[lt]:
                m = metrics_by_leadtime[lt][var]
                
                mape_str = f"{m['mape']['mean']:.2f}" if 'mape' in m and len(m['mape']) > 0 else "N/A"
                nrmse_str = f"{m['nrmse']['mean']:.2f}" if 'nrmse' in m and len(m['nrmse']) > 0 else "N/A"
                pearson_str = f"{m['pearson_r']['mean']:.4f}" if 'pearson_r' in m and len(m['pearson_r']) > 0 else "N/A"
                
                # Get percentile MAE values
                mae_p90_str = f"{m['mae_p90']['mean']:.4f}" if 'mae_p90' in m and len(m['mae_p90']) > 0 else "N/A"
                mae_p95_str = f"{m['mae_p95']['mean']:.4f}" if 'mae_p95' in m and len(m['mae_p95']) > 0 else "N/A"
                mae_p98_str = f"{m['mae_p98']['mean']:.4f}" if 'mae_p98' in m and len(m['mae_p98']) > 0 else "N/A"
                mae_p99_str = f"{m['mae_p99']['mean']:.4f}" if 'mae_p99' in m and len(m['mae_p99']) > 0 else "N/A"
                
                # Get normalized MAE values
                nmae_str = f"{m['nmae']['mean']:.4f}" if 'nmae' in m and len(m['nmae']) > 0 else "N/A"
                nmae_p90_str = f"{m['nmae_p90']['mean']:.4f}" if 'nmae_p90' in m and len(m['nmae_p90']) > 0 else "N/A"
                nmae_p95_str = f"{m['nmae_p95']['mean']:.4f}" if 'nmae_p95' in m and len(m['nmae_p95']) > 0 else "N/A"
                nmae_p98_str = f"{m['nmae_p98']['mean']:.4f}" if 'nmae_p98' in m and len(m['nmae_p98']) > 0 else "N/A"
                nmae_p99_str = f"{m['nmae_p99']['mean']:.4f}" if 'nmae_p99' in m and len(m['nmae_p99']) > 0 else "N/A"
                
                row = {
                    'Lead Time (h)': int(lt_hours),
                    'Variable': var,
                    'MAE': f"{m['mae']['mean']:.4f}",
                    'MAE_p90': mae_p90_str,
                    'MAE_p95': mae_p95_str,
                    'MAE_p98': mae_p98_str,
                    'MAE_p99': mae_p99_str,
                    'NMAE': nmae_str,
                    'NMAE_p90': nmae_p90_str,
                    'NMAE_p95': nmae_p95_str,
                    'NMAE_p98': nmae_p98_str,
                    'NMAE_p99': nmae_p99_str,
                    'RMSE': f"{m['rmse']['mean']:.4f}",
                    'Bias': f"{m['bias']['mean']:.4f}",
                    'Pearson R': pearson_str,
                    'MAPE (%)': mape_str,
                    'nRMSE (%)': nrmse_str,
                    'N Valid Points (avg)': f"{m['n_valid_points']['mean']:.0f}",
                }
                detailed_rows.append(row)
    
    df_detailed = pd.DataFrame(detailed_rows)
    
    # Create summary DataFrame
    summary_data = []
    for lt in lead_times:
        if '_overall' in metrics_by_leadtime[lt]:
            row = {
                'Lead Time': str(lt),
                'MAE (mean±std)': f"{metrics_by_leadtime[lt]['_overall']['mae']['mean']:.4f}±{metrics_by_leadtime[lt]['_overall']['mae']['std']:.4f}",
                'RMSE (mean±std)': f"{metrics_by_leadtime[lt]['_overall']['rmse']['mean']:.4f}±{metrics_by_leadtime[lt]['_overall']['rmse']['std']:.4f}",
                'Bias (mean±std)': f"{metrics_by_leadtime[lt]['_overall']['bias']['mean']:.4f}±{metrics_by_leadtime[lt]['_overall']['bias']['std']:.4f}",
            }
            if 'pearson_r' in metrics_by_leadtime[lt]['_overall']:
                row['Pearson R (mean±std)'] = f"{metrics_by_leadtime[lt]['_overall']['pearson_r']['mean']:.4f}±{metrics_by_leadtime[lt]['_overall']['pearson_r']['std']:.4f}"
            summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert defaultdicts to regular dicts for pickling (lambdas can't be pickled)
    def convert_to_dict(d):
        if isinstance(d, defaultdict):
            d = {k: convert_to_dict(v) for k, v in d.items()}
        return d
    
    metrics_per_timestep_dict = convert_to_dict(metrics_per_timestep)
    metrics_by_leadtime_dict = convert_to_dict(metrics_by_leadtime)
    
    # Save comprehensive pickle
    metrics_output_path = os.path.join(args.output_dir, 'station_evaluation_metrics_full.pkl')
    with open(metrics_output_path, 'wb') as f:
        pickle.dump({
            'metrics_per_timestep': metrics_per_timestep_dict,
            'metrics_by_leadtime': metrics_by_leadtime_dict,
            'p_preds': args.p_preds,
            'init_times': init_times,
            'lead_times': lead_times,
            'var_abr': all_vars,  # Use all_vars to include synthetic ws/wd
            'var_a2f': var_a2f,
            'ensemble': ensemble,
            'df_detailed': df_detailed,
            'df_summary': df_summary,
            'save_timestamp': datetime.datetime.now().isoformat(),
        }, f)
    print(f"✓ Saved comprehensive metrics to: {metrics_output_path}")
    
    # Save summary pickle
    metrics_summary_path = os.path.join(args.output_dir, 'station_evaluation_metrics_summary.pkl')
    with open(metrics_summary_path, 'wb') as f:
        pickle.dump({
            'metrics_by_leadtime': metrics_by_leadtime_dict,
            'lead_times': lead_times,
            'var_abr': all_vars,  # Use all_vars to include synthetic ws/wd
            'var_a2f': var_a2f,
            'p_preds': args.p_preds,
        }, f)
    print(f"✓ Saved summary metrics to: {metrics_summary_path}")
    
    # Save detailed CSV
    csv_path = os.path.join(args.output_dir, 'station_evaluation_metrics_detailed.csv')
    df_detailed.to_csv(csv_path, index=False)
    print(f"✓ Saved detailed metrics CSV to: {csv_path}")
    
    # Save summary CSV
    summary_csv_path = os.path.join(args.output_dir, 'station_evaluation_metrics_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved summary metrics CSV to: {summary_csv_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print(f"Total time: {total_elapsed/60:.2f} minutes")
    if args.subsample_init_times is not None:
        print(f"Subsampling: Evaluated {len(init_times)} init_times (every {args.subsample_init_times}th)")
    if len(existing_checkpoints) > 0:
        print(f"Resumed from {len(existing_checkpoints)} existing checkpoints")
    print(f"All checkpoints saved in: {checkpoint_dir}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)
    print("\n💡 TIP: If job crashes, simply rerun - it will resume from checkpoints!")
    print("    To start fresh, delete the checkpoints/ directory.")
    print("=" * 80)
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)


if __name__ == '__main__':
    main()