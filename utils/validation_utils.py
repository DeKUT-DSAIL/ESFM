# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import shutil
import os
import threading
from tqdm import tqdm
import traceback
from utils.merge_predictions import merge_prediction_files

def get_every_nth_index(inds, n=6, shift_n_every_m=3):
    """
    Returns a subset of indices from the input list, selecting every n-th index,
    with an optional shift applied every m steps.

    Parameters:
        inds (list or np.ndarray): List or array of indices to select from.
        n (int): Step size to select indices.
        shift_n_every_m (int): Number of steps after which the step size is increased by 1.

    Returns:
        list: List of selected indices.
    """

    # Select a subset of indices from val set (6h intervals, but shift it by 1h every day):
    selected_inds = []
    step = 0
    i = 0
    while i < len(inds):
        if step < shift_n_every_m:
            selected_inds.append(inds[i])
            i += n  # Take every n-th index
            step += 1
        else:
            selected_inds.append(inds[i])
            i += n + 1  # Take the (n+1)-th index
            step = 0  # Reset the step counter
    return selected_inds

def get_region_indices(latlon_grid, region_str="switzerland", tolerance=0.):
    """
    Returns indices from the input lat-lon grid that fall within the specified region's bounding box.

    Parameters:
        latlon_grid (np.ndarray): N x 2 array where each row is [lat, lon]

    Returns:
        np.ndarray: Indices of rows corresponding to pixels over the specified region
    """

    if region_str.lower() == "switzerland":
        # Switzerland bounding box (approximate)
        lat_min, lat_max = 45.8, 47.8
        lon_min, lon_max = 5.9, 10.5
    elif region_str.lower() == "europe":
        # Europe bounding box (approximate)
        lat_min, lat_max = 35.0, 72.0
        lon_min, lon_max = -25.0, 50.0
    elif region_str.lower() == "usa":
        # USA (contiguous) bounding box
        lat_min, lat_max = 24.5, 49.5
        lon_min, lon_max = 235.0, 293.5  # Converted from -125 to -66.5
    elif region_str.lower() == "typhoon_doksuri_2023":
        # Typhoon Doksuri 2023 bounding box (approximate)
        lat_min, lat_max = 10.0, 35.0
        lon_min, lon_max = 115.0, 135.0

    if tolerance > 0:
        lat_min -= tolerance
        lat_max += tolerance
        lon_min -= tolerance
        lon_max += tolerance

    # Filter based on bounding box
    lat = latlon_grid[:, 0]
    lon = latlon_grid[:, 1]

    # If longitudes are in [-180, 180], convert to [0, 360]
    if np.any(lon < 0):
        lon = (lon + 360) % 360

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    indices = np.where(mask)[0]

    return indices


def get_region_index_intervals(lat, lon, region_str="switzerland", tolerance=0.0):
    """
    Returns start-end indices for lat and lon axes separately that fall within the specified region's bounding box.

    Parameters:
        lat (np.ndarray): 1D array of latitude values
        lon (np.ndarray): 1D array of longitude values
        region_str (str): Region name ("switzerland", "europe", "usa")

    Returns:
        tuple: (lat_indices, lon_indices) - indices for lat and lon axes separately
    """
    
    if region_str.lower() == "switzerland":
        # Switzerland bounding box (approximate)
        lat_min, lat_max = 45.8, 47.8
        lon_min, lon_max = 5.9, 10.5
    elif region_str.lower() == "europe":
        # Europe bounding box (approximate)
        lat_min, lat_max = 35.0, 72.0
        lon_min, lon_max = -25.0, 50.0
    elif region_str.lower() == "usa":
        # USA (contiguous) bounding box
        lat_min, lat_max = 24.5, 49.5
        lon_min, lon_max = 235.0, 293.5  # Converted from -125 to -66.5
        
    if tolerance > 0:
        lat_min -= tolerance
        lat_max += tolerance
        lon_min -= tolerance
        lon_max += tolerance

    # Convert longitude to [0, 360] if it's in [-180, 180]
    lon_converted = lon.copy()
    if np.any(lon_converted < 0):
        lon_converted = (lon_converted + 360) % 360

    # Find indices for lat and lon separately
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon_converted >= lon_min) & (lon_converted <= lon_max)
    
    # get start and end indices for lat and lon
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    if lat_indices.size == 0 or lon_indices.size == 0:
        raise ValueError(f"No indices found for region '{region_str}' with the given tolerance.")
    lat_start, lat_end = lat_indices[0], lat_indices[-1]
    lon_start, lon_end = lon_indices[0], lon_indices[-1]

    return (lat_start, lat_end), (lon_start, lon_end)

def merge_prediction_files_and_copy_to_capstor(log_dir, ckpt_step, config, tmp_prediction_save_dir=None, execute_copy_to_capstor=False):
    # Store exception info if copy fails
    thread_exception = {'error': None}
    
    def merge_and_copy_to_capstor():
        try:
            save_path = merge_prediction_files(log_dir, ckpt_step=ckpt_step, tmp_prediction_save_dir=tmp_prediction_save_dir, tmp_predictions_path=tmp_pred_fname)
            if execute_copy_to_capstor:
                # save_path: os.path.join(log_dir, tmp_prediction_save_dir, f'predictions_step{ckpt_step}.zarr')
                logging.info("Merging of prediction files completed. Now copying them to capstor.")
                
                p_copy_results_parent = "/capstor/store/cscs/swissai/a122/ESFM_Results"
                exp_name = os.path.basename(log_dir) # esfm_small_aa_unmask_rplv
                if tmp_prediction_save_dir is not None:
                    sd_dn = os.path.basename(os.path.dirname(save_path)) # tmp_prediction_save_dir or esfm_small_aa_unmask_rplv
                    exp_name = os.path.join(exp_name, sd_dn) # e.g., esfm_small_aa_unmask_rplv/mask_var_10u_surf_var
                sd_bn = os.path.basename(save_path)
                td_results = os.path.join(p_copy_results_parent, exp_name, sd_bn) # e.g., /capstor/store/cscs/swissai/a122/ESFM_Results/esfm_small_aa_unmask_rplv/mask_var_10u_surf_var/predictions_step50000.zarr
                os.makedirs(td_results, exist_ok=True)
                
                # copy save_path directory to p_copy_results
                copytree_with_progress(save_path, td_results)
                logging.info(f"Copied {save_path} to {td_results}")

                # copy config yaml (args.config) to p_copy_results
                td= os.path.join(p_copy_results_parent, exp_name)
                shutil.copy(config, td)
                logging.info(f"Copied config yaml {config} to {td}")
        except Exception as e:
            error_msg = f"Error in merge/copy thread for {tmp_prediction_save_dir}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            thread_exception['error'] = e

    # Start the copy operation in a separate thread
    logging.info("Starting merge and (maybe) copying files to capstor in a background thread")
    copy_thread = threading.Thread(target=merge_and_copy_to_capstor, daemon=False)  # Non-daemon to ensure completion
    copy_thread.start()
    # Store exception info on thread object for later checking
    copy_thread.thread_exception = thread_exception
    
    return copy_thread

def copytree_with_progress(src, dst):
    files = []
    for root, _, filenames in os.walk(src):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    for file in tqdm(files, desc="Copying files", unit="file"):
        rel_path = os.path.relpath(file, src)
        dst_file = os.path.join(dst, rel_path)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(file, dst_file)