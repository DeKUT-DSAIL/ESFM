# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import os
import glob
import pickle
import logging
import subprocess
import sys
import numpy as np
import xarray as xr
import dask.array as da
import zarr
from collections import defaultdict
from packaging import version
print(os.getcwd())
from config import parse_args

from utils.dataset import d_srf_abr2full

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check xarray version for zarr_format parameter support
# zarr_format parameter was added in xarray 2023.1.0
XARRAY_VERSION = version.parse(xr.__version__)
SUPPORTS_ZARR_FORMAT = XARRAY_VERSION >= version.parse("2023.1.0")
logging.info(f"xarray version: {xr.__version__}, supports zarr_format parameter: {SUPPORTS_ZARR_FORMAT}")

# Check zarr version for compressor API compatibility
# zarr 3.0+ changed the compressor API
ZARR_VERSION = version.parse(zarr.__version__)
USE_NUMCODECS = ZARR_VERSION >= version.parse("3.0.0")
logging.info(f"zarr version: {zarr.__version__}, use numcodecs for compressor: {USE_NUMCODECS}")

if USE_NUMCODECS:
    try:
        import numcodecs
        logging.info(f"numcodecs version: {numcodecs.__version__}")
    except ImportError:
        logging.warning("numcodecs not found, falling back to zarr compressor (may cause issues with zarr 3+)")
        USE_NUMCODECS = False

def get_zlib_compressor(level=1):
    """Get appropriate Zlib compressor based on zarr version.
    
    Args:
        level: Compression level (1-9)
        
    Returns:
        Compressor object compatible with current zarr version
    """
    if USE_NUMCODECS:
        # For zarr 3.0+, use numcodecs
        import numcodecs
        return numcodecs.Zlib(level=level)
    else:
        # For zarr < 3.0, use zarr.Zlib
        return zarr.Zlib(level=level)


def _format_init_time_units(times: np.ndarray) -> str:
    """Return CF-compliant units string for init_time in hours."""
    base = np.min(times).astype('datetime64[h]')
    base_str = np.datetime_as_string(base, unit='h')
    if 'T' in base_str:
        base_str = base_str.replace('T', ' ') + ":00:00"
    return f"hours since {base_str}"


def estimate_memory_usage(pred_files, memory_buffer_gb=2.0):
    """
    Estimate total memory needed to load all prediction files.
    
    Args:
        pred_files: List of pickle file paths
        memory_buffer_gb: Additional memory buffer to reserve (GB)
    
    Returns:
        Tuple of (total_size_gb, available_memory_gb, will_cause_oom)
    """
    import psutil
    
    # Calculate total size of all pickle files
    total_size_bytes = sum(os.path.getsize(f) for f in pred_files)
    total_size_gb = total_size_bytes / (1024**3)
    
    # Get available system memory
    memory_info = psutil.virtual_memory()
    available_memory_gb = memory_info.available / (1024**3)
    
    # Estimate actual memory usage (pickle files typically expand 2-3x when loaded)
    # Add buffer for xarray/dask overhead
    estimated_usage_gb = total_size_gb * 2.5 + memory_buffer_gb
    
    # Will cause OOM if estimated usage > 85% of available memory
    will_cause_oom = estimated_usage_gb > (available_memory_gb * 0.85)
    
    logging.info(f"Memory estimation:")
    logging.info(f"  Pickle files total size: {total_size_gb:.2f} GB")
    logging.info(f"  Estimated in-memory usage: {estimated_usage_gb:.2f} GB")
    logging.info(f"  Available memory: {available_memory_gb:.2f} GB")
    logging.info(f"  Predicted OOM risk: {'YES - will use incremental saving' if will_cause_oom else 'NO - using batch processing'}")
    
    return total_size_gb, available_memory_gb, will_cause_oom


def merge_prediction_files_incremental(log_dir, ckpt_step=-1, tmp_prediction_save_dir=None, tmp_predictions_path="tmp_predictions"):
    """
    Merge prediction files incrementally (one at a time) to avoid OOM errors.
    This is slower but uses minimal memory by processing each file individually.
    """
    logging.info("Using INCREMENTAL mode: processing files one-by-one to minimize memory usage")
    
    # Setup paths (same as main function)
    if tmp_predictions_path != "tmp_predictions":
        suffix = tmp_predictions_path.replace("tmp_predictions", "")
        output_filename = f'predictions{suffix}_step{ckpt_step}.zarr'
    else:
        output_filename = f'predictions_step{ckpt_step}.zarr'
    
    if tmp_prediction_save_dir is None:
        tmp_dir = os.path.join(log_dir, tmp_predictions_path)
        save_path = os.path.join(log_dir, output_filename)
    else:
        p_logdir = os.path.join(log_dir, tmp_prediction_save_dir)
        os.makedirs(p_logdir, exist_ok=True)
        tmp_dir = os.path.join(log_dir, tmp_prediction_save_dir, tmp_predictions_path)
        save_path = os.path.join(log_dir, tmp_prediction_save_dir, output_filename)
    
    if not os.path.exists(tmp_dir):
        raise ValueError(f"Temporary directory {tmp_dir} does not exist")

    pred_files = sorted(glob.glob(os.path.join(tmp_dir, 'pred_*.pkl')))
    if not pred_files:
        raise ValueError(f"No prediction files found in {tmp_dir}")

    # Track what we've initialized
    zarr_initialized = False
    
    # Get coordinate info from first file
    with open(pred_files[0], 'rb') as f:
        first_data = pickle.load(f)
        lats = first_data['latitude']
        lons = first_data['longitude']
        levels = first_data['level']
        ensemble = first_data.get('ensemble', 0)
        lead_time = first_data.get('lead_time', ['6h'])
        
        if lats.ndim > 1:
            lats = lats[:, 0] if lats.shape[1] > 1 else lats[0, :]
        if lons.ndim > 1:
            lons = lons[0, :] if lons.shape[0] > 1 else lons[:, 0]
    
    surface_vars = d_srf_abr2full.values()
    
    # Process each file individually
    for file_idx, pred_file in enumerate(pred_files):
        logging.info(f"Processing file {file_idx + 1}/{len(pred_files)}: {os.path.basename(pred_file)}")
        
        with open(pred_file, 'rb') as f:
            batch_preds = pickle.load(f)
        
        # Get variable names
        var_names = {k for k in batch_preds.keys() 
                    if k not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']}
        
        # Prepare coordinates
        times = np.array(batch_preds['init_time'], dtype='datetime64[ns]')
        lats_arr = np.array(lats)
        lons_arr = np.array(lons)
        levels_arr = np.array(levels)
        num_ensemble = ensemble if ensemble > 0 else 1
        ensembles = np.arange(num_ensemble)
        
        l_lt = []
        for lt in lead_time:
            if isinstance(lt, str) and lt.endswith('h'):
                hours = int(lt[:-1])
                l_lt.append(np.timedelta64(hours * 60 * 60 * 1_000_000_000, 'ns'))
            elif isinstance(lt, (int, np.integer)):
                l_lt.append(np.timedelta64(int(lt) * 60 * 60 * 1_000_000_000, 'ns'))
        lead_times = np.array(l_lt)
        
        # Create coordinates dict
        if lats_arr.ndim == 2 and lons_arr.ndim == 2:
            coords_dict = {
                'init_time': times,
                'latitude': (['y', 'x'], lats_arr),
                'longitude': (['y', 'x'], lons_arr),
                'level': levels_arr,
                'ensemble': ensembles,
                'lead_time': lead_times,
            }
            spatial_dims = ['y', 'x']
        else:
            coords_dict = {
                'init_time': times,
                'latitude': lats_arr,
                'longitude': lons_arr,
                'level': levels_arr,
                'ensemble': ensembles,
                'lead_time': lead_times,
            }
            spatial_dims = ['latitude', 'longitude']
        
        # Process variables
        ds_dict = {}
        
        for var_name in var_names:
            if var_name == '<N/A>':
                continue
            
            data = batch_preds[var_name]
            
            # Handle shape and dimension naming (same logic as batch version)
            if any(v in var_name for v in surface_vars):
                if data.ndim == 5:
                    dims = ['init_time', 'lead_time', 'ensemble'] + spatial_dims
                    chunks = (1, 1, -1, -1, -1)
                elif data.ndim == 4:
                    if ensemble > 0:
                        data = np.expand_dims(data, axis=1)
                        dims = ['init_time', 'lead_time', 'ensemble'] + spatial_dims
                        chunks = (1, 1, -1, -1, -1)
                    else:
                        dims = ['init_time', 'lead_time'] + spatial_dims
                        chunks = (1, 1, -1, -1)
                elif data.ndim == 3:
                    data = np.expand_dims(data, axis=1)
                    dims = ['init_time', 'lead_time'] + spatial_dims
                    chunks = (1, 1, -1, -1)
                else:
                    raise ValueError(f"Unexpected surface variable shape for {var_name}: {data.shape}")
            else:
                if data.ndim == 6:
                    data = np.transpose(data, (0, 1, 3, 2, 4, 5))
                    dims = ['init_time', 'lead_time', 'level', 'ensemble'] + spatial_dims
                    chunks = (1, 1, 1, -1, -1, -1)
                elif data.ndim == 5:
                    if ensemble > 0:
                        data = np.expand_dims(data, axis=1)
                        data = np.transpose(data, (0, 1, 3, 2, 4, 5))
                        dims = ['init_time', 'lead_time', 'level', 'ensemble'] + spatial_dims
                        chunks = (1, 1, 1, -1, -1, -1)
                    else:
                        dims = ['init_time', 'lead_time', 'level'] + spatial_dims
                        chunks = (1, 1, 1, -1, -1)
                elif data.ndim == 4:
                    data = np.expand_dims(data, axis=1)
                    dims = ['init_time', 'lead_time', 'level'] + spatial_dims
                    chunks = (1, 1, 1, -1, -1)
                else:
                    raise ValueError(f"Unexpected pressure level variable shape for {var_name}: {data.shape}")
            
            dask_array = da.from_array(data, chunks=chunks)
            coords = {k: v for k, v in coords_dict.items() if k in dims}
            ds_dict[var_name] = xr.DataArray(data=dask_array, dims=dims, coords=coords)
        
        ds = xr.Dataset(ds_dict)
        
        # Create encoding
        encoding = {}
        for var_name in ds.data_vars:
            dask_chunks = ds[var_name].chunks
            zarr_chunks = tuple(c[0] for c in dask_chunks) if dask_chunks else None
            encoding[var_name] = {
                'compressor': get_zlib_compressor(level=1),
                'chunks': zarr_chunks
            }
        encoding['init_time'] = {
            'units': _format_init_time_units(times),
            'dtype': 'int64'
        }
        
        # Write to zarr
        if not zarr_initialized:
            logging.info(f"Creating new zarr file at {save_path}")
            try:
                ds.to_zarr(save_path, mode='w', encoding=encoding, zarr_format=2)
            except TypeError:
                logging.info("zarr_format parameter not supported, using default")
                ds.to_zarr(save_path, mode='w', encoding=encoding)
            zarr_initialized = True
        else:
            logging.info(f"Appending to zarr file")
            try:
                ds.to_zarr(save_path, append_dim='init_time', zarr_format=2)
            except TypeError:
                logging.info("zarr_format parameter not supported, using default")
                ds.to_zarr(save_path, append_dim='init_time')
        
        # Free memory explicitly
        del batch_preds, ds, ds_dict, data
    
    logging.info("Incremental merge completed successfully")
    logging.info(f"Output saved at: {save_path}")
    
    # Automatically convert to fully sharded format
    sharded_path = save_path.replace('.zarr', '_sharded.zarr')
    logging.info(f"Converting to fully sharded format at: {sharded_path}")
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        zarrv2tov3_script = os.path.join(project_root, 'utils', 'zarrv2tov3.py')
        
        if not os.path.exists(zarrv2tov3_script):
            logging.warning(f"zarrv2tov3.py script not found at {zarrv2tov3_script}")
            logging.info(f"Skipping conversion. You can manually run: python utils/zarrv2tov3.py {save_path} {sharded_path}")
        else:
            result = subprocess.run(
                [sys.executable, zarrv2tov3_script, save_path, sharded_path],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info("Conversion to sharded format completed successfully")
            if result.stdout:
                logging.info(f"Conversion output: {result.stdout}")
            return sharded_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Zarr conversion failed with error: {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        logging.info(f"Returning non-sharded zarr file: {save_path}")
    except Exception as e:
        logging.error(f"Unexpected error during zarr conversion: {e}")
        logging.info(f"Returning non-sharded zarr file: {save_path}")
    
    return save_path


def merge_prediction_files(log_dir, ckpt_step=-1, tmp_prediction_save_dir=None, tmp_predictions_path="tmp_predictions"):
    """
    This function is used to process pkl files produced by inference.py.
    Merge prediction files in batches of 10, creating a zarr file with the first batch
    and appending subsequent batches.
    
    Args:
        log_dir (str): Path to the log directory containing the tmp_predictions folder
    """
    # Derive output filename suffix from tmp_predictions_path to differentiate output types
    # E.g., "tmp_predictions_rollout" -> "predictions_rollout_step{ckpt_step}.zarr"
    if tmp_predictions_path != "tmp_predictions":
        # Extract suffix from tmp path (e.g., "_rollout" from "tmp_predictions_rollout")
        suffix = tmp_predictions_path.replace("tmp_predictions", "")
        output_filename = f'predictions{suffix}_step{ckpt_step}.zarr'
    else:
        output_filename = f'predictions_step{ckpt_step}.zarr'
    
    if tmp_prediction_save_dir is None:
        tmp_dir = os.path.join(log_dir, tmp_predictions_path)
        save_path = os.path.join(log_dir, output_filename)
    else:
        p_logdir = os.path.join(log_dir, tmp_prediction_save_dir)
        os.makedirs(p_logdir, exist_ok=True)
        tmp_dir = os.path.join(log_dir, tmp_prediction_save_dir, tmp_predictions_path)
        save_path = os.path.join(log_dir, tmp_prediction_save_dir, output_filename)
    if not os.path.exists(tmp_dir):
        raise ValueError(f"Temporary directory {tmp_dir} does not exist")

    # Get all prediction files
    pred_files = sorted(glob.glob(os.path.join(tmp_dir, 'pred_*.pkl')))
    if not pred_files:
        raise ValueError(f"No prediction files found in {tmp_dir}")
    
    # Check if we should use incremental mode to avoid OOM
    try:
        total_size_gb, available_memory_gb, will_cause_oom = estimate_memory_usage(pred_files)
        if will_cause_oom:
            logging.warning("Estimated memory usage exceeds safe threshold. Switching to incremental mode.")
            return merge_prediction_files_incremental(log_dir, ckpt_step, tmp_prediction_save_dir, tmp_predictions_path)
    except Exception as e:
        logging.warning(f"Could not estimate memory usage: {e}. Proceeding with batch mode.")
    
    # Original batch processing code continues below

    # Process files in batches of 10
    batch_size = 10
    
    for batch_idx in range(0, len(pred_files), batch_size):
        batch_files = pred_files[batch_idx:batch_idx + batch_size]
        logging.info(f"Processing batch {batch_idx//batch_size + 1}, files {batch_idx+1} to {batch_idx+len(batch_files)}")

        # Initialize containers for this batch
        combined_data = defaultdict(list)
        times = []
        
        # Get coordinates from first file (they're the same for all files)
        with open(batch_files[0], 'rb') as f:
            first_batch = pickle.load(f)
            lats = first_batch['latitude']
            lons = first_batch['longitude']
            levels = first_batch['level']
            ensemble = first_batch.get('ensemble', 0)
            lead_time = first_batch.get('lead_time', ['6h'])
            
            # Ensure lats and lons are 1D arrays (in case they're 2D from meshgrid)
            if lats.ndim > 1:
                lats = lats[:, 0] if lats.shape[1] > 1 else lats[0, :]
            if lons.ndim > 1:
                lons = lons[0, :] if lons.shape[0] > 1 else lons[:, 0]

        # Load and combine predictions for this batch
        # First pass: collect all variable names from all files to detect inconsistencies
        all_var_names = set()
        file_var_names = {}
        
        for pred_file in batch_files:
            with open(pred_file, 'rb') as f:
                batch_preds = pickle.load(f)
            var_names = {k for k in batch_preds.keys() 
                        if k not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']}
            file_var_names[os.path.basename(pred_file)] = var_names
            all_var_names.update(var_names)
        
        logging.info(f"All variables found across batch: {sorted(all_var_names)}")
        
        # Second pass: load data and validate
        for pred_file in batch_files:
            logging.info(f"Processing {os.path.basename(pred_file)}")
            with open(pred_file, 'rb') as f:
                batch_preds = pickle.load(f)

            # Collect times only
            times.extend(batch_preds['init_time'])

            # Check for missing variables and log warning
            current_vars = {k for k in batch_preds.keys() 
                          if k not in ['init_time', 'latitude', 'longitude', 'level', 'ensemble', 'lead_time']}
            missing_vars = all_var_names - current_vars
            if missing_vars:
                logging.warning(f"File {os.path.basename(pred_file)} is missing variables: {missing_vars}")

            # Collect predictions
            for key in all_var_names:
                if key in batch_preds:
                    combined_data[key].append(batch_preds[key])
                else:
                    # Variable is missing from this file - this is the problem!
                    logging.error(f"Variable '{key}' missing from {os.path.basename(pred_file)}. Skipping this file's data.")
                    # Don't append anything - this will cause length mismatch which we'll catch below

        # Validate that all variables have the same number of samples
        expected_num_samples = len(batch_files)
        logging.info(f"Expected {expected_num_samples} samples per variable (one per file)")
        
        problematic_vars = {}
        for var_name, data_list in combined_data.items():
            actual_num_samples = len(data_list)
            if actual_num_samples != expected_num_samples:
                problematic_vars[var_name] = actual_num_samples
                logging.error(f"Variable '{var_name}' has {actual_num_samples} samples, expected {expected_num_samples}")
        
        if problematic_vars:
            logging.error(f"Batch {batch_idx//batch_size + 1} has inconsistent variable counts!")
            logging.error(f"Files in this batch: {[os.path.basename(f) for f in batch_files]}")
            logging.error(f"Variable counts: {problematic_vars}")
            logging.error("Skipping this batch due to data inconsistency.")
            continue  # Skip this batch and move to the next one
        
        # Convert to unique coordinates
        times = np.array(times, dtype='datetime64[ns]')
        lats = np.array(lats)
        lons = np.array(lons)
        levels = np.array(levels)
        
        # Log lat/lon shapes to understand the grid structure
        logging.info(f"Latitude shape: {lats.shape}, Longitude shape: {lons.shape}")
        logging.info(f"Latitude dtype: {lats.dtype}, Longitude dtype: {lons.dtype}")
        if lats.ndim == 2:
            logging.info(f"Detected 2D irregular grid (e.g., station data)")
            logging.info(f"Latitude range: [{lats.min():.4f}, {lats.max():.4f}]")
            logging.info(f"Longitude range: [{lons.min():.4f}, {lons.max():.4f}]")
        elif lats.ndim == 1:
            logging.info(f"Detected 1D regular grid")
            logging.info(f"Latitude: {len(lats)} points from {lats[0]:.4f} to {lats[-1]:.4f}")
            logging.info(f"Longitude: {len(lons)} points from {lons[0]:.4f} to {lons[-1]:.4f}")
        
        # Create ensemble coordinate array with correct number of members
        num_ensemble = ensemble if ensemble > 0 else 1
        ensembles = np.arange(num_ensemble)
        
        # Convert lead_time to timedelta64 (lead_time is now a list of hours like [6, 12, 18, ...])
        l_lt = []
        for lt in lead_time:
            if isinstance(lt, str) and lt.endswith('h'):
                # Handle string format like '6h', '12h', etc.
                hours = int(lt[:-1])
                l_lt.append(np.timedelta64(hours * 60 * 60 * 1_000_000_000, 'ns'))
            elif isinstance(lt, (int, np.integer)):
                # Handle integer format (hours)
                l_lt.append(np.timedelta64(int(lt) * 60 * 60 * 1_000_000_000, 'ns'))
            else:
                raise ValueError(f"Unexpected lead time format: {lt}")
        lead_times = np.array(l_lt)
        

        # Create xarray dataset for this batch
        ds_dict = {}
        
        # Handle both 1D regular grids and 2D irregular grids (e.g., station data)
        if lats.ndim == 2 and lons.ndim == 2:
            # For 2D irregular grids, we need to specify dimensions explicitly
            # The data variables will need to use these as 2D coordinates
            coords_dict = {
                'init_time': times,
                'latitude': (['y', 'x'], lats),  # 2D coordinate with explicit dims
                'longitude': (['y', 'x'], lons),  # 2D coordinate with explicit dims
                'level': levels,
                'ensemble': ensembles,
                'lead_time': lead_times,
            }
        else:
            # For 1D regular grids (standard case)
            coords_dict = {
                'init_time': times,
                'latitude': lats,
                'longitude': lons,
                'level': levels,
                'ensemble': ensembles,
                'lead_time': lead_times,
            }
        
        surface_vars = d_srf_abr2full.values()

        # Process each variable
        for var_name, data_list in combined_data.items():
            # Skip placeholder variables
            if var_name == '<N/A>':
                logging.info(f"Skipping placeholder variable: {var_name}")
                continue
            
            logging.info(f"Processing data for variable: {var_name}")
            data = np.concatenate(data_list, axis=0)
            
            # Log the data shape for debugging
            logging.info(f"Variable {var_name}: concatenated data shape = {data.shape}, expected times = {len(times)}")
            
            # Determine spatial dimension names based on grid type
            if lats.ndim == 2:
                # 2D irregular grid (e.g., station data)
                spatial_dims = ['y', 'x']
            else:
                # 1D regular grid
                spatial_dims = ['latitude', 'longitude']
            
            if any(var in var_name for var in surface_vars):
                # Surface variables can have different shapes:
                # WITH lead_time: [batch_size, lead_time, ensemble, lat, lon] (5D) or [batch_size, lead_time, lat, lon] (4D)
                # WITHOUT lead_time (non-rollout): [batch_size, ensemble, lat, lon] (4D) or [batch_size, lat, lon] (3D)
                if data.shape[0] != len(times):
                    logging.warning(f"Variable {var_name}: batch size {data.shape[0]} != expected times {len(times)}")
                
                # Check dimensions and handle missing lead_time
                if data.ndim == 5:
                    # Shape: [init_time, lead_time, ensemble, lat, lon]
                    dims = ['init_time', 'lead_time', 'ensemble'] + spatial_dims
                    chunks = (1, 1, -1, -1, -1)
                elif data.ndim == 4:
                    # Could be either [init_time, lead_time, lat, lon] OR [init_time, ensemble, lat, lon]
                    # Check if we have ensemble metadata to disambiguate
                    if ensemble > 0:
                        # This is [init_time, ensemble, lat, lon] - need to add lead_time dimension
                        logging.info(f"Variable {var_name}: detected non-rollout format [init_time, ensemble, lat, lon], expanding lead_time dimension")
                        data = np.expand_dims(data, axis=1)  # Insert lead_time at position 1
                        dims = ['init_time', 'lead_time', 'ensemble'] + spatial_dims
                        chunks = (1, 1, -1, -1, -1)
                    else:
                        # This is [init_time, lead_time, lat, lon]
                        dims = ['init_time', 'lead_time'] + spatial_dims
                        chunks = (1, 1, -1, -1)
                elif data.ndim == 3:
                    # Shape: [init_time, lat, lon] - no ensemble, no lead_time (non-rollout)
                    logging.info(f"Variable {var_name}: detected non-rollout format [init_time, lat, lon], expanding lead_time dimension")
                    data = np.expand_dims(data, axis=1)  # Insert lead_time at position 1
                    dims = ['init_time', 'lead_time'] + spatial_dims
                    chunks = (1, 1, -1, -1)
                else:
                    raise ValueError(f"Unexpected surface variable shape for {var_name}: {data.shape}")
            else:
                # Pressure level variables can have different shapes:
                # WITH lead_time: [bs, lead_time, ensemble, level, lat, lon] (6D) or [bs, lead_time, level, lat, lon] (5D)
                # WITHOUT lead_time (non-rollout): [bs, ensemble, level, lat, lon] (5D) or [bs, level, lat, lon] (4D)
                if data.shape[0] != len(times):
                    logging.warning(f"Variable {var_name}: batch size {data.shape[0]} != expected times {len(times)}")
                
                # Check if ensemble dimension is present
                if data.ndim == 6:
                    # Expected shape: [batch_size, lead_time, ensemble, level, lat, lon]
                    # Swap ensemble and level: [bs, lead_time, ens, plev, lat, lon] -> [bs, lead_time, plev, ens, lat, lon]
                    data = np.transpose(data, (0, 1, 3, 2, 4, 5))
                    dims = ['init_time', 'lead_time', 'level', 'ensemble'] + spatial_dims
                    chunks = (1, 1, 1, -1, -1, -1)
                elif data.ndim == 5:
                    # Could be either [bs, lead_time, level, lat, lon] OR [bs, ensemble, level, lat, lon]
                    # Check if we have ensemble metadata to disambiguate
                    if ensemble > 0:
                        # This is [bs, ensemble, level, lat, lon] - need to add lead_time dimension
                        logging.info(f"Variable {var_name}: detected non-rollout format [init_time, ensemble, level, lat, lon], expanding lead_time dimension")
                        data = np.expand_dims(data, axis=1)  # Insert lead_time at position 1
                        # Now shape is [bs, lead_time, ensemble, level, lat, lon]
                        # Swap ensemble and level: [bs, lead_time, ens, plev, lat, lon] -> [bs, lead_time, plev, ens, lat, lon]
                        data = np.transpose(data, (0, 1, 3, 2, 4, 5))
                        dims = ['init_time', 'lead_time', 'level', 'ensemble'] + spatial_dims
                        chunks = (1, 1, 1, -1, -1, -1)
                    else:
                        # This is [bs, lead_time, level, lat, lon]
                        dims = ['init_time', 'lead_time', 'level'] + spatial_dims
                        chunks = (1, 1, 1, -1, -1)
                elif data.ndim == 4:
                    # Shape: [bs, level, lat, lon] - no ensemble, no lead_time (non-rollout)
                    logging.info(f"Variable {var_name}: detected non-rollout format [init_time, level, lat, lon], expanding lead_time dimension")
                    data = np.expand_dims(data, axis=1)  # Insert lead_time at position 1
                    dims = ['init_time', 'lead_time', 'level'] + spatial_dims
                    chunks = (1, 1, 1, -1, -1)
                else:
                    raise ValueError(f"Unexpected pressure level variable shape for {var_name}: {data.shape}")
                
            dask_array = da.from_array(data, chunks=chunks)

            coords = {k:v for k,v in coords_dict.items() if k in dims}
            
            ds_dict[var_name] = xr.DataArray(
                data=dask_array,
                dims=dims,
                coords=coords
            )

        ds = xr.Dataset(ds_dict)
        
        # Create encoding to avoid 2GB buffer limit with Blosc compressor
        # Use zlib compression (level 1) which doesn't have the 2GB limit
        encoding = {}
        for var_name in ds.data_vars:
            # Convert dask chunks (tuple of tuples) to zarr chunks (tuple of ints)
            # ds[var_name].chunks returns ((1,1,1,...), (56,), ...) but we need (1, 56, ...)
            dask_chunks = ds[var_name].chunks
            if dask_chunks:
                zarr_chunks = tuple(c[0] for c in dask_chunks)
            else:
                zarr_chunks = None
                
            encoding[var_name] = {
                'compressor': get_zlib_compressor(level=1),  # zlib doesn't have 2GB limit like Blosc
                'chunks': zarr_chunks
            }
        encoding['init_time'] = {
            'units': _format_init_time_units(times),
            'dtype': 'int64'
        }
        
        # For first batch, create new zarr file
        if batch_idx == 0:
            logging.info(f"Creating new zarr file at {save_path}")
            logging.info(f"Using zlib compression (level 1) to avoid 2GB buffer limit")
            try:
                # Try with zarr_format parameter (xarray >= 2023.1.0)
                ds.to_zarr(save_path, mode='w', encoding=encoding, zarr_format=2)
            except TypeError:
                # Fall back to without zarr_format for older xarray versions
                logging.info("zarr_format parameter not supported, using default")
                ds.to_zarr(save_path, mode='w', encoding=encoding)
        # For subsequent batches, append to existing zarr file
        else:
            logging.info(f"Appending to existing zarr file at {save_path}")
            # When appending, don't provide encoding - the structure is already set by the initial write
            try:
                # Try with zarr_format parameter (xarray >= 2023.1.0)
                ds.to_zarr(save_path, append_dim='init_time', zarr_format=2)
            except TypeError:
                # Fall back to without zarr_format for older xarray versions
                logging.info("zarr_format parameter not supported, using default")
                ds.to_zarr(save_path, append_dim='init_time')
    print(ds)
    logging.info("Merge completed successfully")
    logging.info(f"Output saved at: {save_path}")
    
    # Automatically convert to fully sharded format (one file per variable)
    # Preserve the filename structure (e.g., predictions_rollout -> predictions_rollout_sharded)
    sharded_path = save_path.replace('.zarr', '_sharded.zarr')
    logging.info(f"Converting to fully sharded format at: {sharded_path}")
    
    try:
        # Get the project root directory (assumes merge_predictions.py is in scripts/inference/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        zarrv2tov3_script = os.path.join(project_root, 'utils', 'zarrv2tov3.py')
        
        if not os.path.exists(zarrv2tov3_script):
            logging.warning(f"zarrv2tov3.py script not found at {zarrv2tov3_script}")
            logging.info(f"Skipping conversion. You can manually run: python utils/zarrv2tov3.py {save_path} {sharded_path}")
        else:
            result = subprocess.run(
                [sys.executable, zarrv2tov3_script, save_path, sharded_path],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info("Conversion to sharded format completed successfully")
            if result.stdout:
                logging.info(f"Conversion output: {result.stdout}")
            return sharded_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Zarr conversion failed with error: {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        logging.info(f"Returning non-sharded zarr file: {save_path}")
    except Exception as e:
        logging.error(f"Unexpected error during zarr conversion: {e}")
        logging.info(f"Returning non-sharded zarr file: {save_path}")
    
    return save_path

if __name__ == "__main__":
    args = parse_args()
    merge_prediction_files(args.log_dir)