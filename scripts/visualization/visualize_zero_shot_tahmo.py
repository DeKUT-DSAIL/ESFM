# Do not add emojis to the code
# Copyright (c) 2026 DeKUT-DSAIL Research Lab
# Licensed under the MIT License. See LICENSE in repository root.

"""
Zero-Shot TAHMO Error Analysis & Spatial Diagnostics
Computes lead-time error curves, station rankings, hydrostatic elevation correlation,
and continental African spatial error heatmaps for ESFM zero-shot rollout evaluations.
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Mapping from abbreviation to descriptive names and standard units
VAR_INFO = {
    'pa': {'name': 'Surface Pressure', 'unit': 'Pa', 'scale': 1.0, 'disp_unit': 'hPa', 'disp_scale': 0.01},
    'msl': {'name': 'Mean Sea Level Pressure', 'unit': 'Pa', 'scale': 1.0, 'disp_unit': 'hPa', 'disp_scale': 0.01},
    '2t': {'name': '2m Temperature', 'unit': 'K', 'scale': 1.0, 'disp_unit': 'deg C', 'disp_scale': 1.0},
    'dt': {'name': 'Dew Point Temperature', 'unit': 'K', 'scale': 1.0, 'disp_unit': 'deg C', 'disp_scale': 1.0},
    '10u': {'name': '10m U Wind', 'unit': 'm/s', 'scale': 1.0, 'disp_unit': 'm/s', 'disp_scale': 1.0},
    '10v': {'name': '10m V Wind', 'unit': 'm/s', 'scale': 1.0, 'disp_unit': 'm/s', 'disp_scale': 1.0},
    'ws': {'name': 'Wind Speed', 'unit': 'm/s', 'scale': 1.0, 'disp_unit': 'm/s', 'disp_scale': 1.0},
    'wd': {'name': 'Wind Direction', 'unit': 'deg', 'scale': 1.0, 'disp_unit': 'deg', 'disp_scale': 1.0},
}

d_srf_abr2full = {
    '2t': '2m_temperature',
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'msl': 'mean_sea_level_pressure',
    'pa': 'air_pressure',
    'dt': 'dew_point_temperature',
    'wd': 'wind_from_direction',
    'ws': 'wind_speed',
    'ts': 'time_shift_minutes',
}


def plot_lead_time_error_curves(metrics_csv_path, output_dir):
    """
    Generate multi-panel lead-time error curves (MAE, clipped MAE p95, RMSE) up to 240h.
    """
    print(f"Reading metrics from {metrics_csv_path}...")
    df = pd.read_csv(metrics_csv_path)

    # Clean numeric columns
    numeric_cols = ['Lead Time (h)', 'MAE', 'MAE_p90', 'MAE_p95', 'MAE_p98', 'MAE_p99', 'RMSE', 'Bias']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    target_vars = ['pa', 'msl', '2t', 'dt', '10u', '10v', 'ws']
    available_vars = [v for v in target_vars if v in df['Variable'].unique()]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), dpi=300)
    axes = axes.flatten()

    for idx, var in enumerate(available_vars):
        ax = axes[idx]
        sub_df = df[df['Variable'] == var].sort_values('Lead Time (h)')

        lt = sub_df['Lead Time (h)']
        mae = sub_df['MAE']
        mae_p95 = sub_df['MAE_p95'] if 'MAE_p95' in sub_df.columns else None
        rmse = sub_df['RMSE'] if 'RMSE' in sub_df.columns else None

        info = VAR_INFO.get(var, {'name': var, 'disp_unit': '', 'disp_scale': 1.0})
        scale = info['disp_scale']
        disp_unit = info['disp_unit']

        ax.plot(lt, mae * scale, 'o-', color='#1f77b4', linewidth=2.0, markersize=4, label='MAE')
        if mae_p95 is not None and not mae_p95.isna().all():
            ax.plot(lt, mae_p95 * scale, 's--', color='#2ca02c', linewidth=1.5, markersize=3, label='MAE (p95 clipped)')
        if rmse is not None and not rmse.isna().all():
            ax.plot(lt, rmse * scale, '^-.', color='#d62728', linewidth=1.5, markersize=3, label='RMSE')

        ax.set_title(f"{info['name']} ({var})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Lead Time (hours)', fontsize=10)
        ax.set_ylabel(f"Error [{disp_unit}]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim(0, 246)

    # Use the 8th subplot for summary or hide if unused
    if len(available_vars) < len(axes):
        ax_last = axes[-1]
        overall_rows = df[df['Variable'].str.contains('overall', case=False, na=False)]
        if not overall_rows.empty and 'MAE' in overall_rows.columns:
            overall_rows = overall_rows.sort_values('Lead Time (h)')
            ax_last.plot(overall_rows['Lead Time (h)'], overall_rows['MAE'], 'o-', color='#333333', label='Mean Overall MAE')
            ax_last.set_title('Overall Multi-Variable MAE', fontsize=12, fontweight='bold')
            ax_last.set_xlabel('Lead Time (hours)', fontsize=10)
            ax_last.set_ylabel('Normalized / Composite MAE', fontsize=10)
            ax_last.grid(True, linestyle='--', alpha=0.6)
            ax_last.legend(loc='best', fontsize=8)
        else:
            ax_last.axis('off')

    plt.suptitle('ESFM Zero-Shot Rollout Forecast Error Trajectories on African TAHMO Stations (2024-2026)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = os.path.join(output_dir, 'lead_time_error_curves.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved lead-time error curves to {out_file}")


def compute_station_diagnostics(p_preds, p_station_dir, station_meta_json_path, output_dir, experiments_dir):
    """
    Extract station-level error metrics across holdout stations and rank worst performers.
    """
    print(f"Loading holdout ground-truth metadata from {p_station_dir}...")
    p_meta = os.path.join(p_station_dir, 'tahmo_holdout_meta.npz')
    p_data = os.path.join(p_station_dir, 'tahmo_holdout_data.npy')

    meta = np.load(p_meta, allow_pickle=True)
    mapping = meta['mapping']
    holdout_mask = meta['holdout_mask']
    gt_vars = [v.decode() if isinstance(v, bytes) else v for v in meta['variables']]
    timestamps_raw = meta['timestamps']
    gt_timestamps = np.array([np.datetime64(t, 'h') for t in timestamps_raw])
    time_to_idx = {int(t.astype('int64')): i for i, t in enumerate(gt_timestamps)}

    with open(station_meta_json_path, 'r') as f:
        station_metadata = json.load(f)

    # Filter holdout stations (150 stations)
    holdout_stations = []
    for row in mapping:
        code = row[0]
        h = int(row[2])
        w = int(row[3])
        lat = float(row[4])
        lon = float(row[5])
        if code == 'TA00933':
            # Correct swapped latitude/longitude in station metadata
            lat, lon = lon, lat
        if holdout_mask[h, w]:
            loc_data = station_metadata.get(code, {}).get('location', {})
            elev = loc_data.get('elevationmsl', None)
            name = loc_data.get('name', code)
            cc = loc_data.get('countrycode', '')
            holdout_stations.append({
                'code': code,
                'name': name,
                'h': h,
                'w': w,
                'lon': lon,
                'lat': lat,
                'elevationmsl': elev,
                'country': cc
            })

    print(f"Found {len(holdout_stations)} verified holdout stations.")

    print(f"Opening prediction store from {p_preds}...")
    ds_preds = xr.open_zarr(p_preds, consolidated=False)
    pred_init_times = ds_preds['init_time'].values
    lead_times = ds_preds['lead_time'].values

    # Memory map GT data
    data_mmap = np.load(p_data, mmap_mode='r')

    # Variables to analyze
    eval_vars = ['pa', 'msl', '2t', 'dt', '10u', '10v', 'ws']
    station_error_records = {v: [] for v in eval_vars}

    # Build target timestamp index matrix for sampled init times and lead times
    step_sample = max(1, len(pred_init_times) // 50)
    sampled_init_times = pred_init_times[::step_sample]
    print(f"Computing station errors across {len(sampled_init_times)} sampled init times and {len(lead_times)} lead times...")

    n_it = len(sampled_init_times)
    n_lt = len(lead_times)
    gt_indices = np.full((n_it, n_lt), -1, dtype=int)
    for i, it in enumerate(sampled_init_times):
        for j, lt in enumerate(lead_times):
            target_time = np.datetime64(it, 'h') + np.timedelta64(lt, 'h')
            t_key = int(target_time.astype('int64'))
            if t_key in time_to_idx:
                gt_indices[i, j] = time_to_idx[t_key]

    for v in eval_vars:
        if v not in gt_vars and v != 'ws':
            continue

        pred_var_name = d_srf_abr2full.get(v, v)
        if pred_var_name not in ds_preds and v != 'ws':
            continue

        print(f"Loading predictions for {v} ({pred_var_name})...")
        if v == 'ws' and pred_var_name in ds_preds:
            pred_arr = ds_preds[pred_var_name].sel(init_time=sampled_init_times).values
        elif v == 'ws':
            pred_u = ds_preds['10m_u_component_of_wind'].sel(init_time=sampled_init_times).values
            pred_v = ds_preds['10m_v_component_of_wind'].sel(init_time=sampled_init_times).values
            pred_arr = np.sqrt(pred_u**2 + pred_v**2)
        else:
            pred_arr = ds_preds[pred_var_name].sel(init_time=sampled_init_times).values

        if pred_arr.ndim == 5:
            pred_arr = pred_arr[:, :, 0, :, :]

        # Extract GT and compute station metrics
        for st in holdout_stations:
            h = st['h']
            w = st['w']

            # Extract ground truth
            if v == 'ws' and 'ws' in gt_vars:
                var_idx = gt_vars.index('ws')
                gt_vals = data_mmap[gt_indices, var_idx, h, w]
                valid = ~np.isnan(gt_vals) & (gt_indices >= 0)
            elif v == 'ws':
                if '10u' in gt_vars and '10v' in gt_vars:
                    u_idx = gt_vars.index('10u')
                    v_idx = gt_vars.index('10v')
                    gt_u = data_mmap[gt_indices, u_idx, h, w]
                    gt_v = data_mmap[gt_indices, v_idx, h, w]
                    valid = ~np.isnan(gt_u) & ~np.isnan(gt_v) & (gt_indices >= 0)
                    gt_vals = np.sqrt(gt_u**2 + gt_v**2)
                else:
                    continue
            else:
                var_idx = gt_vars.index(v)
                gt_vals = data_mmap[gt_indices, var_idx, h, w]
                valid = ~np.isnan(gt_vals) & (gt_indices >= 0)

            pred_vals = pred_arr[:, :, h, w]
            diff = pred_vals[valid] - gt_vals[valid]

            if len(diff) > 0:
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff**2)))
                bias = float(np.mean(diff))
                n_pts = int(len(diff))
            else:
                mae = np.nan
                rmse = np.nan
                bias = np.nan
                n_pts = 0

            station_error_records[v].append({
                'station_code': st['code'],
                'name': st['name'],
                'latitude': st['lat'],
                'longitude': st['lon'],
                'elevationmsl': st['elevationmsl'],
                'country': st['country'],
                'mae': mae,
                'rmse': rmse,
                'bias': bias,
                'n_observations': n_pts
            })

    ds_preds.close()

    # Create Rankings per variable
    rankings_output = {}
    all_stations_summary_rows = []

    for v in eval_vars:
        records = station_error_records[v]
        if not records:
            continue
        v_df = pd.DataFrame(records).dropna(subset=['mae'])
        v_df = v_df.sort_values('mae', ascending=False)

        top15 = v_df.head(15).to_dict(orient='records')
        rankings_output[v] = {
            'variable_name': VAR_INFO[v]['name'],
            'unit': VAR_INFO[v]['disp_unit'],
            'top15_worst_stations': top15
        }

        v_df['variable'] = v
        all_stations_summary_rows.append(v_df)

    combined_df = pd.concat(all_stations_summary_rows, ignore_index=True) if all_stations_summary_rows else pd.DataFrame()

    # Save rankings JSON and CSV to both output_dir and experiments_dir
    target_dirs = [output_dir, experiments_dir]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        json_path = os.path.join(d, 'zero_shot_station_error_ranking.json')
        with open(json_path, 'w') as f:
            json.dump(rankings_output, f, indent=2)
        print(f"Saved station rankings JSON to {json_path}")

        csv_path = os.path.join(d, 'zero_shot_station_error_ranking.csv')
        combined_df.to_csv(csv_path, index=False)
        print(f"Saved station error CSV to {csv_path}")

    # Plot Hydrostatic Bias Correlation (pa vs elevationmsl)
    plot_hydrostatic_correlation(station_error_records['pa'], output_dir)

    # Plot Continental Spatial Map
    plot_african_spatial_error_map(station_error_records, output_dir)

    return rankings_output, combined_df


def plot_hydrostatic_correlation(pa_records, output_dir):
    """
    Plot station elevation (elevationmsl) against surface pressure error (bias).
    Quantifies the hydrostatic lapse rate offset: Delta p = p0 * exp(-g z / R T) - p_obs.
    """
    df = pd.DataFrame(pa_records).dropna(subset=['elevationmsl', 'bias'])
    if df.empty:
        print("No valid surface pressure elevation records to plot.")
        return

    elev = df['elevationmsl'].values
    # Convert Pa to hPa for readability
    bias_hpa = df['bias'].values / 100.0
    mae_hpa = df['mae'].values / 100.0

    slope, intercept, r_val, p_val, std_err = stats.linregress(elev, bias_hpa)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    scatter = ax.scatter(elev, bias_hpa, c=mae_hpa, cmap='plasma', s=60, alpha=0.85, edgecolors='k', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Surface Pressure MAE [hPa]', fontsize=11)

    x_vals = np.linspace(min(elev), max(elev), 100)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, 'r--', linewidth=2.0,
            label=f'Linear Fit: Bias = {slope:.3f} * z + {intercept:.1f} hPa (R^2 = {r_val**2:.3f}, p < {p_val:.1e})')

    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_title('Station Altitude vs Surface Pressure Bias', fontsize=13, fontweight='bold')
    ax.set_xlabel('Station Altitude [m a.s.l.]', fontsize=11)
    ax.set_ylabel('Mean Forecast Bias Delta pa [hPa]', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=10)

    out_file = os.path.join(output_dir, 'elevation_vs_pressure_error.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hydrostatic correlation plot to {out_file}")


def plot_african_spatial_error_map(station_error_records, output_dir):
    """
    Generate geospatial African error maps showing holdout station error distributions.
    Marker color: mean error magnitude / bias.
    Marker size: RMSE.
    """
    vars_to_plot = ['pa', '2t', 'dt', 'ws']
    fig = plt.figure(figsize=(18, 14), dpi=300)

    for i, var in enumerate(vars_to_plot, 1):
        records = station_error_records.get(var, [])
        if not records:
            continue
        df = pd.DataFrame(records).dropna(subset=['mae', 'latitude', 'longitude'])
        if df.empty:
            continue

        scale = VAR_INFO[var]['disp_scale']
        disp_unit = VAR_INFO[var]['disp_unit']
        var_name = VAR_INFO[var]['name']

        if HAS_CARTOPY:
            ax = fig.add_subplot(2, 2, i, projection=ccrs.PlateCarree())
            ax.set_extent([-20, 55, -38, 40], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#f5f5f2', edgecolor='none')
            ax.add_feature(cfeature.OCEAN, facecolor='#e6f2ff', edgecolor='none')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333')
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', edgecolor='#777777')
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax = fig.add_subplot(2, 2, i)
            ax.set_xlim(-20, 55)
            ax.set_ylim(-38, 40)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_xlabel('Longitude [deg E]')
            ax.set_ylabel('Latitude [deg N]')

        # Marker size scaled by RMSE
        rmse_vals = df['rmse'].values * scale
        marker_sizes = 20 + 120 * (rmse_vals - np.nanmin(rmse_vals)) / (np.nanmax(rmse_vals) - np.nanmin(rmse_vals) + 1e-6)

        scatter = ax.scatter(
            df['longitude'],
            df['latitude'],
            c=df['mae'] * scale,
            s=marker_sizes,
            cmap='coolwarm',
            alpha=0.85,
            edgecolors='k',
            linewidths=0.5,
            transform=ccrs.PlateCarree() if HAS_CARTOPY else None
        )
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.06, shrink=0.75)
        cbar.set_label(f'MAE [{disp_unit}] (marker size propto RMSE)', fontsize=10)
        ax.set_title(f"{var_name} ({var}) Holdout Error Distribution", fontsize=12, fontweight='bold')

    plt.suptitle('African Spatial Distribution of Zero-Shot Forecast Errors (150 TAHMO Stations)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_file = os.path.join(output_dir, 'tahmo_spatial_error_map.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved African spatial error map to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot TAHMO Error Analysis & Spatial Diagnostics")
    parser.add_argument('--metrics_csv', type=str, required=True,
                        help='Path to station_evaluation_metrics_detailed.csv')
    parser.add_argument('--p_preds', type=str, required=True,
                        help='Path to predictions rollout zarr')
    parser.add_argument('--p_station', type=str,
                        default='/ritom/scratch/cscs/tligawa/ai-4-weather/data/TAHMO',
                        help='Path to TAHMO ground truth directory')
    parser.add_argument('--station_metadata', type=str,
                        default='/ritom/scratch/cscs/tligawa/ai-4-weather/station_metadata.json',
                        help='Path to station_metadata.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots and ranking summaries')
    parser.add_argument('--experiments_dir', type=str,
                        default='/ritom/scratch/cscs/tligawa/ai-4-weather/experiments/ESFMs',
                        help='Root experiments directory for master rankings')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.experiments_dir, exist_ok=True)

    print("Phase 4 Visual Diagnostics & Spatial Analytics initiated.")
    plot_lead_time_error_curves(args.metrics_csv, args.output_dir)
    compute_station_diagnostics(args.p_preds, args.p_station, args.station_metadata, args.output_dir, args.experiments_dir)
    print("Phase 4 Visual Diagnostics completed successfully.")


if __name__ == '__main__':
    main()
