# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import torch
import logging
import torch.nn.functional as F
from esfm import Batch

epsilon = 1e-6

class WeightedMAELoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        replace_nan_with_zero: bool = False,
    ) -> None:
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.dataset_weight = dataset_weight
        self.surf_var_weights = surf_var_weights
        self.atmos_var_weights = atmos_var_weights
        self.reduction = reduction
        if not reduction:
            raise NotImplementedError('reduction=False is not implemented for WeightedMAELoss. Please use reduction=True.')
        self.latitude_weight = latitude_weight
        self.replace_nan_with_zero = replace_nan_with_zero

    def __call__(self, pred_batch, target_batch, latitude_weight_enabled=None, pred_batch_has_ensemble_dim=True):
        if latitude_weight_enabled is None:
            latitude_weight_enabled = self.latitude_weight
        
        if latitude_weight_enabled:
            latitudes = torch.deg2rad(pred_batch.metadata.lat)
            latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean()
            latitude_weight = latitude_weight[None, None, :, None] # shape (1, 1, lat, 1)

        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, self.surf_weight, self.surf_var_weights), 
            (target_batch.atmos_vars, pred_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]
        loss_dict = {}
        total_loss = 0
        for target, pred, group_weight, var_weights in groups:
            group_loss = 0
            for var_name in sorted(target.keys()):
                pred_var = pred[var_name]
                target_var = target[var_name]
                
                # Create mask for non-NaN values in target_var
                mask_nonnan = ~torch.isnan(target_var) # [bs, ..., lat, lon]
                
                # Handle dimensional mismatch between pred_var and target_var
                mask_expanded = mask_nonnan
                if pred_var.shape != target_var.shape:
                    # Expand mask to match pred_var shape for proper indexing
                    while mask_expanded.ndim < pred_var.ndim:
                        mask_expanded = mask_expanded.unsqueeze(1)
                    # Broadcast the mask to match pred_var shape
                    mask_expanded = mask_expanded.expand_as(pred_var)
                    
                    # Compute error on full tensors first, then mask
                    if pred_batch_has_ensemble_dim: 
                        target_var = target_var.unsqueeze_(1)  # Add ensemble dimension to target
                        target_var = target_var.expand_as(pred_var)  # Expand target to match pred shape
                    err_full = abs(pred_var - target_var)  # Broadcasting handles dimension mismatch
                    err = err_full[mask_expanded]  # Select only valid elements
                else:
                    # Same shape case - direct masking
                    err = abs(pred_var[mask_nonnan] - target_var[mask_nonnan])
                
                # Handle case where all target values are NaN
                if err.numel() == 0:
                    err = torch.tensor(0.0, device=target_var.device)
                else:
                    if latitude_weight_enabled:
                        lat_weights_expanded = latitude_weight
                        while lat_weights_expanded.ndim < pred_var.ndim:
                            lat_weights_expanded = lat_weights_expanded.unsqueeze(1)
                        lat_weights_expanded = lat_weights_expanded.expand_as(pred_var)
                        lat_weights_masked = lat_weights_expanded[mask_expanded]
                        
                        err = err * lat_weights_masked
 
                # calculate the average loss over entire loss
                if self.replace_nan_with_zero:
                    mean = err.nanmean()
                    mean = torch.tensor(0, device=target_var.device) if mean.isnan() else mean
                else:
                    mean = err.mean()

                loss_dict[var_name] = mean
                group_loss += mean * var_weights.get(var_name, torch.tensor(1.0, device=target_var.device))

            total_loss += group_weight * group_loss

        loss_dict['total_mae'] = total_loss
        #return (self.dataset_weight / num_vars) * total_loss, loss_dict # leave the normalization to CombinedLoss
        return  total_loss, loss_dict

    def get_loss(self, pred_batch, target_batch, **kwargs): # this is here only to support backward compatibility in the code.
        return self.__call__(pred_batch, target_batch)


class CombinedLoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        mae_weight: float = 1.0,
        nll_weight: float = 1.0,
        crps_weight: float = 1.0,
        kernel_crps_weight: float = 1.0,
        stats_loss_weight: float = 1.0,
        kill_if_nan_in_preds: bool = True,
        almost_fair_crps_alpha: float = 1.0, # use 0.95 for AIFS afCRPS
        mae_on_ensemble_mean: bool = True,
    ) -> None:
        # Define default weights if None is provided
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        # Initialize the WeightedMAELoss with the same weights
        if kill_if_nan_in_preds:
            replace_mae_nan_with_zero = False
        else:
            replace_mae_nan_with_zero = True
        self.mae_loss = WeightedMAELoss(
            surf_weight=surf_weight,
            atmos_weight=atmos_weight,
            surf_var_weights=surf_var_weights,
            atmos_var_weights=atmos_var_weights,
            dataset_weight=dataset_weight,
            reduction=reduction,
            latitude_weight=latitude_weight,
            replace_nan_with_zero=replace_mae_nan_with_zero,
        )
        
        # Store parameters
        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.surf_var_weights = surf_var_weights  # Now guaranteed to not be None
        self.atmos_var_weights = atmos_var_weights  # Now guaranteed to not be None
        self.dataset_weight = dataset_weight
        self.reduction = reduction
        self.latitude_weight = latitude_weight
        self.mae_on_ensemble_mean = mae_on_ensemble_mean
        
        # Weights for different loss components
        self.mae_weight = mae_weight
        self.nll_weight = nll_weight
        self.crps_weight = crps_weight
        self.kernel_crps_weight = kernel_crps_weight
        self.stats_loss_weight = stats_loss_weight
        self.almost_fair_crps_alpha = almost_fair_crps_alpha
        
        self.kill_if_nan_in_preds = kill_if_nan_in_preds
        
    def stats_loss(self, x, mu, std):
        ### adapted from stats loss from atmorep: https://github.com/clessig/atmorep/blob/main/atmorep/utils/utils.py#L347 ###
        stats_loss = torch.exp(-0.5 * (x-mu)*(x-mu) / (std*std + epsilon))
        diff = stats_loss - 1.
        stats_loss = torch.mean(diff * diff) + torch.mean(torch.sqrt(torch.abs(std)))
        return stats_loss

    def gaussian_nll(self, y, mean, std):
        """Gaussian Negative Log Likelihood"""
        return 0.5 * torch.log(2 * torch.pi * (std**2 + epsilon)) + \
               0.5 * ((y - mean)**2) / (std**2 + epsilon)

    def crps_loss(self, y, mean, std):
        """Continuous Ranked Probability Score for Gaussian distribution"""
        normalized_diff = (y - mean) / (std + epsilon)
        # Convert pi to a tensor with the same device as the input tensors
        pi_tensor = torch.tensor(torch.pi, device=y.device)
        return std * (normalized_diff * (2 * torch.erf(normalized_diff / torch.sqrt(torch.tensor(2.0, device=y.device))) - 1) + \
               2 / torch.sqrt(pi_tensor) * torch.exp(-(normalized_diff**2 / 2)))

    def expensive_kernel_crps(self, y, ensemble_preds):
        """Kernel CRPS using ensemble predictions"""
        diff_ey = torch.abs(ensemble_preds - y.unsqueeze(1))  # [B, E, ...]
        diff_ee = torch.abs(ensemble_preds.unsqueeze(2) - ensemble_preds.unsqueeze(1))  # [B, E, E, ...]
        return torch.mean(diff_ey, dim=1) - 0.5 * torch.mean(diff_ee, dim=(1,2))

    # Efficient kernel CRPS: O(m log m)
    # https://docs.nvidia.com/physicsnemo/latest/_modules/physicsnemo/metrics/general/crps.html
    # E|X-y|/m - 1/(2m(m-1)) \sum_{i,j=1}\|x_i - x_j\|
    def kernel_crps(self, y, ensemble_preds, biased: bool = False):
        """Efficient kernel CRPS implementation with O(m log m) complexity."""
        if ensemble_preds.ndim == 5 and y.ndim == 3:
            skill = torch.abs(ensemble_preds - y[:, None, None, ...]).mean(1)  # Mean over ensemble dimension
        elif (ensemble_preds.ndim - y.ndim) == 1:
            skill = torch.abs(ensemble_preds - y.unsqueeze(1)).mean(1)  # Mean over ensemble dimension
        elif ensemble_preds.ndim == y.ndim:
            skill = torch.abs(ensemble_preds - y).mean(1)  # Mean over ensemble dimension
        else:
            raise ValueError(f'Unexpected dimensions: ensemble_preds.ndim={ensemble_preds.ndim}, y.ndim={y.ndim}')
        pred, _ = torch.sort(ensemble_preds, dim=1)  # Sort along ensemble dimension so that xi-xj >= 0. (bs, #ens, time=1, [plev], H, W)
        
        # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
        # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
        #                           = 2 sum_(i <= j) |x_i -x_j|
        #                           = 2 sum_(i <= j) (x_j - x_i)
        #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
        #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
        #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
        
        # Efficient spread calculation
        m = pred.size(1)  # Ensemble size
        i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
        denom = m * m if biased else m * (m - 1)
        factor = (2 * i - m - 1) / denom
        spread = torch.sum(factor.view(1, -1, *([1] * (pred.dim() - 2))) * pred, dim=1)
        
        # print(f'shape skill: {skill.shape}, shape spread: {spread.shape}')
        if self.almost_fair_crps_alpha != 1.0:
            eps = (1-self.almost_fair_crps_alpha)/m
            spread = (1-eps)*spread
        return skill - spread
    
    def _check_for_nan(self, pred_batch):
        any_nans = False
        nan_str = ''
        for k in pred_batch.atmos_vars.keys():
            nan = torch.isnan(pred_batch.atmos_vars[k])
            if nan.any():
                any_nans = True
                nan_str += f'Found NaN in pred_batch.atmos_vars[{k}]. Number of NaNs: {nan.sum()}/{nan.numel()}, pred feature shape: {pred_batch.atmos_vars[k].shape}\n'
        for k in pred_batch.surf_vars.keys():
            nan = torch.isnan(pred_batch.surf_vars[k])
            if nan.any():
                any_nans = True
                nan_str += f'Found NaN in pred_batch.surf_vars[{k}]. Number of NaNs: {nan.sum()}/{nan.numel()}, pred feature shape: {pred_batch.surf_vars[k].shape}\n'
        if any_nans:
            logging.warning(f'NaNs found in predictions:\n{nan_str}')
        return any_nans

    def get_loss(self, pred_batch, std_batch, ens_batch, target_batch, latitude_weight_enabled=None):
        
        if latitude_weight_enabled is None:
            latitude_weight_enabled = self.latitude_weight
        if self.mae_on_ensemble_mean:
            mae_total, mae_dict = self.mae_loss(pred_batch, target_batch, latitude_weight_enabled=latitude_weight_enabled)
        else:
            mae_total, mae_dict = self.mae_loss(ens_batch, target_batch, latitude_weight_enabled=latitude_weight_enabled)
        losses = mae_dict  # Start with MAE losses, even if not used in total_loss
        if self.mae_weight != 0:
            total_loss = self.mae_weight * mae_total
        else:
            total_loss = torch.tensor(0.0, device=pred_batch.metadata.lat.device)
        
        if self.kill_if_nan_in_preds and self._check_for_nan(pred_batch):
            raise ValueError('NaNs found in predictions. Aborting training.')
        
        if latitude_weight_enabled:
            latitudes = torch.deg2rad(pred_batch.metadata.lat)
            latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean() if latitude_weight_enabled else 1.0

        num_vars = (len(target_batch.surf_vars) + len(target_batch.atmos_vars))

        # Process both surface and atmospheric variables for other losses
        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, std_batch.surf_vars, 
             ens_batch.surf_vars, self.surf_weight, self.surf_var_weights),
            (target_batch.atmos_vars, pred_batch.atmos_vars, std_batch.atmos_vars, 
             ens_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]

        if self.stats_loss_weight != 0 or self.nll_weight != 0 or self.crps_weight != 0 or self.kernel_crps_weight != 0:
            for target, pred, std, ens, group_weight, var_weights in groups:
                group_loss = 0
                for var_name in sorted(target.keys()):
                    weight = var_weights.get(var_name, 1.0)
                    
                    # Apply latitude weighting if enabled
                    if latitude_weight_enabled:
                        target_var = target[var_name] * latitude_weight[..., None]
                        pred_var = pred[var_name] * latitude_weight[..., None]
                        std_var = std[var_name] * latitude_weight[..., None]
                        ens_var = ens[var_name] * latitude_weight[..., None]
                    else:
                        target_var = target[var_name]
                        pred_var = pred[var_name]
                        std_var = std[var_name]
                        ens_var = ens[var_name]

                    # Only compute losses if their weights are non-zero
                    stats_loss = torch.nan_to_num(self.stats_loss(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.stats_loss_weight != 0 else torch.zeros_like(total_loss)
                    nll = torch.nan_to_num(self.gaussian_nll(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.nll_weight != 0 else torch.zeros_like(total_loss)
                    crps = torch.nan_to_num(self.crps_loss(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.crps_weight != 0 else torch.zeros_like(total_loss)
                    kernel_crps = torch.nan_to_num(self.kernel_crps(target_var, ens_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.kernel_crps_weight != 0 else torch.zeros_like(total_loss)

                    # Store individual losses only if their weights are non-zero
                    if self.stats_loss_weight != 0:
                        losses[f'tail/{var_name}_stats'] = stats_loss
                    if self.nll_weight != 0:
                        losses[f'tail/{var_name}_nll'] = nll
                    if self.crps_weight != 0:
                        losses[f'tail/{var_name}_crps'] = crps
                    if self.kernel_crps_weight != 0:
                        losses[f'tail/{var_name}_kcrps'] = kernel_crps

                    # Combine statistical losses with their respective weights
                    var_loss = (self.stats_loss_weight * stats_loss +
                                self.nll_weight * nll + 
                                self.crps_weight * crps + 
                                self.kernel_crps_weight * kernel_crps)

                    group_loss += var_loss * weight

                total_loss += group_weight * group_loss

            if self.kernel_crps_weight != 0:
                keys_losses = [k for k in losses.keys() if k.startswith('tail/') and k.endswith('_kcrps')]
                losses['total_kcrps'] = sum(losses[k] for k in keys_losses)

        # Before final normalization
        total_loss = (self.dataset_weight / num_vars) * total_loss
        if not isinstance(total_loss, torch.Tensor):
            raise ValueError(f'total_loss is not a torch.Tensor. It is of type {type(total_loss)} with value {total_loss}.')
            # total_loss = torch.tensor(total_loss, requires_grad=True, device=pred_batch.metadata.lat.device)
 
        losses['total'] = total_loss
        # assert total_loss > 0, 'total loss is zero'

        return total_loss, losses

class KD_Loss_activations:
    """Knowledge Distillation Loss on activations. """
    
    def __init__(self, criterion='l1'):
        if str(criterion).lower() == 'l2':
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        elif str(criterion).lower() == 'l1':
            self.loss_fn = torch.nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Supported: 'l2', 'l1'.")
    
    def __call__(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        """Compute the knowledge distillation loss.
        
        Args:
            student_logits: Logits from the student model.
            teacher_logits: Logits from the teacher model.
        
        Returns:
            Computed KD loss.
        """
        loss = self.loss_fn(student_logits, teacher_logits)
        
        return loss
    
    def get_loss(self, pred_batch, target_batch):
        """Get the loss for list of student and teacher logits.
        
        Args:
            pred_batch: List of student logits.
            target_batch: List of teacher logits.
        
        Returns:
            Computed KD loss.
        """
        return self.__call__(student_logits=pred_batch, teacher_logits=target_batch)



