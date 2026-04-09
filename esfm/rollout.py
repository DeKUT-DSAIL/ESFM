"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

import dataclasses
from typing import Generator

import torch

from esfm.batch import Batch
from esfm.model.esfm import ESFM

__all__ = ["rollout"]

def _get_ensemble_median(batch_dict, *, normalize_target: bool = True):
    """
    Compute ensemble median from batch_dict and return batch object.
    """
    for k in batch_dict.surf_vars.keys():
        batch_dict.surf_vars[k] = torch.median(batch_dict.surf_vars[k], dim=1, keepdim=False)[0]
    for k in batch_dict.atmos_vars.keys():
        batch_dict.atmos_vars[k] = torch.median(batch_dict.atmos_vars[k], dim=1, keepdim=False)[0]
    return batch_dict


def rollout(model: ESFM, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`esfm.model.esfm.ESFM`): The model to roll out.
        batch (:class:`esfm.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`esfm.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if model.use_resolution_specific_patch_tokenizers:
        patch_size = model.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
    else:
        patch_size = model.patch_size
    batch = batch.crop(patch_size)
    batch = batch.to(p.device)
    
    print(f'rollout patch size is {patch_size}')

    for _ in range(steps):
        pred = model.forward(batch)
        if isinstance(pred, tuple):
            # If using modified ESFM, preds will have batch_pred, batch_std, batch_ens
            batch_pred, batch_std, batch_ens = pred
            # pred = _get_ensemble_median(batch_ens)
            pred = batch_pred

        yield batch_ens

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
