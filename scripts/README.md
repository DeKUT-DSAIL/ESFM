## Guide for training and evaluation ESFM for experiments in the manuscript:


### ESFM_s* (aka ESFM_s,kd*): ESFM_s_nm: 
1. This model has been previously trained for 40k steps using KD with Aurora small: https://huggingface.co/ESFM/ESFM_s_enc_KD_nm configs/config_ESFM_s_enc_KD_nm.yaml
2. This model has been previously trained for 100k steps: https://huggingface.co/ESFM/ESFM_s_nm_pre configs/config_ESFM_s_nm_pre.yaml
3. https://huggingface.co/ESFM/ESFM_s_nm configs/config_ESFM_s_nm.yaml

Inference on test set: scripts/inference/inference_ESFM_s_nm.sh

Notes: To get ESFM_s*, We first trained an ESFM encoder with KD for 40k steps using train_ESFM_s_enc_KD_nm.sh. Then trained 100k on ERA5 using train_ESFM_s_nm_pre.sh. Then trained 10k on ERA5 using train_ESFM_s_nm.sh. 

### ESFM_s (aka ESFM_s,kd): ESFM_s_wm:
1. https://huggingface.co/ESFM/ESFM_s_enc_KD_nm configs/config_ESFM_s_enc_KD_nm.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_pre configs/config_ESFM_s_wm_pre.yaml
3. https://huggingface.co/ESFM/ESFM_s_wm configs/config_ESFM_s_wm.yaml

Inference on test set: scripts/inference/inference_ESFM_s_wm.sh

Notes: To get ESFM_s, we first trained an ESFM encoder with KD for 40k steps using train_ESFM_s_enc_KD_nm.sh. Then trained 100k on ERA5 using train_ESFM_s_wm_pre.sh. Then trained 10k steps on ERA5 using train_ESFM_s_wm.sh.


### ESFM_s,ri: ESFM_s_wm_ri: 

1. https://huggingface.co/ESFM/ESFM_s_wm_ri_pre configs/config_ESFM_s_wm_ri_pre.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_ri configs/config_ESFM_s_wm_ri.yaml

Inference on test set: scripts/inference/inference_ESFM_s_wm_ri.sh

Notes: To get ESFM_s,ri, we first trained 100k on ERA5 using train_ESFM_s_wm_ri_pre.sh. Then trained 100k steps on ERA5 using train_ESFM_s_wm_ri.sh


### ESFM_s,ci: ESFM_s_wm_ci: 

1. https://huggingface.co/ESFM/ESFM_s_wm_ci_prepre configs/config_ESFM_s_wm_ci_prepre.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_ci_pre configs/config_ESFM_s_wm_ci_pre.yaml
3. https://huggingface.co/ESFM/ESFM_s_wm_ci configs/config_ESFM_s_wm_ci.yaml

Inference on test set: scripts/inference/inference_ESFM_s_wm_ci.sh

Notes: For ESFM_s,ci, we first trained 91.5k on 8xCMIP6 datasets using train_ESFM_s_wm_ci_prepre.sh. Then trained 100k steps on ERA5 using train_ESFM_s_wm_ci_pre.sh. Then trained 10k steps on train_ESFM_s_wm_ci.sh


### ESFM_s+: ESFM_s_nm_ens:

1. See above for ESFM_s_nm Step 1.
2. See above for ESFM_s_nm Step 2.
3. See above for ESFM_s_nm Step 3.
4. https://huggingface.co/ESFM/ESFM_s_nm_ens configs/config_ESFM_s_nm_ens.yaml

Notes: To get ESFM_s+, we initialize ESFM_s_nm, then train it for 10k steps using train_ESFM_s_nm_ens.sh.

### ESFM_s finetuned on MODIS: 

1. https://huggingface.co/ESFM/ESFM_s_enc_KD_nm configs/config_ESFM_s_enc_KD_nm.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_pre configs/config_ESFM_s_wm_pre.yaml
3. https://huggingface.co/ESFM/ESFM_s_wm_modis_pre configs/config_ESFM_s_wm_modis_pre.yaml
4. https://huggingface.co/ESFM/ESFM_s_wm_modis configs/config_ESFM_s_wm_modis.yaml


Inference on test set: scripts/inference/inference_ESFM_s_wm_modis.sh

Notes: For ESFM_s finetuned on MODIS, we take pretrained model from Step 2 of ESFM_s_wm, finetune it on MODIS dataset for 50k steps using train_ESFM_s_wm_modis_pre.sh. Then finetune it on MODIS dataset for 15k steps using train_ESFM_s_wm_modis.sh.


### ESFM_s finetuned on ECMWF 11k:
1. https://huggingface.co/ESFM/ESFM_s_enc_KD_nm configs/config_ESFM_s_enc_KD_nm.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_pre configs/config_ESFM_s_wm_pre.yaml
3. 
    i. 6h lead-time https://huggingface.co/ESFM/ESFM_s_nm_e11k_lt6h configs/config_ESFM_s_nm_e11k_lt6h.yaml; additional config file for holdout station evaluation: configs/config_ESFM_s_nm_e11k_lt6h_ho.yaml
4. 
    ii. 12h lead-time https://huggingface.co/ESFM/ESFM_s_nm_e11k_lt12h configs/config_ESFM_s_nm_e11k_lt12h.yaml; additional config file for holdout station evaluation: configs/config_ESFM_s_nm_e11k_lt12h_ho.yaml
    
    iii. 24h lead-time https://huggingface.co/ESFM/ESFM_s_nm_e11k_lt24h configs/config_ESFM_s_nm_e11k_lt24h.yaml; additional config file for holdout station evaluation: configs/config_ESFM_s_nm_e11k_lt24h_ho.yaml

#### Inference:
**Regular evaluation & Extrapolating holdout stations**: 
1. 6h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval1_lt6h.sh
2. 12h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval1_lt12h.sh
3. 24h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval1_lt24h.sh 

**Holdout station I/O**:
1. 6h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval2_lt6h.sh
2. 12h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval2_lt12h.sh
3. 24h lead time: scripts/inference/inference_ESFM_s_nm_e11k_eval2_lt24h.sh

Notes: For ESFM_s finetuned on ECWMF 11k with 6 hour lead time, we take pretrained model from Step 2 of ESFM_s_wm, then finetuned on ECMWF 11k dataset for 30k steps using train_ESFM_s_nm_e11k_lt6h.sh.
For ESFM_s finetuned on ECWMF 11k with 12 and 24 hour lead times, we take ESFM_s pretrained on ECMWF 11k for 30k steps with 6 hour lead time, and finetune it for 15k steps using train_ESFM_s_nm_e11k_lt12h.sh for 12h lead time model, and train_ESFM_s_nm_e11k_lt24h.sh for 24h lead time model.

### ESFM_s finetuned on Weather-5K: 

1. https://huggingface.co/ESFM/ESFM_s_enc_KD_nm configs/config_ESFM_s_enc_KD_nm.yaml
2. https://huggingface.co/ESFM/ESFM_s_wm_pre configs/config_ESFM_s_wm_pre.yaml
3. 
    i. 6h lead-time https://huggingface.co/ESFM/ESFM_s_wm_w5k_lt6h configs/config_ESFM_s_wm_w5k_lt6h.yaml
    
    ii. 12h lead-time https://huggingface.co/ESFM/ESFM_s_wm_w5k_lt12h configs/config_ESFM_s_wm_w5k_lt12h.yaml
    
    iii. 24h lead-time https://huggingface.co/ESFM/ESFM_s_wm_w5k_lt24h configs/config_ESFM_s_wm_w5k_lt24h.yaml


Inference on test set: 
i. scripts/inference/inference_ESFM_s_wm_w5k_lt6h.sh
ii. scripts/inference/inference_ESFM_s_wm_w5k_lt12h.sh
iii. scripts/inference/inference_ESFM_s_wm_w5k_lt24h.sh

Notes: There is a separate ESFM finetuned for each lead time. All of them are initialized from Step 2 of ESFM_s_wm, then finetuned on Weather-5K dataset for 20k steps. Namely; 
ESFM_s W5K 6h-lead-time: Trained 20k steps using train_ESFM_s_wm_w5k_lt6h.sh.
ESFM_s W5K 12h-lead-time: Trained 20k steps using train_ESFM_s_wm_w5k_lt12h.sh.
ESFM_s W5K 24h-lead-time: Trained 20k steps using train_ESFM_s_wm_w5k_lt24h.sh.
