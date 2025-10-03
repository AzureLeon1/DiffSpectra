# Training

## all spectra, pretrained SpecFormer
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/diffspectra_qm9s.py --config_original_qm9 configs/base_qm9.py --mode train --workdir exp/allspectra_pretrained_specformer --config.data.spectra_version allspectra --config.model.name DMT --config.model.pretrained_specformer_path exp/pretrained_specformer.ckpt

## all spectra
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/diffspectra_qm9s.py --config_original_qm9 configs/base_qm9.py --mode train --workdir exp/allspectra --config.data.spectra_version allspectra --config.model.name DMT

## only IR
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/diffspectra_qm9s.py --config_original_qm9 configs/base_qm9.py --mode train --workdir exp/ir  --config.data.spectra_version ir --config.model.name DMT

## only Raman
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/diffspectra_qm9s.py --config_original_qm9 configs/base_qm9.py --mode train --workdir exp/raman --config.data.spectra_version raman --config.model.name DMT

## only UV-Vis
CUDA_VISIBLE_DEVICES=0,1 python main.py --config configs/diffspectra_qm9s.py --config_original_qm9 configs/base_qm9.py --mode train --workdir exp/uv --config.data.spectra_version uv --config.model.name DMT