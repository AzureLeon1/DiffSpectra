# DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models

This is the code for the Paper: [DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models](https://arxiv.org/abs/2507.06853).

<img src="assets/1_overview.pdf" alt="model" style="zoom: 45%;" />

## Model Description
**DiffSpectra** is a generative framework for **molecular structure elucidation from multi-modal spectral data**. Unlike retrieval-based approaches that rely on finite molecular libraries or SMILES-based autoregressive models that often ignore 3D geometry, DiffSpectra formulates structure elucidation as a **conditional diffusion process**.  

The framework integrates two core components:  

- **Diffusion Molecule Transformer (DMT):** An SE(3)-equivariant denoising network that models both **2D topology** and **3D geometry** of molecules.  
- **SpecFormer:** A transformer-based spectral encoder that captures intra- and inter-spectrum dependencies across diverse spectral modalities (e.g., IR, Raman, UV-Vis).  

Through spectrum-conditioned diffusion modeling, DiffSpectra unifies multi-modal reasoning with 2D/3D generative modeling. Extensive experiments demonstrate that DiffSpectra achieves **40.76% top-1 accuracy** and **99.49% top-10 accuracy** in recovering exact molecular structures.  

<img src="assets/3_performance.pdf" alt="model" style="zoom: 50%;" />

<img src="assets/2_visualization.pdf" alt="model" style="zoom: 55%;" />

To our knowledge, DiffSpectra is the **first framework** that unifies multi-modal spectral reasoning and joint 2D/3D generative modeling for *de novo* molecular structure elucidation.

- **Paper:** [https://arxiv.org/abs/2507.06853](https://arxiv.org/abs/2507.06853)
- **Code:** [https://github.com/AzureLeon1/DiffSpectra](https://github.com/AzureLeon1/DiffSpectra)
- **Dataset:** [https://huggingface.co/datasets/AzureLeon1/DiffSpectra](https://huggingface.co/datasets/AzureLeon1/DiffSpectra)
- **Model Checkpoins:** [https://huggingface.co/AzureLeon1/DiffSpectra](https://huggingface.co/AzureLeon1/DiffSpectra)

---

## Usage

1. Install the environment according to the commands in `env.sh`.
2. Download and place the dataset in your configured `path/to/dataset` directory.  
3. Modify the configuration file in `configs/` to set `data.root` to your configured path.  
4. [Optional] Download and place the trained model checkpoints in the `exp/` directory.
5. Use the provided scripts in `scripts/` for training or evaluation.  

---

### Training

You can train the model with different spectra settings as follows:

- **All spectra with pretrained SpecFormer**  
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode train \
    --workdir exp/allspectra_pretrained_specformer \
    --config.data.spectra_version allspectra \
    --config.model.name DMT \
    --config.model.pretrained_specformer_path exp/pretrained_specformer.ckpt

- **All spectra**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode train \
    --workdir exp/allspectra \
    --config.data.spectra_version allspectra \
    --config.model.name DMT
  ```

- **Only IR**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode train \
    --workdir exp/ir \
    --config.data.spectra_version ir \
    --config.model.name DMT
  ```

- **Only Raman**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode train \
    --workdir exp/raman \
    --config.data.spectra_version raman \
    --config.model.name DMT
  ```

- **Only UV-Vis**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode train \
    --workdir exp/uv \
    --config.data.spectra_version uv \
    --config.model.name DMT
  ```

------

### Evaluation

To evaluate the trained model, run:

- **All spectra with pretrained SpecFormer**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode eval \
    --workdir exp/allspectra_pretrained_specformer \
    --config.eval.ckpts '40' \
    --config.eval.num_samples 10000 \
    --config.eval.save_mols true \
    --config.data.spectra_version allspectra
  ```

- **All spectra**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode eval \
    --workdir exp/allspectra \
    --config.eval.ckpts '40' \
    --config.eval.num_samples 10000 \
    --config.eval.save_mols true \
    --config.data.spectra_version allspectra
  ```

- **Only IR**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode eval \
    --workdir exp/ir \
    --config.eval.ckpts '40' \
    --config.eval.num_samples 10000 \
    --config.eval.save_mols true \
    --config.data.spectra_version ir
  ```

- **Only Raman**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode eval \
    --workdir exp/raman \
    --config.eval.ckpts '40' \
    --config.eval.num_samples 10000 \
    --config.eval.save_mols true \
    --config.data.spectra_version raman
  ```

- **Only UV-Vis**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --config configs/diffspectra_qm9s.py \
    --config_original_qm9 configs/base_qm9.py \
    --mode eval \
    --workdir exp/uv \
    --config.eval.ckpts '40' \
    --config.eval.num_samples 10000 \
    --config.eval.save_mols true \
    --config.data.spectra_version uv
  ```

---

## Contact

For questions, feedback, or collaborations, please reach out to:
📧 liang.wang@cripac.ia.ac.cn

---

## Cite Us

If you use DiffSpectra in your research or applications, please cite:
```
@article{wang2025diffspectra,
  title={DiffSpectra: Molecular Structure Elucidation from Spectra using Diffusion Models},
  author={Liang Wang and Yu Rong and Tingyang Xu and Zhenyi Zhong and Zhiyuan Liu and Pengju Wang and Deli Zhao and Qiang Liu and Shu Wu and Liang Wang and Yang Zhang},
  journal={arXiv},
  volume={abs/2502.09511},
  year={2025}
}
```
