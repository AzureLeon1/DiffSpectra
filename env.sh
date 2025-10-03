conda create -n diffspectra python=3.10 -y
conda activate diffspectra

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch_geometric==2.4.0
pip install "cython<3.0.0" wheel
pip install "pyyaml==5.4.1" --no-build-isolation
pip install pytorch-lightning==1.3.8

pip install rdkit==2023.3.3
pip install ase
pip install h5py
pip install wandb

pip uninstall torchmetrics
pip install torchmetrics==0.7.2

pip install fcd_torch pomegranate
pip install pandas

git clone git@github.com:AzureLeon1/moses.git
mv moses moses-diffspectra
cd moses-diffspectra
python setup.py install

pip install numpy==1.26.4
pip install ml_collections==0.1.1
pip install pulp
pip install myopic_mces