"""Training Conditional JODO with single property on QM9"""

import ml_collections
import torch
import os


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = 'diffspectra'
    config.pred_edge = True
    config.only_2D = False

    # data
    config.data = data = ml_collections.ConfigDict()
    data.root = '/path/to/dataset/QM9S'
    data.name = 'QM9S'
    data.processed_file = ''
    data.info_name = 'qm9_second_half'
    data.num_workers = 16

    data.compress_edge = True
    data.centered = True  # center with 0
    data.include_aromatic = False
    data.atom_types = 5
    data.bond_types = 4
    data.fc_scale = [-1., 1.]
    data.max_node = 29
    
    # spectra
    data.spectra_version = 'allspectra'      # 'ir', 'uv', 'raman', 'allspectra'
    data.aug_translation_scale = 0.1
    data.transform = 'EdgeComSpectra'
    data.use_normalize = True    # use log10 data normalization


    # SDE
    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'  # 'discrete_poly', 'linear', 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    # model
    config.model = model = ml_collections.ConfigDict()

    # # common model parameters
    model.name = 'DMT'
    model.pred_data = True  # False: noise_prediction; True: data_prediction
    model.include_fc_charge = True  # include formal charges of atoms
    model.normalize_factors = '1, 4, 4, 1'  # normalize factors of position, atom types, formal charges, edge types
    model.ema_decay = 0.999
    model.edge_ch = 2   # input edge channels
    model.nf = 256  # node hidden channels
    model.n_layers = 8  # number of blocks
    model.n_heads = 16  # number of attention heads
    model.dropout = 0.1
    model.cond_time = True  # noise level condition
    model.dist_gbf = True  # GBF distance encoding
    model.gbf_name = 'CondGaussianLayer'
    model.self_cond = True  # include self conditioning
    model.self_cond_type = 'ori'  # 'clamp', 'ori'

    model.edge_quan_th = 0.  # edge quantization threshold
    model.n_extra_heads = 2  # attention heads from adjacency matrices
    model.CoM = True  # keep each layer output in CoM
    model.mlp_ratio = 2  # FFN channel ratio
    model.spatial_cut_off = 2.  # distance threshold for spatial adjacency matrix
    model.softmax_inf = True
    model.trans_name = 'TransMixLayer'
    model.cond_ch = 1
    # pretrained_specformer_path
    # model.pretrained_specformer_path= 'exp/pretrained_specformer.ckpt'
    model.pretrained_specformer_path= ''
    model.patch_len = [20,50,50]
    model.stride = [10,25,25]

    # # loss function
    model.loss_weights = '1., 0.25, 0.1'
    model.noise_align = True

    # training
    config.training = training = ml_collections.ConfigDict()
    training.dataloader_drop_last = False
    # Add distributed training related configurations
    training.distributed = True  # whether to enable distributed training
    training.num_gpus = torch.cuda.device_count()  # number of available GPUs
    if training.num_gpus > 1:
        training.dataloader_drop_last = True
    print('number of gpus:', training.num_gpus)
    training.world_size = int(os.environ.get('WORLD_SIZE', training.num_gpus))  # total number of processes
    training.local_rank = int(os.environ.get('LOCAL_RANK', 0))  # local process rank
    
    # Automatically adjust batch size based on GPU count
    base_batch_size = 128  # batch size for single GPU
    training.batch_size = base_batch_size * training.num_gpus  # total batch size
    training.eval_batch_size = base_batch_size * training.num_gpus
    training.eval_samples = base_batch_size * training.num_gpus

    # Modify device configuration, assign GPU based on local_rank
    if training.distributed:
        config.device = torch.device(f'cuda:{training.local_rank}')
    else:
        config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    training.reduce_mean = False
    # training.batch_size = 128
    # training.eval_batch_size = 128
    # training.eval_samples = 128
    training.log_freq = 500 // training.num_gpus

    training.n_iters = 2000000 // training.num_gpus
    training.snapshot_freq = 50000 // training.num_gpus
    # # store additional checkpoints for preemption (meta)
    training.snapshot_freq_for_preemption = 10000 // training.num_gpus
    # # produce samples at each snapshot.
    training.snapshot_sampling = True

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 100000
    optim.grad_clip = 10.
    optim.disable_grad_log = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 1000
    sampling.vis_row = 4
    sampling.vis_col = 4

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.enable_sampling = True
    evaluate.batch_size = base_batch_size * training.num_gpus # generation batch size
    evaluate.num_samples = 10000  # number of samples for evaluation
    evaluate.begin_ckpt = 40
    evaluate.end_ckpt = 40
    evaluate.ckpts = ''  # eg. '30'; '25, 30'
    evaluate.sub_geometry = True
    
    evaluate.save_mols = 'false'

    evaluate.sampling_temperature = 1.0

    config.seed = 42
    # config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config