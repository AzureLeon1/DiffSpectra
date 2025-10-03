import os
import torch
import logging
import numpy as np
import random
import pickle
import pandas as pd
# from torch.utils import tensorboard

from datasets import get_dataset, inf_iterator, get_dataloader
from models.ema import ExponentialMovingAverage
import losses
from utils import *
from evaluation import *
import visualize
from models import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn, get_cond_sampling_eval_fn
from tqdm import tqdm
import compute_metrics


def set_random_seed(config):
    print(f"Setting random seed: {config.seed}")
    seed = config.seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def save_mol_info(processed_mols, sampled_test_pos, sampled_test_rdkit_mols, save_path):
    """
    Save molecular information to local storage
    
    Args:
        processed_mols: List of predicted molecules
        sampled_test_pos: List of true molecular coordinates
        sampled_test_rdkit_mols: List of true molecular RDKit objects
        save_path: Save path
    """
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    # Save predicted molecular information
    with open(os.path.join(save_path, 'pred_mols.pkl'), 'wb') as f:
        pickle.dump(processed_mols, f)
    # Save true molecular coordinates
    with open(os.path.join(save_path, 'true_pos.pkl'), 'wb') as f:
        pickle.dump(sampled_test_pos, f)
    # Save true molecular RDKit objects
    with open(os.path.join(save_path, 'true_mols.pkl'), 'wb') as f:
        pickle.dump(sampled_test_rdkit_mols, f)


def compute_similarity_metrics(pred_rdmols, true_rdmols, eval_dir, ckpt, version_name):
    """
    Compute similarity metrics between predicted and true molecules, automatically skip None values
    
    Args:
        pred_rdmols: List of predicted RDKit molecules
        true_rdmols: List of true RDKit molecules
        eval_dir: Evaluation results save directory
        ckpt: Checkpoint number
        version_name: Version name ('3D' or '2D')
    
    Returns:
        bool: Whether metrics were successfully computed
    """
    try:
        logging.info(f"Computing {version_name} version similarity metrics")
        
        # Prepare data - each true molecule corresponds to a list of generated molecules, skip None values
        true_mols = []
        pred_mols = []
        skipped_count = 0
        
        # Ensure data length matches and filter valid molecules
        min_len = min(len(pred_rdmols), len(true_rdmols))
        for i in range(min_len):
            # Skip None values
            if pred_rdmols[i] is None:
                logging.debug(f"Skipping predicted molecule {i}: None value")
                skipped_count += 1
                continue
            if true_rdmols[i] is None:
                logging.debug(f"Skipping true molecule {i}: None value")
                skipped_count += 1
                continue
                
            # Validate if molecules are valid
            try:
                import traceback
                from rdkit import Chem
                
                # Initialize ring information for molecules
                pred_mol = pred_rdmols[i]
                true_mol = true_rdmols[i]
                
                # Ensure ring information is initialized for molecules
                if pred_mol is not None:
                    pred_mol.UpdatePropertyCache()
                    Chem.GetSymmSSSR(pred_mol)
                if true_mol is not None:
                    true_mol.UpdatePropertyCache()
                    Chem.GetSymmSSSR(true_mol)
                
                pred_smiles = Chem.MolToSmiles(pred_mol)
                true_smiles = Chem.MolToSmiles(true_mol)
                
                if not pred_smiles or not true_smiles:
                    logging.debug(f"Skipping molecule pair {i}: Unable to generate valid SMILES")
                    skipped_count += 1
                    continue
                    
                true_mols.append(true_mol)
                pred_mols.append([pred_mol])  # Each true molecule corresponds to a list of predicted molecules
                
            except Exception as e:
                logging.debug(f"Skipping molecule pair {i}: Validation failed - {e}")
                skipped_count += 1
                continue
        
        logging.info(f"{version_name} molecule pair statistics - Input: {min_len}, Valid: {len(true_mols)}, Skipped: {skipped_count}")
        
        if len(true_mols) > 0:
            csv_path = os.path.join(eval_dir, f"similarity_metrics_{version_name.lower()}_ckpt_{ckpt}.csv")
            # Pass true_mols and pred_mols as tuple
            compute_metrics.evaluate_jsonl_predictions((true_mols, pred_mols), csv_path)
            logging.info(f"{version_name} similarity metrics saved to: {csv_path}")
            
            # Read CSV file to get detailed metric results
            try:
                results_df = pd.read_csv(csv_path)
                # Output main metrics
                metrics_to_show = [
                    "Exact Match Rate (SMILES)",
                    "Exact Match Rate (InChI Key)",
                    "Tanimoto Similarity (Average)",
                    "Tanimoto (MACCS) Similarity (Average)",
                    "Fraggle Similarity (Average)",
                    "Cosine Similarity (Average)",
                    "Approximate Match Rate (Tanimoto≥0.675)",
                    "Valid Match Rate (Tanimoto≥0.4)",
                    "Functional Group Similarity (Average)",
                    "MCES (Average)",
                ]
                
                for metric in metrics_to_show:
                    if metric in results_df["Evaluation Metric"].values:
                        value = results_df.loc[results_df["Evaluation Metric"] == metric, "Value"].iloc[0]
                        logging.info(f"{version_name} {metric}: {value}")
                
            except Exception as e:
                logging.error(f"Error reading results CSV: {e}")
            
            logging.info(f"Successfully computed {version_name} metrics for {len(true_mols)} molecule pairs")
            return True
        else:
            logging.warning(f"No valid {version_name} molecule pairs found for similarity computation")
            return False
            
    except Exception as e:
        logging.error(f"Error computing {version_name} similarity metrics: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False


def diffspectra_train(config, config_original_qm9, workdir):
    """Runs the training pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Build dataset and dataloader
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)  # only use second_train (second part of training set) when training diffusion
    _, second_train_ds_original, val_ds_original, test_ds_original, dataset_info_original = get_dataset(config_original_qm9)

    train_loader, val_loader, test_loader = get_dataloader(second_train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    # Initialize model
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                        continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    num_train_steps = config.training.n_iters

    if initial_step == 0:
        logging.info(config)

    # get loss step optimizer
    optimize_fn = losses.optimization_manager(config)
    # change step fn
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    # Build sampling functions
    if config.training.snapshot_sampling:
        # change sampling fn
        sampling_fn = get_sampling_fn(config, noise_scheduler, config.training.eval_batch_size,
                                        config.training.eval_samples, inverse_scaler, val_ds)

    # Build evaluation metric
    EDM_metric = get_edm_metric(dataset_info)
    EDM_metric_2D = get_2D_edm_metric(dataset_info)
    logging.info('loading test mols')

    # Training iterations
    for step in range(initial_step, num_train_steps + 1):
        batch = next(train_iter)

        # Execute one training step
        loss = train_step_fn(state, batch)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate, evaluate and save samples
            if config.training.snapshot_sampling:
                ema.store(model.parameters())
                ema.copy_to(model.parameters())

                # Wrapper EDM sampling
                processed_mols, groundtruth_pos, groundtruth_rdmols = sampling_fn(model)

                # EDM evaluation metrics
                stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
                logging.info("step: %d, n_mol: %d, 3D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                                "complete: %.4f, unique & valid: %.4f" % (
                                step, len(sample_rdmols), stability_res['atom_stable'],
                                stability_res['mol_stable'], rdkit_res['Validity'],
                                rdkit_res['Complete'], rdkit_res['Unique']))

                # 2D evaluations
                stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
                logging.info("step: %d, n_mol: %d, 2D atom stability: %.4f, mol stability: %.4f, validity: %.4f, "
                                "complete: %.4f, unique & valid: %.4f" % (
                                step, len(sample_rdmols), stability_res['atom_stable'],
                                stability_res['mol_stable'], rdkit_res['Validity'],
                                rdkit_res['Complete'], rdkit_res['Unique']))

                ema.restore(model.parameters())

                # Visualization of predicted mols
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)
                visualize.visualize_mols(sample_rdmols, this_sample_dir, config)

                # Visualization of ground truth mols
                this_gt_vis_dir = os.path.join(sample_dir, "iter_{}_gt".format(step))
                if not os.path.exists(this_gt_vis_dir):
                    os.makedirs(this_gt_vis_dir)
                visualize.visualize_mols(groundtruth_rdmols, this_gt_vis_dir, config)



def diffspectra_evaluate(config, config_original_qm9, workdir, eval_folder="eval"):
    """Runs the evaluation pipeline with VPSDE for geometry graphs with additional quantum property conditioning."""

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    _, second_train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    _, second_train_ds_original, val_ds_original, test_ds_original, dataset_info_original = get_dataset(config_original_qm9)

    # Initialize model
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                        continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_cond_sampling_eval_fn(config, noise_scheduler, config.eval.batch_size,
                                                config.eval.num_samples, inverse_scaler, test_ds)

    logging.info('loading training mols')
    train_mols = [second_train_ds_original[i].rdmol for i in tqdm(range(len(second_train_ds_original)), desc='Loading training mols')]
    logging.info('loading test mols')
    test_mols = [test_ds_original[i].rdmol for i in tqdm(range(len(test_ds_original)), desc='Loading test mols')]

    # Build evaluation metrics
    logging.info('build EDM metric')
    EDM_metric = get_edm_metric(dataset_info, train_mols)
    logging.info('build EDM 2D metric')
    EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    logging.info('build mose metric')
    mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)
    if config.eval.sub_geometry:
        logging.info('build sub geometry metric')
        sub_geo_mmd_metric = get_sub_geometry_metric(test_mols, dataset_info, config.data.root)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        logging.info('load checkpoint: {}'.format(ckpt_path))
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        if config.eval.enable_sampling:
            logging.info('Sampling -- ckpt: %d' % (ckpt,))
            # Wrapper EDM sampling
            processed_mols, groundtruth_pos, groundtruth_rdmols = sampling_fn(model)
            logging.info('Sampling accomplished')
            
            # EDM evaluation metrics
            stability_res, rdkit_res, sample_rdmols = EDM_metric(processed_mols)
            logging.info('Number of molecules: %d' % len(sample_rdmols))
            logging.info("Metric-3D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f," %
                            (stability_res['atom_stable'], stability_res['mol_stable'], rdkit_res['Validity'],
                            rdkit_res['Complete']))

            # Mose evaluation metrics
            print('compute FCD metric')
            mose_res = mose_metric(sample_rdmols)
            logging.info("Metric-3D || FCD: %.4f" % (mose_res['FCD']))

            # 2D evaluation metrics
            stability_res, rdkit_res, complete_rdmols = EDM_metric_2D(processed_mols)
            logging.info("Metric-2D || atom stability: %.4f, mol stability: %.4f, validity: %.4f, complete: %.4f,"
                            " unique & valid: %.4f, unique & valid & novelty: %.4f" % (stability_res['atom_stable'],
                            stability_res['mol_stable'], rdkit_res['Validity'], rdkit_res['Complete'], rdkit_res['Unique'],
                            rdkit_res['Novelty']))
            mose_res = mose_metric(complete_rdmols)
            logging.info("Metric-2D || FCD: %.4f, SNN: %.4f, Frag: %.4f, Scaf: %.4f, IntDiv: %.4f" % (mose_res['FCD'],
                            mose_res['SNN'], mose_res['Frag'], mose_res['Scaf'], mose_res['IntDiv']))

            # Substructure Geometry MMD
            if config.eval.sub_geometry:
                sub_geo_mmd_res = sub_geo_mmd_metric(complete_rdmols)
                logging.info("Metric-Align || Bond Length MMD: %.4f, Bond Angle MMD: %.4f, Dihedral Angle MMD: %.6f" % (
                    sub_geo_mmd_res['bond_length_mean'], sub_geo_mmd_res['bond_angle_mean'],
                    sub_geo_mmd_res['dihedral_angle_mean']))
                ## bond length
                bond_length_str = ''
                for sym in dataset_info['top_bond_sym']:
                    bond_length_str += f"{sym}: %.4f " % sub_geo_mmd_res[sym]
                logging.info(bond_length_str)
                ## bond angle
                bond_angle_str = ''
                for sym in dataset_info['top_angle_sym']:
                    bond_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(bond_angle_str)
                ## dihedral angle
                dihedral_angle_str = ''
                for sym in dataset_info['top_dihedral_sym']:
                    dihedral_angle_str += f'{sym}: %.4f ' % sub_geo_mmd_res[sym]
                logging.info(dihedral_angle_str)

            # Compute similarity metrics from compute_metrics
            logging.info("Computing similarity metrics from compute_metrics.py")
            
            # 3D version metrics (sample_rdmols vs groundtruth_rdmols)
            compute_similarity_metrics(sample_rdmols, groundtruth_rdmols, eval_dir, ckpt, "3D")
            
            # 2D version metrics (complete_rdmols vs groundtruth_rdmols)
            compute_similarity_metrics(complete_rdmols, groundtruth_rdmols, eval_dir, ckpt, "2D")
            
            # Save molecular data for subsequent analysis
            save_for_analysis = config.eval.save_mols.lower()
            if save_for_analysis == 'true':
                analysis_dir = os.path.join(eval_dir, f"molecules_ckpt_{ckpt}")
                os.makedirs(analysis_dir, exist_ok=True)
                
                # Save 3D version molecules
                with open(os.path.join(analysis_dir, 'sample_rdmols_3d.pkl'), 'wb') as f:
                    pickle.dump(sample_rdmols, f)
                
                # Save 2D version molecules  
                with open(os.path.join(analysis_dir, 'complete_rdmols_2d.pkl'), 'wb') as f:
                    pickle.dump(complete_rdmols, f)
                
                # Save true molecules
                with open(os.path.join(analysis_dir, 'groundtruth_rdmols.pkl'), 'wb') as f:
                    pickle.dump(groundtruth_rdmols, f)
                
                logging.info(f"Molecules saved for analysis in: {analysis_dir}")


run_train_dict = {
    'diffspectra': diffspectra_train,
}


run_eval_dict = {
    'diffspectra': diffspectra_evaluate,
}


def train(config, config_original_qm9, workdir):
    run_train_dict[config.exp_type](config, config_original_qm9, workdir)


def evaluate(config, config_original_qm9, workdir, eval_folder='eval'):
    run_eval_dict[config.exp_type](config, config_original_qm9, workdir, eval_folder)