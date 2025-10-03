import os
import pickle
import logging
import argparse
import traceback
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# add path
import sys
sys.path.append('/path/to/DiffSpectra')
from compute_metrics import evaluate_jsonl_predictions


# Set logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_molecules(file_path):
    """Load saved molecule pickle files"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def validate_and_prepare_mols(pred_rdmols, true_rdmols):
    """Validate and prepare molecules, including initializing ring information
    
    Args:
        pred_rdmols: List of predicted RDKit molecules
        true_rdmols: List of true RDKit molecules
        
    Returns:
        tuple: (true_mols, pred_mols, skipped_count)
    """
    true_mols = []
    pred_mols = []
    skipped_count = 0
    
    # Ensure data length matches and filter valid molecules
    min_len = min(len(pred_rdmols), len(true_rdmols))
    for i in tqdm(range(min_len), desc="Validating molecules"):
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
            
    return true_mols, pred_mols, skipped_count

def compute_metrics_for_saved_mols(base_path, output_path):
    """Compute similarity metrics between saved molecules
    
    Args:
        base_path: Directory path containing molecule pkl files
        output_path: Directory path for output results
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Build file paths
        files = {
            '2d': os.path.join(base_path, 'complete_rdmols_2d.pkl'),
            '3d': os.path.join(base_path, 'sample_rdmols_3d.pkl'),
            'ground_truth': os.path.join(base_path, 'groundtruth_rdmols.pkl')
        }
        
        # Check if all files exist
        for name, path in files.items():
            if not os.path.exists(path):
                logging.error(f"File not found: {path}")
                return
        
        # Load molecules
        mols = {}
        for name, path in files.items():
            mols[name] = load_molecules(path)
            if mols[name] is None:
                logging.error(f"Failed to load molecules from {path}")
                return
            logging.info(f"Loaded {len(mols[name])} molecules from {name}")
        
        # Compute similarity metrics for 2D molecules vs ground truth
        logging.info("Computing metrics for 2D molecules vs ground truth...")
        true_mols_2d, pred_mols_2d, skipped_2d = validate_and_prepare_mols(mols['2d'], mols['ground_truth'])
        logging.info(f"2D molecule pair statistics - Input: {len(mols['2d'])}, Valid: {len(true_mols_2d)}, Skipped: {skipped_2d}")
        
        if len(true_mols_2d) > 0:
            csv_path_2d = os.path.join(output_path, 'similarity_metrics_2d.csv')
            evaluate_jsonl_predictions((true_mols_2d, pred_mols_2d), csv_path_2d)
            logging.info(f"2D similarity metrics saved to: {csv_path_2d}")
            
            # Read and display results
            try:
                results_df = pd.read_csv(csv_path_2d)
                metrics_to_show = [
                    "Top-1 Accuracy",
                    "MCES",
                    "Tanimoto Similarity (Morgan)",
                    "Cosine Similarity (Morgan)",
                    "Tanimoto Similarity (MACCS)",
                    "Fraggle Similarity",
                    "Functional Group Similarity"
                ]
                
                for metric in metrics_to_show:
                    if metric in results_df["Evaluation Metric"].values:
                        value = results_df.loc[results_df["Evaluation Metric"] == metric, "Value"].iloc[0]
                        logging.info(f"2D {metric}: {value}")
            except Exception as e:
                logging.error(f"Error reading 2D results CSV: {e}")
        
        # Compute similarity metrics for 3D molecules vs ground truth
        logging.info("Computing metrics for 3D molecules vs ground truth...")
        true_mols_3d, pred_mols_3d, skipped_3d = validate_and_prepare_mols(mols['3d'], mols['ground_truth'])
        logging.info(f"3D molecule pair statistics - Input: {len(mols['3d'])}, Valid: {len(true_mols_3d)}, Skipped: {skipped_3d}")
        
        if len(true_mols_3d) > 0:
            csv_path_3d = os.path.join(output_path, 'similarity_metrics_3d.csv')
            evaluate_jsonl_predictions((true_mols_3d, pred_mols_3d), csv_path_3d)
            logging.info(f"3D similarity metrics saved to: {csv_path_3d}")
            
            # Read and display results
            try:
                results_df = pd.read_csv(csv_path_3d)
                for metric in metrics_to_show:
                    if metric in results_df["Evaluation Metric"].values:
                        value = results_df.loc[results_df["Evaluation Metric"] == metric, "Value"].iloc[0]
                        logging.info(f"3D {metric}: {value}")
            except Exception as e:
                logging.error(f"Error reading 3D results CSV: {e}")
        
        logging.info(f"All results have been saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error in compute_metrics_for_saved_mols: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")

def main():
    parser = argparse.ArgumentParser(description='Compute metrics for saved molecules')
    parser.add_argument('--base_path', type=str, 
                        default='/path/to/DiffSpectra/exp/allspectra_pretrained_specformer/eval',
                        help='Path to the directory containing molecule pkl files')
    args = parser.parse_args()

    saved_mols_path = args.base_path + '/molecules_ckpt_40'
    output_path = args.base_path + '/metrics_results'
    
    compute_metrics_for_saved_mols(saved_mols_path, output_path)

if __name__ == '__main__':
    main() 