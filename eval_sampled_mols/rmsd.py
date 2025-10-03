# -*- coding: utf-8 -*-
import os
import sys
import pickle
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import AllChem

def hungarian_atom_mapping(ref_mol, prb_mol, max_distance=5.0, min_atoms=3):
    """
    Calculate atom mapping between two molecules using Hungarian algorithm (first translate/rotate alignment, then final matching)
    
    Parameters:
    - ref_mol: reference molecule
    - prb_mol: molecule to compare  
    - max_distance: maximum allowed inter-atomic distance (Å)
    - min_atoms: minimum number of matching atoms
    
    Returns:
    - atom_map: atom mapping dictionary {prb_idx: ref_idx}
    - rmsd: calculated RMSD value
    - atom_type_accuracy: atom type mapping accuracy
    """
    # Preprocess molecules
    ref = _prep_mol_for_mapping(ref_mol)
    prb = _prep_mol_for_mapping(prb_mol)
    if ref is None or prb is None:
        return None, None, None

    # Get 3D coordinates
    ref_coords = _get_3d_coords(ref)
    prb_coords = _get_3d_coords(prb)
    if ref_coords is None or prb_coords is None:
        return None, None, None

    # -------- Step 1: Translate to respective centroids --------
    ref_centered, ref_centroid = _center_coords(ref_coords)
    prb_centered, prb_centroid = _center_coords(prb_coords)

    # -------- Step 2: Calculate optimal rotation based on rough matching (Kabsch) --------
    # 2.1 Perform a "temporary Hungarian matching" on translation-eliminated coordinates (without threshold or with loose threshold) to estimate correspondence
    tmp_map = _hungarian_match_given_coords(
        ref_centered, prb_centered, ref, prb,
        max_distance=np.inf  # No clipping here first, get as many correspondences as possible for robust rotation estimation
    )
    if not tmp_map or len(tmp_map) < min_atoms:
        # Even if rough correspondence is insufficient, still try PCA principal axis alignment as fallback (correspondence-independent)
        R_pca = _pca_principal_axes_alignment(prb_centered, ref_centered)
        prb_aligned = _apply_rotation(prb_centered, R_pca)
    else:
        # 2.2 Use temporary correspondence to run Kabsch, get optimal rotation
        P = prb_centered[list(tmp_map.keys()), :]  # prb subset
        Q = ref_centered[list(tmp_map.values()), :]  # ref subset
        R = _kabsch_rotation(P, Q)
        prb_aligned = _apply_rotation(prb_centered, R)

    # -------- Step 3: Perform "final Hungarian matching + threshold clipping + RMSD" on aligned coordinates --------
    final_map = _hungarian_match_given_coords(
        ref_centered, prb_aligned, ref, prb, max_distance=max_distance
    )
    if not final_map or len(final_map) < min_atoms:
        return None, None, None

    # Calculate RMSD (on already aligned coordinates)
    rmsd = _calculate_rmsd_with_mapping(prb_aligned, ref_centered, final_map)
    
    # Calculate atom type mapping accuracy
    atom_type_accuracy = _calculate_atom_type_accuracy(ref, prb, final_map)
    
    return final_map, rmsd, atom_type_accuracy


# ====================== Utility Functions ======================

def _prep_mol_for_mapping(mol):
    """Preprocess molecule for mapping"""
    if mol is None:
        return None
    m = Chem.Mol(mol)  # Copy
    # Largest connected fragment
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if frags:
        m = max(frags, key=lambda x: x.GetNumAtoms())
    # Ensure 3D conformation exists
    if m.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(m, randomSeed=42)
        except Exception:
            return None
    return m

def _get_3d_coords(mol):
    """Get 3D coordinates of molecule"""
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords, dtype=float)

def _center_coords(coords):
    """Return coordinates after centroid removal and centroid"""
    centroid = coords.mean(axis=0, keepdims=True)
    return coords - centroid, centroid

def _kabsch_rotation(P, Q):
    """
    Kabsch algorithm: Given two point sets P, Q (Nx3, already centroid-removed, and one-to-one correspondence),
    find rotation matrix R (right multiplication) that minimizes || P R - Q ||_F
    """
    # SVD of P^T Q
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    # Handle chirality flip (reflection) case, ensure det(R) = +1
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    return R

def _apply_rotation(coords, R):
    """Right multiply coordinates by rotation matrix R (Nx3 with 3x3)"""
    return coords @ R

def _pca_principal_axes_alignment(P, Q):
    """
    Pure geometric principal axis alignment (no need for atom one-to-one correspondence):
    - Align P's principal axes to Q's principal axes, return a 3x3 rotation matrix
    - As fallback when Kabsch cannot obtain stable correspondence
    """
    def _principal_axes(X):
        # Eigenvalue decomposition of covariance matrix (X already centroid-removed)
        C = np.cov(X.T)
        w, V = np.linalg.eigh(C)
        # From small to large, reverse to from large to small
        idx = np.argsort(w)[::-1]
        return V[:, idx]

    VP = _principal_axes(P)
    VQ = _principal_axes(Q)
    # Align P's axes to Q: R such that P*R ≈ Q (in principal axis space)
    R = VP @ VQ.T
    # Handle reflection
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R

def _hungarian_match_given_coords(ref_coords, prb_coords, ref_mol, prb_mol, max_distance=np.inf):
    """
    On given (possibly already aligned) coordinates, build distance matrix and perform Hungarian matching.
    Return {prb_idx: ref_idx} (only keep matches with distance <= max_distance)
    """
    dist_mat = _build_distance_matrix(ref_coords, prb_coords, ref_mol, prb_mol)
    # Distance threshold clipping
    if np.isfinite(max_distance):
        dist_mat = dist_mat.copy()
        dist_mat[dist_mat > max_distance] = 1000.0

    try:
        prb_indices, ref_indices = linear_sum_assignment(dist_mat)
    except Exception as e:
        warnings.warn(f"Hungarian algorithm failed: {e}")
        return None

    atom_map = {}
    for p_idx, r_idx in zip(prb_indices, ref_indices):
        if dist_mat[p_idx, r_idx] <= (max_distance if np.isfinite(max_distance) else np.inf):
            atom_map[p_idx] = r_idx
    return atom_map

def _build_distance_matrix(ref_coords, prb_coords, ref_mol, prb_mol):
    """Build distance matrix considering atom types and spatial distances"""
    n_ref = len(ref_coords)
    n_prb = len(prb_coords)
    distance_matrix = np.zeros((n_prb, n_ref), dtype=float)
    for i in range(n_prb):
        ai = prb_mol.GetAtomWithIdx(i)
        for j in range(n_ref):
            aj = ref_mol.GetAtomWithIdx(j)
            spatial_dist = np.linalg.norm(prb_coords[i] - ref_coords[j])
            atom_penalty = _get_atom_type_penalty(ai, aj)
            distance_matrix[i, j] = spatial_dist + atom_penalty
    return distance_matrix

def _get_atom_type_penalty(atom1, atom2):
    """Calculate penalty value for atom type mismatch"""
    s1, s2 = atom1.GetSymbol(), atom2.GetSymbol()
    if s1 == s2:
        return 0.0
    elif s1 in ['C', 'N', 'O', 'S'] and s2 in ['C', 'N', 'O', 'S']:
        return 2.0  # Substitution penalty between light atoms
    else:
        return 10.0  # Severe penalty for heavy atom substitution

def _calculate_rmsd_with_mapping(prb_coords, ref_coords, atom_map):
    """Calculate RMSD based on atom mapping (based on prb_coords already aligned to ref_coords)"""
    if not atom_map:
        return None
    diffs2 = []
    for prb_idx, ref_idx in atom_map.items():
        d = np.linalg.norm(prb_coords[prb_idx] - ref_coords[ref_idx])
        diffs2.append(d * d)
    return float(np.sqrt(np.mean(diffs2))) if diffs2 else None


def _calculate_atom_type_accuracy(ref_mol, prb_mol, atom_map):
    """Calculate accuracy of atom type mapping"""
    if not atom_map:
        return 0.0
    
    correct_mappings = 0
    for prb_idx, ref_idx in atom_map.items():
        # Ensure indices are Python int type, not numpy type
        ref_idx_int = int(ref_idx)
        prb_idx_int = int(prb_idx)
        
        ref_atom = ref_mol.GetAtomWithIdx(ref_idx_int)
        prb_atom = prb_mol.GetAtomWithIdx(prb_idx_int)
        if ref_atom.GetSymbol() == prb_atom.GetSymbol():
            correct_mappings += 1
    
    return correct_mappings / len(atom_map) if atom_map else 0.0


# ====================== Batch Processing Interface (kept as is) ======================

def hungarian_rmsd_batch(ref_mols, prb_mols, max_distance=5.0, min_atoms=3, verbose=False):
    """
    Batch calculate RMSD and atom type mapping accuracy for multiple molecule pairs (using two-stage alignment + final matching)
    """
    assert len(ref_mols) == len(prb_mols), "Molecule list lengths must be consistent"
    rmsd_list = []
    atom_type_accuracy_list = []
    success_count = 0

    for i, (ref, prb) in enumerate(zip(ref_mols, prb_mols)):
        try:
            atom_map, rmsd, atom_type_accuracy = hungarian_atom_mapping(ref, prb, max_distance, min_atoms)
            if rmsd is not None:
                rmsd_list.append(rmsd)
                atom_type_accuracy_list.append(atom_type_accuracy)
                success_count += 1
                if verbose and i % 100 == 0:
                    print(f"Molecule {i}: RMSD = {rmsd:.3f}, matched atoms = {len(atom_map)}, atom type accuracy = {atom_type_accuracy:.2%}")
            else:
                rmsd_list.append(None)
                atom_type_accuracy_list.append(None)
                if verbose:
                    print(f"Molecule {i}: mapping failed")
        except Exception as e:
            rmsd_list.append(None)
            atom_type_accuracy_list.append(None)
            if verbose:
                print(f"Molecule {i}: error - {e}")

    valid_rmsds = [r for r in rmsd_list if r is not None]
    valid_accuracies = [a for a in atom_type_accuracy_list if a is not None]
    success_rate = success_count / len(ref_mols)
    mean_rmsd = np.mean(valid_rmsds) if valid_rmsds else None
    mean_atom_type_accuracy = np.mean(valid_accuracies) if valid_accuracies else None

    print(f"Total molecules: {len(ref_mols)}")
    print(f"Successfully calculated: {success_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean RMSD: {mean_rmsd:.3f}" if mean_rmsd is not None else "Mean RMSD: no valid data")
    print(f"Mean atom type accuracy: {mean_atom_type_accuracy:.2%}" if mean_atom_type_accuracy is not None else "Mean atom type accuracy: no valid data")

    return rmsd_list, success_rate, mean_rmsd, mean_atom_type_accuracy


if __name__ == "__main__":
    exp_name_list = [
        "allspectra_pretrained_specformer",
        # "allspectra",
        # "ir",
        # "raman",
        # "uv",
    ]
    for exp_name in exp_name_list:
        print(f"Processing {exp_name}...")
        DEFAULT_BASE = f"/path/to/DiffSpectra/exp/{exp_name}/eval/molecules_ckpt_40"
        GT_NAME = "groundtruth_rdmols.pkl"
        SM_NAME = "sample_rdmols_3d.pkl"

        with open(os.path.join(DEFAULT_BASE, GT_NAME), "rb") as f:
            groundtruth_rdmols = pickle.load(f)
        with open(os.path.join(DEFAULT_BASE, SM_NAME), "rb") as f:
            sampled_rdmols = pickle.load(f)

        rmsds, success_rate, mean_rmsd, mean_atom_type_accuracy = hungarian_rmsd_batch(
            groundtruth_rdmols,
            sampled_rdmols,
            max_distance=5.0,
            min_atoms=3,
            verbose=False
        )