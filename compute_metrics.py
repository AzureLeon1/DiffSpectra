import json
from typing import List, Tuple

import pulp
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
import pandas as pd
from myopic_mces import MCES
from tqdm import tqdm
from rdkit.Chem.Fraggle import FraggleSim

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
try:
    from rdkit.Chem.MolStandardize.tautomer import (
        TautomerCanonicalizer,
        TautomerTransform,
    )

    _RD_TAUTOMER_CANONICALIZER = "v1"
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform(
            "1,3 heteroatom H shift",
            "[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]",
        ),
        TautomerTransform("1,3 (thio)keto/enol r", "[O,S,Se,Te;X2!H0]-[C]=[C]"),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        TautomerEnumerator,
    )  # newer rdkit

    _RD_TAUTOMER_CANONICALIZER = "v2"

FUNCTIONAL_GROUPS = {
    "alkane": "[CX4]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#C",
    "arene": "[$([cX3](:*):*),$([cX2+](:*):*)]",
    "alcohol": "[#6][OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "haloalkane": "[#6][F,Cl,Br,I]",
    "acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "amine": "[NX3;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitrile": "[NX1]#[CX2]",
    "sulfide": "[#16X2H0]",
    "thiol": "[#16X2H]",
}


def canonical_mol_from_smiles(smiles):
    """Standardize SMILES and return mol object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == "v1":
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol, canonical=True)  # ensure output standardized SMILES


def is_valid(mol):
    smiles = mol2smiles(mol)
    if smiles is None:
        return False

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        return False
    if len(mol_frags) > 1:
        return False

    return True


def load_smiles_from_jsonl(jsonl_path: str) -> List[Tuple[str, str]]:
    """Load predicted and true SMILES from JSONL file

    Args:
        jsonl_path: JSONL file path

    Returns:
        List of (predicted_smiles, true_smiles) tuples
    """
    pred_true_pairs = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Extract predicted SMILES (remove "##SMILES: " prefix)
            pred_smiles = data["predict"].replace("##SMILES: ", "")
            # Extract true SMILES (remove "##SMILES: " prefix)
            true_smiles = data["label"].replace("##SMILES: ", "")
            pred_true_pairs.append((pred_smiles, true_smiles))
    return pred_true_pairs


def identify_functional_groups(mol, functional_groups=FUNCTIONAL_GROUPS):
    results = {}
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                results[name] = len(matches)
    return results


def functional_group_similarity(mol1, mol2):
    fg1 = identify_functional_groups(mol1)
    fg2 = identify_functional_groups(mol2)

    # unique functional groups
    all_groups = set(list(fg1.keys()) + list(fg2.keys()))

    # types
    groups_in_1 = set(fg1.keys())
    groups_in_2 = set(fg2.keys())
    common_groups = groups_in_1.intersection(groups_in_2)

    if len(all_groups) > 0:
        sim = len(common_groups) / len(all_groups)
    else:
        sim = 1.0
    return sim


def evaluate_jsonl_predictions(input_data, output_csv: str):
    """Evaluate molecular prediction results
    
    Args:
        input_data: Can be a JSONL file path (str) or a tuple of (true_mols, pred_mols)
        output_csv: Output CSV file path
    """
    # Process input data
    if isinstance(input_data, str):
        # If it's a JSONL file path
        pred_true_pairs = load_smiles_from_jsonl(input_data)
        print(f"\nLoaded {len(pred_true_pairs)} SMILES pairs in total")

        # Convert to mol objects
        true_mols = []
        pred_mols = []
        invalid_count = 0

        print("\nConverting SMILES to mol objects...")
        for pred_smiles, true_smiles in tqdm(pred_true_pairs, desc="Conversion progress"):
            true_mol = canonical_mol_from_smiles(true_smiles)
            pred_mol = canonical_mol_from_smiles(pred_smiles)

            if true_mol is not None and pred_mol is not None:
                true_mols.append(true_mol)
                pred_mols.append(pred_mol)
            else:
                invalid_count += 1

        print(f"\nConversion completed:")
        print(f"- Successfully converted: {len(true_mols)} pairs")
        print(f"- Invalid SMILES: {invalid_count} pairs")
        print(f"- Conversion rate: {len(true_mols)/len(pred_true_pairs)*100:.2f}%\n")
    else:
        # If it's a list of molecular objects
        true_mols, pred_mols = input_data

    if len(true_mols) > 0:
        exact_match = 0
        exact_match_inchikey = 0  # v2 version exact match
        exactmatch_inchikey_list = []
        tanimoto_scores = []
        tanimoto_maccs_scores = []
        cosine_scores = []
        fraggle_scores = []
        mces_scores = []
        fg_scores = []

        solver = pulp.listSolvers(onlyAvailable=True)[0]
        print("Calculating similarity metrics...")

        # Add atom count lists
        true_atom_counts = []
        pred_atom_counts = []
        
        # Add functional group statistics
        functional_group_stats = {}
        
        for true_mol, pred_mol_list in tqdm(zip(true_mols, pred_mols), total=len(true_mols)):
            # If pred_mol_list is a list, take the first molecule
            pred_mol = pred_mol_list[0] if isinstance(pred_mol_list, list) else pred_mol_list
            
            true_smi = mol2smiles(true_mol)
            pred_smi = mol2smiles(pred_mol)

            # Calculate atom count
            true_atom_count = true_mol.GetNumAtoms()
            pred_atom_count = pred_mol.GetNumAtoms()
            true_atom_counts.append(true_atom_count)
            pred_atom_counts.append(pred_atom_count)

            # exact match (SMILES)
            if true_smi == pred_smi:
                exact_match += 1
            
            # exact match (InChI Key) - v2 version
            try:
                true_inchikey = Chem.MolToInchiKey(true_mol)
                pred_inchikey = Chem.MolToInchiKey(pred_mol)
                if true_inchikey == pred_inchikey:
                    exact_match_inchikey += 1
                    exactmatch_inchikey_list.append(True)
                else:
                    exactmatch_inchikey_list.append(False)
            except Exception:
                # If InChI Key generation fails, skip
                pass

            mces_score = MCES(
                true_smi,
                pred_smi,
                solver=solver,
                threshold=100,
                always_stronger_bound=False,
                solver_options=dict(msg=0),
            )[1]
            mces_scores.append(mces_score)

            # tanimoto of Morgan fingerprints and MACCS
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
            pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
            true_maccs = MACCSkeys.GenMACCSKeys(true_mol)
            pred_maccs = MACCSkeys.GenMACCSKeys(pred_mol)

            tanimoto = DataStructs.TanimotoSimilarity(true_fp, pred_fp)
            tanimoto_maccs = DataStructs.TanimotoSimilarity(true_maccs, pred_maccs)
            cosine = DataStructs.CosineSimilarity(true_fp, pred_fp)
            fg_score = functional_group_similarity(true_mol, pred_mol)

            # Fraggle similarity
            try:
                fraggle = FraggleSim.GetFraggleSimilarity(true_mol, pred_mol)
                if isinstance(fraggle, tuple):
                    fraggle = fraggle[0]
            except Exception:
                fraggle = 0.0
            fraggle_scores.append(fraggle)

            tanimoto_scores.append(tanimoto)
            tanimoto_maccs_scores.append(tanimoto_maccs)
            cosine_scores.append(cosine)
            fg_scores.append(fg_score)

            

        # final results
        results = {
            "Evaluation Metric": [
                "Top-1 Accuracy",
                "MCES",
                "Tanimoto Similarity (Morgan)",
                "Cosine Similarity (Morgan)",
                "Tanimoto Similarity (MACCS)",
                "Fraggle Similarity",
                "Functional Group Similarity"
            ],
            "Value": [
                f"{exact_match_inchikey/len(true_mols):.4f}",
                f"{sum(mces_scores)/len(mces_scores):.4f}",
                f"{sum(tanimoto_scores)/len(tanimoto_scores):.4f}",
                f"{sum(cosine_scores)/len(cosine_scores):.4f}",
                f"{sum(tanimoto_maccs_scores)/len(tanimoto_maccs_scores):.4f}",
                f"{sum(fraggle_scores)/len(fraggle_scores):.4f}",
                f"{sum(fg_scores)/len(fg_scores):.4f}"
            ],
        }

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        # Save detailed lists of all metrics
        detailed_scores = {
            "Top-1 Accuracy": exactmatch_inchikey_list,
            "MCES": mces_scores,
            "Tanimoto Similarity (Morgan)": tanimoto_scores,
            "Cosine Similarity (Morgan)": cosine_scores,
            "Tanimoto Similarity (MACCS)": tanimoto_maccs_scores,
            "Fraggle Similarity": fraggle_scores,
            "Functional Group Similarity": fg_scores,
        }
        
        # Save detailed scores to CSV file
        detailed_output = output_csv.replace('.csv', '_detailed_scores.csv')
        detailed_df = pd.DataFrame(detailed_scores)
        detailed_df.to_csv(detailed_output, index=False, encoding="utf-8-sig")
        
        # Save detailed scores to JSON file (for subsequent analysis)
        json_output = output_csv.replace('.csv', '_detailed_scores.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(detailed_scores, f, ensure_ascii=False, indent=2)
    else:
        print("Error: No valid mol objects available for evaluation!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SMILES prediction results")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")

    args = parser.parse_args()

    evaluate_jsonl_predictions(
        input_data=args.input, output_csv=args.output
    )
