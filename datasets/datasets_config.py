qm9_with_h = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 4, 5: 5, 6: 9, 7: 16, 8: 49, 9: 124, 10: 362, 11: 807, 12: 1689, 13: 3060, 14: 5136,
                        15: 7796, 16: 10644, 17: 13025, 18: 13364, 19: 13832, 20: 9482, 21: 9970, 22: 3393, 23: 4848,
                        24: 539, 25: 1506, 26: 48, 27: 266, 29: 25},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                            'H1C-C1N-N1C', 'H1C-C1C-C1N'],
}


qm9_second_half = {
    'name': 'QM9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'train_n_nodes': {3: 1, 4: 3, 5: 3, 6: 5, 7: 7, 8: 25, 9: 62, 10: 178, 11: 412, 12: 845, 13: 1541, 14: 2587,
                        15: 3865, 16: 5344, 17: 6461, 18: 6695, 19: 6944, 20: 4794, 21: 4962, 22: 1701, 23: 2380,
                        24: 267, 25: 754, 26: 17, 27: 132, 29: 15},
    'max_n_nodes': 29,
    'atom_fc_num': {'N1': 20738, 'N-1': 8024, 'C1': 4117, 'O-1': 192, 'C-1': 764},
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'top_bond_sym': ['C1H', 'C1C', 'C1O', 'N1C', 'N1H', 'C2O', 'O1H', 'C2C'],
    'top_angle_sym': ['C1C-C1H', 'C1C-C1C', 'C1C-C1O', 'C1C-C1N', 'C1N-N1C', 'C1O-O1C', 'O1C-C1H', 'C2C-C1C'],
    'top_dihedral_sym': ['H1C-C1C-C1C', 'C1C-C1C-C1C', 'H1C-C1C-C1H', 'H1C-C1C-C1O', 'C1C-C1C-C1O', 'C1N-N1C-C1C',
                            'H1C-C1N-N1C', 'H1C-C1C-C1N'],
    'prop2idx': {'mu': 0, 'alpha': 1, 'homo': 2, 'lumo': 3, 'gap': 4, 'Cv': 11},
}

dataset_info_dict = {
    'qm9_with_h': qm9_with_h,
    'qm9_second_half': qm9_second_half,
}


def get_dataset_info(info_name):
    return dataset_info_dict[info_name]
