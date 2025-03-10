import os
from dataset_src.RNA_graph_generate_ss import RNA_imem, dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from dataset_src.utils import NormalizeRNA, get_stat
from Bio.PDB import PDBParser
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

amino_acids_type = ['A', 'U', 'G', 'C']

def get_struc2ndRes(pdb_filename):
    integer_encoded = []
    struc_2nds_res_alphabet = ['S', 'M', 'I', 'B', 'H', 'K', 'E', 'X']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))
    p = PDBParser()
    structure = p.get_structure('random_id', pdb_filename)
    model = structure[0]
    model_residues = [(chain.id, residue.id[1]) for chain in model for residue in chain if residue.id[0] == ' ']
    one_hot_list = torch.zeros(len(model_residues), len(struc_2nds_res_alphabet))
    # st_file_name = pdb_filename.replace(".pdb", ".st").replace(f'{key}', f"{key}_ss")
    st_file_name = pdb_filename.replace(".pdb", ".st")
    st_file = st_file_name
    if not os.path.exists(st_file):
        return one_hot_list 
    else:
        with open(st_file, 'r') as f:
            counter = 0
            for line in f:
                if counter != 5:
                    counter += 1
                    continue
                else:
                    counter += 1
                    ss = line.strip()
                    current_position = 0
                    for cha in ss:
                        integer_encoded.append(char_to_int[cha])
                        one_hot = F.one_hot(torch.tensor(integer_encoded[-1]), num_classes=8)
                        one_hot_list[current_position] = one_hot
                        current_position += 1
        return one_hot_list

def prepare_graph(data):
    del data['distances']
    del data['edge_dist']
    mu_r_norm = data.mu_r_norm

    extra_x_feature = torch.cat([data.x[:, 4:], mu_r_norm], dim=1)
    graph = Data(
        x=data.x[:, :4],
        extra_x=extra_x_feature,
        pos=data.pos,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        ss=data.ss[:data.x.shape[0], :],
        sasa=data.x[:, 4]
    )
    return graph

def pdb2graph(filename, normalize_path='dataset_src/mean_attr.pt', if_transform=False):
    #### dataset  ####
    dataset_arg = dataset_argument(n=666)

    dataset = RNA_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                                divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                                c_4_max_neighbors=dataset_arg['c_4_max_neighbors'],
                                set_length=dataset_arg['set_length'],
                                struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
                                random_sampling=True,diffusion=True)

    rec, rec_coords, c_4_coords, p_coords, n_coords = dataset.get_receptor_inference(filename)

    struc_2nd_res = get_struc2ndRes(filename)
    rec_graph = dataset.get_c4prime_graph(
                rec, c_4_coords, p_coords, n_coords, rec_coords, struc_2nd_res)
    if rec_graph:
        if if_transform:
            normalize_transform = NormalizeRNA(filename=normalize_path)
            graph = normalize_transform(rec_graph)
        else:
            graph = rec_graph
        return graph
    else:
        return None


def get_graph(filename):
    if os.path.exists(save_dir + filename.replace('.pdb', '.pt')):
        pass
    else:
        try:
            if filename in exclude_ls:
                print(filename + "  excluded")
            else:
                graph = pdb2graph(pdb_dir + filename, mean_dir)
                if graph:
                    torch.save(graph, save_dir + filename.replace('.pdb', '.pt'))
                    torch.save(graph, all_dir + filename.replace('.pdb', '.pt'))
                else:
                    error_pdb.append(filename)
                    print("err")
        except (IndexError, KeyError, ValueError, torch.serialization.pickle.UnpicklingError) as e:
            error_pdb.append(filename)
            print(f"err: {filename} - {e}")
            
if __name__ == '__main__':
    out_dir = 'graph_dataset'
    data_set_id = "dataset_0.8"
    all_dir = f'graph_dataset/{data_set_id}/all/'
    mean_dir = f'graph_dataset/{data_set_id}/mean_attr.pt'
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)    
    error_pdb = []
    for key in ['test_0.8', 'validation_0.8', 'train_0.8']:
        pdb_dir = f'dataset_src/{key}/'
        save_dir = f'{out_dir}/{data_set_id}/{key}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename_list = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]

        for filename in tqdm(filename_list):
            if os.path.exists(save_dir + filename.replace('.pdb', '.pt')):
                pass
            else:
                try:
                    graph = pdb2graph(pdb_dir + filename)
                    if graph:
                        torch.save(graph, save_dir + filename.replace('.pdb', '.pt'))
                        torch.save(graph, all_dir + filename.replace('.pdb', '.pt'))
                except:
                    pass