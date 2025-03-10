import torch
from torch_geometric.data import Batch, Data
import os
from ema_pytorch import EMA
import numpy as np
from RIdiffusion import HEGNN, RIdiffusion
from generate_graph_ss import prepare_graph, pdb2graph
import argparse

parser = argparse.ArgumentParser()

#parser.add_argument("--times", default=1)

args = parser.parse_args()
nt_types = ['A', 'U', 'G', 'C']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

ckpt = torch.load('./weight/weight.pt', map_location=device)

config = ckpt['config']
config['noise_type'] = 'uniform'

gnn = HEGNN(config, input_feat_dim=config['input_feat_dim'], hidden_channels=config['hidden_dim'], edge_attr_dim=config['edge_attr_dim'], dropout=config['drop_out'], n_layers=config['depth'], update_edge=config['update_edge'], embedding=config['embedding'], embedding_dim=config['embedding_dim'], embed_ss=config['embed_ss'], norm_feat=config['norm_feat'])

diffusion = RIdiffusion(model=gnn, config=config, timesteps=config['timesteps']).to(device)
diffusion = EMA(diffusion)
diffusion.load_state_dict(ckpt['ema'])
diffusion = diffusion.to(device)

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

pdb_dir = './pdb'
pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
output_file = './pdb.fasta'
error_files = []

with open(output_file, 'w') as file:
    for pdb_file in pdb_files:
        # try:
        graph = pdb2graph(pdb_file,if_transform=False)
        input_graph = Batch.from_data_list([prepare_graph(graph)]).to(device)
        original_sequence = ''.join([nt_types[idx.item()] for idx in input_graph.x.argmax(dim=1)])

        pdb_filename = os.path.splitext(os.path.basename(pdb_file))[0]
        file.write(f'>Original_{pdb_filename}\n')
        file.write(f'{original_sequence}\n')
        
        print(f'>Original_{pdb_filename}')
        print(original_sequence)
            
            
        for i in range(10):
            prob, sample_graph = diffusion.ema_model.ddim_sample(input_graph)
            sampled_sequence = ''.join([nt_types[idx.item()] for idx in sample_graph.argmax(dim=1)])
            recovery = (prob.argmax(dim=1) == input_graph.x.argmax(dim=1)).sum().item() / input_graph.x.shape[0]

            print(f'Iteration {i+1}: Sampled: {sampled_sequence}')
            print(f'Iteration {i+1}: Original: {original_sequence}')
            print(f'Iteration {i+1}: Recovery: {recovery}\n')

            file.write(f'>seq{i+1}_{pdb_filename}--{recovery}\n')
            file.write(f'{sampled_sequence}\n')
            
        
        # except Exception as e:
        #     print(f"Error processing {pdb_file}: {e}")
        #     error_files.append(pdb_file)

# if error_files:
#     error_log_file = './error_log.txt'
#     with open(error_log_file, 'w') as ef:
#         for error_file in error_files:
#             ef.write(f"{error_file}\n")
#     print(f"Errors occurred in {len(error_files)} files. Check {error_log_file} for details.")
