import os
import sys
first_path = sys.path[0]
parent_path = os.path.dirname(first_path)
sys.path.insert(0, parent_path)
from model.hegnn.hegnn_model import EGNN, EGNN_Network
from model.hegnn.layers import EGNN_Sparse, EGNN_Sparse_Network
