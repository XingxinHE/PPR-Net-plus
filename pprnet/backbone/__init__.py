import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
from pointnet2 import pointnet2_part_seg_msg
from pointnet2.pointnet2_part_seg_msg import Pointnet2MSGBackbone

#from rscnn.rscnn_backbone import RSCNNBackbone
