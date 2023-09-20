import os
import sys

import numpy as np

sys.path.append("/home/inseer/engineering/handpose-virtualview")
import json
import torch
import torch.nn as nn

from models.multiview_a2j import MultiviewA2J
from utils.parser_utils import get_a2j_parser
from utils.hand_detector import crop_area_3d, calculate_com_2d

class Hand_Model_3d(object):

    def __init__(self, args):
        self.args = args
        self.dataset_config = json.load(open("config/dataset/nyu.json", 'r'))
        self.fx = self.dataset_config['camera']['fx']
        self.fy = self.dataset_config['camera']['fy']
        self.num_joints = len(self.dataset_config['selected'])
        self.pre_model_name = "/home/inseer/engineering/handpose-virtualview/checkpoint/nyu/25select15.pth"
        self.model = self.load_model()
        self.cube = np.expand_dims(np.asarray([280, 280, 280]), axis=0)
        self.crop_size = 176
        self.com_2d = None
        self.level = 3


    def load_model(self):
        model = MultiviewA2J(self.dataset_config['camera'], self.num_joints, self.args.n_head, self.args.d_attn,
                             self.args.d_k, self.args.d_v, self.args.d_inner, self.args.dropout_rate,
                             self.args.num_select, use_conf=self.args.use_conf, random_select=self.args.random_select,
                             random_sample=self.args.random_sample)

        self.checkpoint = torch.load(self.pre_model_name)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model = nn.DataParallel(model).cuda()

        return model

    def inference(self, depth_map):
        com_2d = calculate_com_2d(depth_map)
        cropped, crop_trans, com_2d = crop_area_3d(depth_map, com_2d, self.fx, self.fy, size=[280, 280, 280],
                                                   dsize=[self.crop_size, self.crop_size], docom=False)
        cropped = cropped[None, None]
        crop_trans = crop_trans[None]
        com_2d = com_2d[None]

        crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d, view_trans, \
            joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf =\
            self.model(cropped, crop_trans, com_2d, self.cube, self.level)

        return joints_3d, joint_3d_fused



if __name__ == "__main__":
    import cv2
    from utils.parser_utils import get_a2j_parser
    parser = get_a2j_parser()
    args = parser.parse_args()
    h_model = Hand_Model_3d(args)
    # load 2d hand kps
    # kps_path = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_kps_cropped.npy"
    # kps = np.load(kps_path, allow_pickle=True)
    # kps = [item[1] for item in kps if item[0] == 105]
    # kps = np.squeeze(kps[0])
    # com_2d = np.mean(kps, axis=0)
    # print("com_2d", com_2d)
    # load depth map
    depth_map_path = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_depth_maps/frame_105_person_0_right.jpg"
    depth_map = cv2.imread(depth_map_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    print("depth map shape", depth_map.shape)

    joints_3d, joint_3d_fused = h_model.inference(depth_map)

    print("joints_3d shape: ", joints_3d.shape)
    print("joint_3d_fused:", joint_3d_fused.shape)
    print("Fused 3d joints")
    joint_3d_fused = joint_3d_fused[0]
    for i in range(len(joint_3d_fused)):
        print(f"joint {i}: {joint_3d_fused[i]}")


