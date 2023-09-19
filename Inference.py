import os
import sys

import numpy as np

sys.path.append("/home/inseer/engineering/handpose-virtualview")
import json
import torch
import torch.nn as nn

from models.multiview_a2j import MultiviewA2J
from utils.parser_utils import get_a2j_parser
from utils.hand_detector import crop_area_3d

class Hand_Model_3d(object):

    def __init__(self, args):
        self.args = args
        self.dataset_config = json.load(open("config/dataset/nyu.json", 'r'))
        self.fx = self.dataset_config['camera']['fx']
        self.fy = self.dataset_config['camera']['fy']
        self.num_joints = len(self.dataset_config['selected'])
        self.pre_model_name = "/home/inseer/engineering/handpose-virtualview/checkpoint/nyu/25select15.pth"
        self.model = self.load_model()
        self.cube = [280, 280, 280]
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
        # cropped, crop_trans, com_2d = crop_area_3d(depth_map, com_2d, self.fx, self.fy, size=self.cube,
        #                                            dsize=[self.crop_size, self.crop_size], docom=False)
        cropped = depth_map.clone()
        crop_trans = torch.zeros((1, 3, 3))
        com_2d = None
        crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d, view_trans, \
            joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf =\
            self.model(cropped, crop_trans, com_2d, self.cube, self.level)

        return joints_3d, joint_3d_fused


def convert_coco_2_numpy(annotation_path):
    """converts coco cvat keypoints annotations into a numpy of the shape
       (frames, joints, 2) note 2d coordinates are in form [x,y,vis]
    """
    with open(annotation_path, "r") as file:
        ann_dict = json.load(file)

    np_data = []
    annotations = ann_dict["annotations"]
    for frame_data in annotations:
        kps = frame_data['keypoints']
        if kps:
            np_data.append(kps)

    np_data = np.asarray(np_data)
    np_data = np.reshape(np_data, newshape=(np_data.shape[0], np_data.shape[1] // 3, 3))

    return np_data[..., 0:2]



if __name__ == "__main__":
    import cv2
    from utils.parser_utils import get_a2j_parser
    parser = get_a2j_parser()
    args = parser.parse_args()
    h_model = Hand_Model_3d(args)

    # load depth map
    json_coords = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/pixel_coords.json"
    kps = convert_coco_2_numpy(json_coords)
    thumb = kps[1, 104, 25]
    knuckle = kps[1, 104, 26]
    # TODO get hand kps in cropped coordinate system. Use these to compute com_2d
    depth_map_path = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_depth_maps/frame_105_person_0_right.jpg"
    depth_map = cv2.imread(depth_map_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    depth_map = depth_map[None, None]
    depth_map = torch.from_numpy(depth_map)
    print("depth map shape", depth_map.shape)

    joints_3d, joints_3d_fused = h_model.inference(depth_map)

