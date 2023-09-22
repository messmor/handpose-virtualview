import os
import sys

import numpy as np

sys.path.append("/home/inseer/engineering/handpose-virtualview")
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.multiview_a2j import MultiviewA2J
from utils.parser_utils import get_a2j_parser
from utils.hand_detector import crop_area_3d, calculate_com_2d
from ops import point_transform as np_pt
from ops.render import depth_crop_expand
mpl.use('TkAgg')
class Hand_Model_3d(object):

    def __init__(self, args):
        self.args = args
        self.dataset_config = json.load(open("config/dataset/nyu.json", 'r'))
        self.fx = self.dataset_config['camera']['fx']
        self.fy = self.dataset_config['camera']['fy']
        self.u0 = self.dataset_config['camera']['u0']
        self.v0 = self.dataset_config['camera']['v0']
        self.num_joints = len(self.dataset_config['selected'])
        self.pre_model_name = "/home/inseer/engineering/handpose-virtualview/checkpoint/nyu/25select15.pth"
        self.model = self.load_model()
        self.cube = np.expand_dims(np.asarray([280, 280, 280]), axis=0)
        self.crop_size = 176
        self.com_2d = None
        self.level = 3
        self.joint_3d = None


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
        cropped, crop_trans, com_2d = crop_area_3d(depth_map, com_2d, self.fx, self.fy, size=np.squeeze(self.cube),
                                                   dsize=[self.crop_size, self.crop_size], docom=False)
        cropped = cropped[None, None]
        crop_trans = crop_trans[None]
        com_2d = com_2d[None]

        crop_expand, anchor_joints_2d_crop, regression_joints_2d_crop, depth_value_norm, joints_3d, view_trans, \
            joint_3d_fused, classification, regression, depthregression, conf, joint_3d_conf =\
            self.model(cropped, crop_trans, com_2d, self.cube, self.level)
        self.joint_3d = joint_3d_fused.detach().cpu().numpy()
        self.joint_3d = np.squeeze(self.joint_3d)
        print("inference pass finished!")

        return joints_3d, joint_3d_fused

    def visualize_3d_joints(self, depth_map):
        print("start visualization!")
        com_2d = calculate_com_2d(depth_map)
        print("com_2d calculated!")
        cropped, crop_trans, com_2d = crop_area_3d(depth_map, com_2d, self.fx, self.fy, size=[280, 280, 280],
                                                   dsize=[self.crop_size, self.crop_size], docom=False)
        print("crop_area_3d completed!")
        cropped, crop_trans, com_2d = torch.from_numpy(cropped).to("cuda"), torch.from_numpy(crop_trans).to("cuda"), torch.from_numpy(com_2d).to("cuda")
        cropped = cropped[None, None]
        crop_trans = crop_trans[None]
        com_2d = com_2d[None]

        crop_expand, view_mat = depth_crop_expand(depth_crop=cropped,
                                                  fx=self.fx,
                                                  fy=self.fy,
                                                  u0=self.u0,
                                                  v0=self.v0,
                                                  crop_trans=crop_trans,
                                                  level=1,
                                                  com_2d=com_2d,
                                                  random_sample=False
                                                  )
        print("depth_crop_expand completed!")
        # take off cuda
        cropped = cropped.cpu().numpy()
        print("cropped_shape", cropped.shape)
        crop_trans = crop_trans.cpu().numpy()
        com_2d = com_2d.cpu().numpy()
        crop_expand = crop_expand.cpu().numpy()
        view_mat = view_mat.cpu().numpy()

        # plt.imshow(cropped[0, 0, ...])
        # plt.show()
        print(crop_expand.shape)
        cube = np.squeeze(self.cube)
        com_2d = np.squeeze(com_2d)
        #
        for i in range(0, crop_expand.shape[1], 1):
            print(f"iteration {i}")
            img = cropped[0, 0]
            img[img>1e-3] = img[img>1e-3] - com_2d[2] + cube[2]/2.
            img[img<1e-3] = cube[2]
            img = img / cube[2]
            _joint_3d = self.joint_3d
            _joint_3d = np_pt.transform_3D(_joint_3d, view_mat[0, i])
            _joint_2d = np_pt.transform_3D_to_2D(_joint_3d, self.fx, self.fy, self.u0, self.v0)
            _crop_joint_2d = np_pt.transform_2D(_joint_2d, crop_trans[0])
            fig, ax = plt.subplots(figsize=plt.figaspect(img))
            fig.subplots_adjust(0, 0, 1, 1)
            ax.imshow(img, cmap='gray')
            ax.scatter(_crop_joint_2d[:, 0], _crop_joint_2d[:, 1], c='red', s=100)
            ax.axis('off')
            # plt.savefig('{}.jpg'.format(i))
            plt.show()


def plot_fused_joints(joints_3d):
    joints_3d = joints_3d.to('cpu').detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # connections = [[0, 11], [12, 1], [1, 2], [11, 3], [3, 4], [11, 5], [5, 6], [11, 7], [7, 8], [11, 9], [9, 10], [13, 11]]
    config = json.load(open("config/dataset/nyu.json", 'r'))
    connections = config["connections"]
    colors = config["connection_colors"]
    # Extract x, y, and z coordinates from joint_locations
    x_coords, y_coords, z_coords = zip(*joints_3d)

    # Create scatter plot
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')
    for c_i, con in enumerate(connections):
        ax.plot([x_coords[con[0]], x_coords[con[1]]], [y_coords[con[0]], y_coords[con[1]]], [z_coords[con[0]], z_coords[con[1]]], c=colors[c_i])

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot title
    ax.set_title('3D Joint Locations')

    # Show the plot
    plt.show()





if __name__ == "__main__":
    import cv2
    from pathlib import Path
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
    # depth_map_path = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_depth_maps/frame_120_person_0_right.jpg"
    depth_dir = "/home/inseer/data/Hand_Testing/Orientation/Front_Flexion_And_Extension/hand_depth_maps/"
    for depth_map_path in Path(depth_dir).iterdir():
        depth_map = cv2.imread(depth_map_path.as_posix())
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        plt.imshow(depth_map)
        plt.show()


        joints_3d, joint_3d_fused = h_model.inference(depth_map)
        joint_3d_fused = joint_3d_fused[0]
        h_model.visualize_3d_joints(depth_map)



