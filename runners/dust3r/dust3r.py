import os
import json
import cv2
import torch
import numpy as np
import copy


class Dust3R:
    # constants
    max_image_dim = -1
    img_hw_resized = None

    def __init__(self, result_path):
        self.result_path = result_path

        # scene to load
        self.scene_id = None
        self.scene_dir = None
        self.stride = 1
        self.dust3r_data = None

        # initialize empty infos
        self.imname_list = []
        self.K, self.img_hw = None, None
        self.Rs, self.Ts = [], []

    @classmethod
    def set_max_dim(cls, max_image_dim):
        cls.max_image_dim = max_image_dim

    @classmethod
    def set_img_hw_resized(cls, img_hw_resized):
        cls.img_hw_resized = img_hw_resized

    def set_scene_id(self, scene_id):
        self.scene_id = scene_id

    def set_scene_dir(self, scene_dir):
        self.scene_dir = scene_dir

    def set_stride(self, stride):
        self.stride = stride
        # load all infos here
        self.loadinfos()

    def read_pose(self, mat):
        R_cam2world, t_cam2world = mat[:3, :3], mat[:3, 3]
        R = R_cam2world.T
        t = -R @ t_cam2world
        return R, t

    def loadinfos(self):
        self.dust3r_data = torch.load(self.result_path, weights_only=False)
        K_b33 = self.dust3r_data["K_b33"]
        world_T_cam_b44 = self.dust3r_data["world_T_cam_b44"]
        rgb_hw3_list = self.dust3r_data["rgb_hw3_list"]
        # depth_hw_list = self.dust3r_data.depth_hw_list
        # conf_hw_list = self.dust3r_data.conf_hw_list
        masks_list = self.dust3r_data["masks_list"]

        for i in range(len(rgb_hw3_list)):
            mask_hw = masks_list[i]
            self.dust3r_data["depth_hw_list"][i][mask_hw == 0] = 0

        n_images = len(rgb_hw3_list)
        index_list = list(range(n_images))

        # load intrinsic
        K_orig = K_b33[0]
        img_hw_orig = rgb_hw3_list[0].shape[:2]
        h_orig, w_orig = img_hw_orig[0], img_hw_orig[1]
        # reshape w.r.t max_image_dim
        K = copy.deepcopy(K_orig)
        img_hw = (h_orig, w_orig)
        max_image_dim = self.max_image_dim
        if (max_image_dim is not None) and max_image_dim != -1:
            ratio = max_image_dim / max(h_orig, w_orig)
            if ratio < 1.0:
                h_new = int(round(h_orig * ratio))
                w_new = int(round(w_orig * ratio))
                K[0, :] = K[0, :] * w_new / w_orig
                K[1, :] = K[1, :] * h_new / h_orig
                img_hw = (h_new, w_new)
        if self.img_hw_resized is not None:
            h_new, w_new = self.img_hw_resized[0], self.img_hw_resized[1]
            K[0, :] = K[0, :] * w_new / w_orig
            K[1, :] = K[1, :] * h_new / h_orig
            img_hw = (h_new, w_new)
        self.K, self.img_hw = K, img_hw

        # get imname_list and cameras
        self.imname_list, self.Rs, self.Ts = [], [], []
        for index in index_list:
            rgb_hw3 = rgb_hw3_list[index] * 255
            rgb_hw3 = rgb_hw3.astype(np.uint8)
            rgb_hw3 = cv2.cvtColor(rgb_hw3, cv2.COLOR_RGB2BGR)
            os.makedirs(self.scene_dir, exist_ok=True)
            imname = os.path.join(self.scene_dir, "{0}.png".format(index))
            cv2.imwrite(imname, rgb_hw3)
            self.imname_list.append(imname)

            pose = world_T_cam_b44[index]
            R, T = self.read_pose(pose)
            self.Rs.append(R)
            self.Ts.append(T)

    def get_depth_fname(self, imname):
        os.makedirs(self.scene_dir, exist_ok=True)
        img_id = os.path.basename(imname)[:-4]
        depth_fname = os.path.join(self.scene_dir, "{0}.npy".format(img_id))
        index = int(img_id)
        np.save(depth_fname, self.dust3r_data["depth_hw_list"][index])
        return depth_fname

    def get_depth(self, imname):
        depth_fname = int(imname)
        depth = self.dust3r_data["depth_hw_list"][depth_fname]
        depth = depth.astype(np.float32)
        return depth

    def get_img_hw(self):
        if self.img_hw is None:
            self.loadinfos()
        return self.img_hw

    def load_intrinsics(self):
        if self.K is None:
            self.loadinfos()
        return self.K

    def load_imname_list(self):
        if len(self.imname_list) == 0:
            self.loadinfos()
        return self.imname_list

    def load_cameras(self):
        if len(self.Rs) == 0:
            self.loadinfos()
        return self.Ts, self.Rs
