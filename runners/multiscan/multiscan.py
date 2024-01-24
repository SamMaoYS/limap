import os
import json
import cv2
import numpy as np
import copy


class MultiScan:
    # constants
    max_image_dim = -1
    img_hw_resized = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # scene to load
        self.scene_id = None
        self.scene_dir = None
        self.stride = 1

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
        self.scene_dir = os.path.join(self.data_dir, scene_id)

    def set_stride(self, stride):
        self.stride = stride
        # load all infos here
        self.loadinfos()

    def read_intrinsics(self, fname, mode="color"):
        with open(fname, "r") as f:
            camera_data = json.load(f)
        img_hw = [480, 640]
        K = np.array(camera_data["intrinsic"]).reshape(3, 3).transpose()
        return K, img_hw

    def read_pose(self, pose_json):
        with open(pose_json, "r") as f:
            camera_data = json.load(f)

        mat = np.array(camera_data["extrinsic"]).reshape(4, 4).transpose()
        mat = np.linalg.inv(mat)
        R_cam2world, t_cam2world = mat[:3, :3], mat[:3, 3]
        R = R_cam2world.T
        t = -R @ t_cam2world
        return R, t

    def loadinfos(self):
        img_folder = os.path.join(self.scene_dir, "rgb")
        pose_folder = os.path.join(self.scene_dir, "camera")
        depth_folder = os.path.join(self.scene_dir, "depth")
        n_images = len(os.listdir(img_folder))
        index_list = os.listdir(img_folder)
        index_list = [x.split(".")[0] for x in index_list]

        # load intrinsic
        fname_meta = os.path.join(pose_folder, "{0}.json".format(index_list[0]))
        K_orig, img_hw_orig = self.read_intrinsics(fname_meta)
        h_orig, w_orig = img_hw_orig[0], img_hw_orig[1]
        # reshape w.r.t max_image_dim
        K = copy.deepcopy(K_orig)
        img_hw_orig = (h_orig, w_orig)
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
            imname = os.path.join(self.scene_dir, "rgb", "{0}.png".format(index))
            self.imname_list.append(imname)

            pose_json = os.path.join(self.scene_dir, "camera", "{0}.json".format(index))
            R, T = self.read_pose(pose_json)
            self.Rs.append(R)
            self.Ts.append(T)

    def get_depth_fname(self, imname):
        depth_folder = os.path.join(self.scene_dir, "depth")
        img_id = os.path.basename(imname)[:-4]
        depth_fname = os.path.join(depth_folder, "{0}.png".format(img_id))
        return depth_fname

    def get_depth(self, imname):
        depth_fname = self.get_depth_fname(imname)
        depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0
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
