import os, sys
import numpy as np
import cv2

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.base as _base


class Dust3rDepthReader(_base.BaseDepthReader):
    def __init__(self, filename):
        super(Dust3rDepthReader, self).__init__(filename)

    def read(self, filename):
        ref_depth = np.load(filename)
        ref_depth = ref_depth.astype(np.float32)
        return ref_depth


def read_scene_dust3r(cfg, dataset, scene_id, load_depth=False):
    # set scene id
    dataset.set_scene_id(scene_id)
    scene_dir = os.path.join(os.path.dirname(dataset.result_path), scene_id)
    dataset.set_scene_dir(scene_dir)
    dataset.set_max_dim(cfg["max_image_dim"])

    # get imname_list and cameras
    dataset.set_stride(cfg["stride"])
    imname_list = dataset.load_imname_list()
    K = dataset.load_intrinsics()
    img_hw = dataset.get_img_hw()
    Ts, Rs = dataset.load_cameras()
    cameras = [_base.Camera("PINHOLE", K, cam_id=0, hw=img_hw)]
    camimages = [
        _base.CameraImage(
            0, _base.CameraPose(Rs[idx], Ts[idx]), image_name=imname_list[idx]
        )
        for idx in range(len(imname_list))
    ]
    imagecols = _base.ImageCollection(cameras, camimages)

    # TODO: advanced implementation with the original ids
    # trivial neighbors
    index_list = np.arange(0, len(imname_list)).tolist()
    neighbors = {}
    for idx, image_id in enumerate(index_list):
        val = np.abs(np.array(index_list) - image_id)
        val[idx] = val.max() + 1
        neighbor = np.array(index_list)[np.argsort(val)[: cfg["n_neighbors"]]]
        neighbors[image_id] = neighbor.tolist()

    # get depth
    if load_depth:
        depths = {}
        for img_id, imname in enumerate(imname_list):
            depth_fname = dataset.get_depth_fname(imname)
            depth = Dust3rDepthReader(depth_fname)
            depths[img_id] = depth
        return imagecols, neighbors, depths
    else:
        return imagecols, neighbors
