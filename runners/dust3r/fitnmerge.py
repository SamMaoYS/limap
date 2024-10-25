import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dust3r import Dust3R
from loader import read_scene_dust3r

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import limap.util.config as cfgutils
import limap.runners


def run_scene_dust3r(cfg, dataset, scene_id):
    imagecols, neighbors, depths = read_scene_dust3r(
        cfg, dataset, scene_id, load_depth=True
    )
    linetracks = limap.runners.line_fitnmerge(
        cfg, imagecols, depths, neighbors=neighbors
    )
    return linetracks


def parse_config():
    import argparse

    arg_parser = argparse.ArgumentParser(description="fitnmerge 3d lines")
    arg_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="cfgs/fitnmerge/dust3r.yaml",
        help="config file",
    )
    arg_parser.add_argument(
        "--default_config_file",
        type=str,
        default="cfgs/fitnmerge/default.yaml",
        help="default config file",
    )

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    shortcuts["-nn"] = "--n_neighbors"
    shortcuts["-sid"] = "--scene_id"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = os.path.join("precomputed", "dust3r", cfg["scene_id"])
    return cfg


def main():
    cfg = parse_config()
    dataset = Dust3R(cfg["result_path"])
    run_scene_dust3r(cfg, dataset, cfg["scene_id"])


if __name__ == "__main__":
    main()
