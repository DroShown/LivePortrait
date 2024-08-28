import cv2
from pathlib import Path
import json
import numpy as np
import argparse
import time
from tqdm import tqdm
import torch

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.motion_extractor_pipeline import ExtractMotionPipeline
from inference import partial_fields, fast_check_ffmpeg, fast_check_args

dir_name = "crop_output/tracked_crop"
motion_name = "motion_feature.pkl"
motion_json_name = "xueshenCN_motion.json"


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--begin_idx', type=str, default='1490', help='begin index of segment')
    parser.add_argument('--end_idx', type=str, default='3000', help='end index of segment')
    parser.add_argument('--img_root', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN', help='root directory of images')
    parser.add_argument('--config', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN/xueshenCN.json', help='path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model')
    parser.add_argument('--log_file', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN/xueshenCN_extract_motion_exception_log', help='log file')
    return parser.parse_args()

def main(args):
    img_root = Path(args.img_root)
    with open(args.config, "r") as f:
        sequences = json.load(f)

    target_seg_lst =[]
    for seq in tqdm(sequences, desc='Preparing data'):
        path = str(img_root / seq / motion_name)
        if os.path.exists(path):
            target_seg_lst.append(seq)
        else:
            print(seq)

    motion_json_path = img_root / motion_json_name
    with open(str(motion_json_path),'w') as motion_json:
        json.dump(target_seg_lst,motion_json)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
