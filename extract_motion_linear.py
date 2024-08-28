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


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--begin_idx', type=str, default='1490', help='begin index of segment')
    parser.add_argument('--end_idx', type=str, default='3000', help='end index of segment')
    parser.add_argument('--img_root', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN', help='root directory of images')
    parser.add_argument('--config', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN/xueshenCN.json', help='path to the config file')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model')
    parser.add_argument('--log_file', type=str, default='/home/bml/storage/mnt/v-f028498a5029402d/org/student/dataset/xueshenCN/xueshenCN_extract_motion_exception_log', help='log file')
    return parser.parse_args()



def worker_init(extractor):
    global extractor_pipeline
    extractor_pipeline = extractor

def process_task(args):
    path = str(args[0])
    out_path = str(args[1])
    extractor_pipeline.extract_motion_features(path, out_path)

def main(args):
    ######## preparations for motion extractor pipeline ########
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    motion_args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(motion_args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, motion_args.__dict__)
    crop_cfg = partial_fields(CropConfig, motion_args.__dict__)

    extractor_pipeline = ExtractMotionPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    ################################################################


    log_file_path = f"{args.log_file}_{args.begin_idx}_{args.end_idx}.txt"
    img_root = Path(args.img_root)
    with open(args.config, "r") as f:
        sequences = json.load(f)

    args_lst = []
    for seq in tqdm(sequences, desc='Preparing data'):
        path = img_root / seq / dir_name
        out_path = img_root / seq / motion_name
        args_lst.append((path, out_path))

    start = time.time()
    with open(log_file_path,'w') as log_file:
        for task_args in tqdm(args_lst[int(args.begin_idx):int(args.end_idx)]):
            try:
                extractor_pipeline.extract_motion_features(str(task_args[0]), str(task_args[1]))
            except BaseException as e:
                print(f"No face detected in {task_args[0].parent.parent.name}")
                log_file.write(f"No face detected in {task_args[0].parent.parent.name}")

    end = time.time()
    print(f"time for motion extraction:{end-start}")

    print("Finished processing all motion features.")

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
