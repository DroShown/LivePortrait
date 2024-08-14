import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from multi_threads import JobManager



def parse_args():
    parser = argparse.ArgumentParser(description='Draw sketch')
    parser.add_argument('--img_root', type=str, default='/home/yifei/xueshenCN_toy', help='root directory of images')
    parser.add_argument('--config', type=str, default='/home/yifei/xueshenCN_toy/xueshenCN_toy.json', help='path to the config file')
    parser.add_argument('--visualize', action='store_true', help='visualize the landmarks')
    parser.add_argument('--device', type=str, default='cpu', help='device to run the model')

    return parser.parse_args()

def draw_sketch_thread(lmk_paths):
    preprocessor = Preprocessor()
    for lmk_path in tqdm(lmk_paths, desc="draw sketch"):
        lmk_npy = np.load(lmk_path)
        _, ref_sketch =  preprocessor.post_process_landmarks(lmk_npy, target_size=512)
        align_sketch_path = str(lmk_path).replace('aligner_landmarks', 'aligner_sketch').replace('.npy', '.jpg')
        Path(align_sketch_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(align_sketch_path, ref_sketch)

    return True


def main(args):
    num_workers = 32

    img_root = Path(args.img_root)
    with open(args.config, "r") as f:
        sequences = json.load(f)

    print(f"sequences: {len(sequences)}")

    # preprocessors = [Preprocessor() for _ in range(num_workers)]

    ori_align_lmk_paths = []
    for seq in tqdm(sequences, desc='Preparing data'):
        align_lmk_root_seq = img_root / seq / 'aligner_landmarks'
        ori_align_lmk_paths.extend(list(align_lmk_root_seq.resolve().glob('*.npy')))



    init_ctx_func=None
    multithread=True
    queue = -1

    manager = JobManager(num_workers, init_ctx_func, multithread, queue)

    job_infos = []
    num_frames = len(ori_align_lmk_paths)
    frames_per_thread = num_frames // num_workers
    print("frames_per_thread: ", frames_per_thread)
    extra_frames = num_frames % num_workers

    start_index = 0
    for i in tqdm(range(num_workers), desc='Preparing jobs'):
        lmk_paths = []
        end_index = start_index + frames_per_thread + (1 if i < extra_frames else 0)
        for p_idx in range(start_index, end_index):
            lmk_path = ori_align_lmk_paths[p_idx]
            lmk_paths.append(lmk_path)

        job_infos.append([lmk_paths])
        start_index = end_index

    manager.add_jobs(draw_sketch_thread, job_infos)

    ordered=True
    return_flags = []
    for job_i, res in manager.get_results(ordered):
        return_flags.append(res)
    print(return_flags)

    manager.close()



if __name__ == "__main__":
    args = parse_args()
    main(args)
