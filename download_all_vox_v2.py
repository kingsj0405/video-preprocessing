"""
Edit history

- Seoung Wug Oh (https://github.com/seoungwugoh), 2022.10.17, Initial commit
- Sejong Yang (https://yangspace.co.kr/), 2022.10.17, Implement audio, landmark extraction
"""
import numpy as np
import pandas as pd
import imageio
import os
import subprocess
import warnings
import glob
import time
from util import bb_intersection_over_union, join, scheduler, crop_bbox_from_frames, save, compute_increased_bbox
from argparse import ArgumentParser
from skimage.transform import resize
import cv2

from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm

warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
REF_FRAME_SIZE = 360
REF_FPS = 25


def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")

    if not os.path.exists(video_path):
        down_video = " ".join([
            "yt-dlp",
            '-f', "'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio'",
            '--skip-unavailable-fragments',
            '--merge-output-format', 'mp4',
            "https://www.youtube.com/watch?v=" + video_id, "--output",
            video_path, "--external-downloader", "aria2c",
            "--external-downloader-args", '"-x 16 -k 1M"',
            '--quiet',
        ])
        # print(down_video)
        status = os.system(down_video)


def secs_to_timestr(secs):
    hrs = secs // (60 * 60)
    # thanks @LeeDongYeun for finding & fixing this bug
    min = (secs - hrs * 3600) // 60
    sec = secs % 60
    end = (secs - int(secs)) * 100
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min),
                                                int(sec), int(end))


def split_in_utterance(person_id, video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")

    if not os.path.exists(video_path):
        print("No video file %s found, probably broken link" % video_id)
        return []

    # get video info
    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid.release()

    utterance_folder = os.path.join(
        args.annotations_folder, person_id, video_id)
    utterance_files = sorted(os.listdir(utterance_folder))
    utterances = [pd.read_csv(os.path.join(
        utterance_folder, f), sep='\t', skiprows=6) for f in utterance_files]

    for i, utterance in enumerate(utterances):
        first_frame, last_frame = utterance['FRAME '].iloc[0], utterance['FRAME '].iloc[-1]
        start_sec = first_frame / float(REF_FPS)
        end_sec = last_frame / float(REF_FPS)

        # get biggest box
        left = np.array(utterance['X ']).min() * width
        top = np.array(utterance['Y ']).min() * height
        right = left + np.array(utterance['W ']).max() * width
        bot = top + np.array(utterance['H ']).max() * height

        left, top, right, bot = compute_increased_bbox(
            [left, top, right, bot], increase_area=args.increase)
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, width-1)
        bot = min(bot, height-1)

        right = right - (right-left) % 16
        bot = bot - (bot-top) % 16

        chunk_name = os.path.join(
            args.chunk_folder, f'{person_id}_{video_id}_{i}.mp4')
        cut_video = f'ffmpeg -i {video_path} -crf 10 -r 25 -vf "crop={int(right-left)}:{int(bot-top)}:{int(left)}:{int(top)}" -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} {chunk_name} -loglevel error -n'

        # "crop=out_w:out_h:x:y" -loglevel error -vcodec libx264 -crf 10  -pix_fmt yuv420p

        # print(cut_video)
        os.system(cut_video)


def run(params):
    person_id, args = params

    video_folder = os.path.join(args.annotations_folder, person_id)

    for video_id in os.listdir(video_folder):
        intermediate_files = []
        try:
            download(video_id, args)

            split_in_utterance(person_id, video_id, args)

        except Exception as e:
            print(e)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--increase", default=0.3, type=float,
                        help='Increase bbox by this amount')
    parser.add_argument("--annotations_folder", default='./vox2_txt',
                        help='Path to utterance annotations')
    parser.add_argument("--video_folder", default='./vox2_raw',
                        help='Path to intermediate videos')
    parser.add_argument("--chunk_folder", default='./vox2',
                        help="Path to folder with video chunks")
    parser.add_argument("--workers", default=1, type=int,
                        help='Number of parallel workers')
    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.chunk_folder):
        os.makedirs(args.chunk_folder)

    person_ids = sorted(set(os.listdir(args.annotations_folder)))
    args_list = cycle([args])

    pool = Pool(processes=args.workers)

    for chunks_data in tqdm(pool.imap_unordered(run, zip(person_ids, args_list))):
        pass


#     for i, pid in enumerate(person_ids):
#         run([pid, args])

#         if i == 2:
#             break

    # scheduler(ids, run, args)
