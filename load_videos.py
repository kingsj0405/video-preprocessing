import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from util import save
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
import torch
import torchaudio
from typing import Tuple

warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
DEVNULL = open('debug.log', 'w')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

def get_mel_spectrogram(audio_path: str, fps: float) -> Tuple[torch.tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    hop_length = int(sample_rate / fps)
    n_fft = hop_length * 2
    get_specs = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        f_max=16000,
        hop_length=hop_length,
        n_fft=n_fft
    )
    mel_spec = get_specs(waveform)
    return mel_spec, sample_rate

def run(data):
    video_id, args = data
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
        download(video_id.split('#')[0], args)

    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
        print('Can not load video %s, broken link' % video_id.split('#')[0])
        return
    raw_path = os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')
    reader = imageio.get_reader(raw_path)
    fps = reader.get_meta_data()['fps']

    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]

    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[]} for j in range(df.shape[0])]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]
    try:
        for i, frame in enumerate(reader):
            for entry in all_chunks_dict:
                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    crop = frame[top:bot, left:right]
                    if args.image_shape is not None:
                        crop = img_as_ubyte(
                            resize(crop, args.image_shape, anti_aliasing=True))
                    entry['frames'].append(crop)
    except imageio.core.format.CannotReadFrameError:
        None

    for entry in all_chunks_dict:
        if 'person_id' in df:
            first_part = df['person_id'].iloc[0] + "#"
        else:
            first_part = ""
        first_part = first_part + '#'.join(video_id.split('#')[::-1])
        path = first_part + '#' + \
            str(entry['start']).zfill(6) + '#' + \
            str(entry['end']).zfill(6) + '.mp4'
        mp4_path = os.path.join(args.out_folder, partition, path)
        save(mp4_path, entry['frames'], args.format)
        if args.audio:
            start = entry["start"] / ref_fps
            frame_cnt = entry["end"] - entry["start"]
            duration = frame_cnt / ref_fps
            # mp3
            path = first_part + '#' + \
                str(entry['start']).zfill(6) + '#' + \
                str(entry['end']).zfill(6) + '.mp3'
            mp3_path = os.path.join(args.out_folder, partition, path)
            command = f'ffmpeg -ss {start} -t {duration} -i {raw_path} -q:a 0 -map a {mp3_path}'
            subprocess.call(command.split(' '), stdout=DEVNULL, stderr=DEVNULL)
            # melspectogram
            spec, sr = get_mel_spectrogram(mp3_path, fps)
            spec = spec[0, :, :frame_cnt]
            path = first_part + '#' + \
                str(entry['start']).zfill(6) + '#' + \
                str(entry['end']).zfill(6) + '.npy'
            spec_path = os.path.join(args.out_folder, partition, path)
            torch.save(spec, spec_path)
            # Remove mp3
            os.remove(mp3_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video_folder", default='vox_raw', help='Path to youtube videos')
    parser.add_argument(
        "--metadata", default='vox-metadata.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='vox',
                        help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=1, type=int,
                        help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl',
                        help='Path to youtube-dl')
    parser.add_argument("--audio", action='store_true',
                        help='extract audio and get melspectogram')

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape, None for no resize")

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.map(run, zip(video_ids, args_list))):
        None
