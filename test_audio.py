from pathlib import Path
import fire
import torch


DATA_DIR = Path('vox/train')


def test():
    video_paths = list(sorted(DATA_DIR.glob('*.mp4')))
    cnt_all = 0
    cnt_warn1 = 0
    cnt_warn2 = 0
    for video_path in video_paths:
        # print(f'[DEBUG] video_path: {video_path}')
        audio_path = Path(f'{video_path.parent}/{video_path.stem}.npy')
        frame_paths = list(sorted(video_path.glob('*.png')))
        # print(f'[DEBUG] frame_paths: size - {len(frame_paths)}, first - {frame_paths[0]}')
        if audio_path.exists():
            audio = torch.load(audio_path)
        else:
            print(f'[WARNING] video id: {video_path.stem}, len(frame_paths): {len(frame_paths)}')
            cnt_warn1 += 1
        # print(f'[DEBUG] audio.shape: {audio.shape}, audio_path: {audio_path}')
        if audio.shape[1] != len(frame_paths):
            print(f'[WARNING] video id: {video_path.stem}, audio.shape: {audio.shape}, len(frame_paths): {len(frame_paths)}')
            cnt_warn2 += 1
        cnt_all += 1
    print(f'[INFO] {cnt_warn1}/{cnt_all}, {cnt_warn2}/{cnt_all}')


if __name__ == '__main__':
    fire.Fire(test)