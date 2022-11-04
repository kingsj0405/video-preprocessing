import csv
from pathlib import Path

import click
import cv2
from tqdm import tqdm


def write_data_to_csv(data, csv_path, header_keys):
    with open(csv_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header_keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


@click.command()
@click.option('--data_root', default='./vox2')
@click.option('--new_data_root', default='./vox2_44k')
@click.option('--metadata_train_path', default='./vox2_meta_train.csv')
@click.option('--metadata_test_path', default='./vox2_meta_test.csv')
def main(
    data_root: str,
    new_data_root: str,
    metadata_train_path: str,
    metadata_test_path: str,
):
    data_root = Path(data_root)
    video_iter = data_root.glob('**/*.mp4')
    video_cnt = {'train':0, 'test':0}
    data_train = []
    data_test = []
    cnt = 0
    num_frames = 60
    for _, video_path in enumerate(tqdm(video_iter)):
        total_frames = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= num_frames:
            print(f'[DEBUG] Shotr video({video_path.stem}) is excluded total_frames: {total_frames}, num_frames: {num_frames}')
            continue
        if cnt < 400:
            new_video_path = Path(new_data_root) / 'test' / video_path.name
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            new_video_path.write_bytes(video_path.read_bytes())
            data_test.append({'video_path': str(new_video_path)})
            video_cnt['train'] += 1
            cnt += 1
        elif cnt < 40400:
            new_video_path = Path(new_data_root) / 'train' / video_path.name
            new_video_path.parent.mkdir(parents=True, exist_ok=True)
            new_video_path.write_bytes(video_path.read_bytes())
            data_train.append({'video_path': str(new_video_path)})
            video_cnt['test'] += 1
            cnt += 1
        else:
            break
    print(f'[DEBUG] video_cnt: {video_cnt}')

    header_keys = ['video_path']
    write_data_to_csv(data_train, metadata_train_path, header_keys)
    write_data_to_csv(data_test, metadata_test_path, header_keys)
    print(f'[INFO] Done!')


if __name__ == '__main__':
    main()
