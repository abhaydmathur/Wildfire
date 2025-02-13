import glob 
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser("Preprocess data for Wildfire Dataset")
    parser.add_argument('--root_dir', type=str, default='data/ROB313')
    parser.add_argument('--labeled_train_ratio', type=float, default=0.8)
    return parser.parse_args()

def parse_filename(filename):
    splits = filename.split('/')
    filename = splits[-1]
    label = splits[-2]
    coords = filename.replace('.jpg', '').split(',')
    coord_x, coord_y = float(coords[0]), float(coords[1])
    return label, coord_x, coord_y

def main():
    args = get_args()

    root_dir = args.root_dir

    train_files = glob.glob(f'{root_dir}/train/*/*.jpg')
    test_files = glob.glob(f'{root_dir}/test/*/*.jpg')
    val_files = glob.glob(f'{root_dir}/valid/*/*.jpg')

    print(f"Number of train files: {len(train_files)}")
    print(f"Number of test files: {len(test_files)}")
    print(f"Number of val files: {len(val_files)}")

    np.random.shuffle(val_files)
    split_idx = int(len(val_files) * args.labeled_train_ratio)
    train_labeled_files = val_files[:split_idx]
    val_files_labeled = val_files[split_idx:]

    print(f"Number of train labeled files: {len(train_labeled_files)}")
    print(f"Number of val labeled files: {len(val_files_labeled)}")

    train_unlabeled = {
        'filename': [],
        'coord_x': [],
        'coord_y': []
    }
    for fname in tqdm(train_files, desc='Processing train unlabeled files'):
        _, coord_x, coord_y = parse_filename(fname)
        train_unlabeled['filename'].append(fname)
        train_unlabeled['coord_x'].append(coord_x)
        train_unlabeled['coord_y'].append(coord_y)
    
    train_unlabeled_df = pd.DataFrame(train_unlabeled)
    train_unlabeled_df.to_csv(f'{root_dir}/train_unlabeled.csv', index=False)

    train_labeled = {
        'filename': [],
        'coord_x': [],
        'coord_y': [],
        'label': []
    }
    for fname in tqdm(train_labeled_files, desc='Processing train labeled files'):
        label, coord_x, coord_y = parse_filename(fname)
        train_labeled['filename'].append(fname)
        train_labeled['coord_x'].append(coord_x)
        train_labeled['coord_y'].append(coord_y)
        train_labeled['label'].append(label)
    train_labeled_df = pd.DataFrame(train_labeled)
    train_labeled_df.to_csv(f'{root_dir}/train.csv', index=False)

    val_labeled = {
        'filename': [],
        'coord_x': [],
        'coord_y': [],
        'label': []
    }
    for fname in tqdm(val_files_labeled, desc='Processing val labeled files'):
        label, coord_x, coord_y = parse_filename(fname)
        val_labeled['filename'].append(fname)
        val_labeled['coord_x'].append(coord_x)
        val_labeled['coord_y'].append(coord_y)
        val_labeled['label'].append(label)

    val_labeled_df = pd.DataFrame(val_labeled)
    val_labeled_df.to_csv(f'{root_dir}/val.csv', index=False)

    test = {
        'filename': [],
        'coord_x': [],
        'coord_y': [],
        'label': []
    }
    for fname in test_files:
        label, coord_x, coord_y = parse_filename(fname)
        test['filename'].append(fname)
        test['coord_x'].append(coord_x)
        test['coord_y'].append(coord_y)
        test['label'].append(label)

    test_df = pd.DataFrame(test)
    test_df.to_csv(f'{root_dir}/test.csv', index=False)

if __name__ == '__main__':
    main()
        