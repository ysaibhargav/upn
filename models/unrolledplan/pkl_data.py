import pickle
import numpy as np
import glob
import os
import argparse

"""
~/anaconda3/bin/python pkl_data.py --path-prefix "../../../../Data/processed/caps-small2/pushfront_cubeenv1*" --save-dir pkl
"""


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path-prefix', type=str, default='../../../../Data/processed/caps-small2/pushfront_cubeenv1*', help='path to images and actions')
    parser.add_argument('--save-dir', type=str, default='pkl', help='path to output dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dirs = glob.glob(args.path_prefix)
    dirs = [_dir for _dir in dirs if os.path.isdir(_dir)]

    txt_data = []
    img_data = []
    lens = []
    for _dir in dirs:
        txt_file = glob.glob(os.path.join(_dir, '*txt'))[0]
        img_files = glob.glob(os.path.join(_dir, '*p'))
        img_files = sorted(img_files, key=lambda x: int(x.split(os.path.sep)[-1].split(".")[0]))
        f = open(txt_file, 'r')
        _txt_data = [line.strip().split() for line in f]#[1:]
        f.close()
        _img_data = []
        for _img_file in img_files:
            img = pickle.load(open(_img_file, 'rb'))
            #plt.close()
            #img = plt.imread(_img_file)
            _img_data.append(np.array(img, dtype=np.float32)[:, :, :3]/255.)

        if len(_txt_data) != len(_img_data):
            _txt_data = _txt_data[1:]
        assert len(_txt_data) == len(_img_data), "{}_{}".format(len(_txt_data), len(_img_data))
        _txt_data = np.array(_txt_data, dtype=np.float32)

        txt_data.append(_txt_data)
        img_data.append(_img_data)
        lens.append(len(_img_data))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    pickle.dump([txt_data, img_data, lens],
        open(os.path.join(args.save_dir,
        args.path_prefix.split(os.path.sep)[-1].split('*')[0]+'.p'),
        'wb'), protocol=2)


if __name__ == '__main__':
    main()
