import glob
import os.path as osp

training_root_dir = '/home/andybi7676/Desktop/ds2022_finalProj/Training'
gt_fpath = osp.join(training_root_dir, "all_gt.txt")

with open(gt_fpath, 'w') as fw:
    for training_fpath in glob.glob(osp.join(training_root_dir, "*.circles")):
        train_ego_id = training_fpath.split('/')[-1].split('.')[0]
        circles_reprs = []
        with open(training_fpath, 'r') as fr:
            for line in fr:
                circles_reprs.append(line.split(': ')[-1].strip())
        circles_repr = ';'.join(circles_reprs)
        print(f"{train_ego_id},{circles_repr}", file=fw)
 
