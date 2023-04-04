import torch
from torch.utils.data import Dataset

import numpy as np
import pickle, argparse, os
from os.path import join, exists

from helper_ply import read_ply
from helper_tool import DataProcessing as DP

class Toronto3D(Dataset):
    def __init__(self, root, split, batch_size, grid_size=0.06, feature_mode=0):
        self.path = root
        self.label_to_names = {0: 'unclassified',
                               1: 'Ground',
                               2: 'Road marking',
                               3: 'Natural',
                               4: 'Building',
                               5: 'Utility line',
                               6: 'Pole',
                               7: 'Car',
                               8: 'Fence'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.feature_mode = feature_mode
        self.grid_size = grid_size

        # train hypers
        self.batch_size = batch_size
        self.split = split
        self.epoch_train_rounds = 500
        self.epoch_eval_rounds = 25
        self.noise_init = 3.5
        self.block_size = 65536


        self.full_pc_folder = join(self.path, 'original_ply')

        # Initial training-validation-testing files
        self.train_files = ['L001', 'L003', 'L004']
        self.val_files = ['L002']
        self.test_files = ['L002']



        self.train_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.train_files]
        self.val_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.val_files]
        self.test_files = [os.path.join(self.full_pc_folder, files + '.ply') for files in self.test_files]

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}
        self.input_trees = {'train': [], 'eval': [], 'test': []}
        self.input_colors = {'train': [], 'eval': [], 'test': []}
        self.input_labels = {'train': [], 'eval': []}
        self.load_sub_sampled_clouds(self.grid_size, self.split)

        # Reset possibility
        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []

         # Random initialize
        for i, tree in enumerate(self.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        if split != 'test':
            _, num_class_total = np.unique(np.hstack(self.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]

    def load_sub_sampled_clouds(self, sub_grid_size, mode):

        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        if mode == 'test':
            files = self.test_files
        else: 
            files = np.hstack((self.train_files, self.val_files))

        for i, file_path in enumerate(files):
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if mode == 'test':
                cloud_split = 'test'
            else:
                if file_path in self.val_files:
                    cloud_split = 'eval'
                else:
                    cloud_split = 'train'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # read ply with data
            data = read_ply(sub_ply_file)

            # read RGB / intensity accoring to configuration
            if self.feature_mode==3:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'], data['intensity'])).T
            elif self.feature_mode==2:
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            elif self.feature_mode==1:
                sub_colors = data['intensity'].reshape(-1,1)
            else:
                sub_colors = np.ones((data.shape[0],1))

            if cloud_split == 'test':
                sub_labels = None
            else:
                sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['train', 'eval']:
                self.input_labels[cloud_split] += [sub_labels]

            # Get test re_projection indices
            if cloud_split == 'test':
                print('\nPreparing reprojection indices for {}'.format(cloud_name))
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]

        print('finished')
        return

    def __len__(self):
        if self.split == 'train':
            return self.epoch_train_rounds*self.batch_size
        else:
            return self.epoch_eval_rounds*self.batch_size
    
    def __getitem__(self, idx):
         # Choose the cloud with the lowest probability
        cloud_idx = int(np.argmin(self.min_possibility[self.split]))

        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])

        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.split][cloud_idx].data, copy=False)

        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=self.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        query_idx = self.input_trees[self.split][cloud_idx].query(pick_point, k=self.block_size)[1][0]

        # Shuffle index
        query_idx = DP.shuffle_idx(query_idx)

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[query_idx]
        queried_pc_xyz[:, 0:2] = queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
        queried_pc_colors = self.input_colors[self.split][cloud_idx][query_idx]
        if self.split == 'test':
            queried_pc_labels = np.zeros(queried_pc_xyz.shape[0])
            queried_pt_weight = 1
        else:
            queried_pc_labels = self.input_labels[self.split][cloud_idx][query_idx]
            queried_pc_labels = np.array([self.label_to_idx[l] for l in queried_pc_labels])
            queried_pt_weight = np.array([self.class_weight[self.split][0][n] for n in queried_pc_labels])
            

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight
        self.possibility[self.split][cloud_idx][query_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        return (queried_pc_xyz,
                queried_pc_colors.astype(np.float32),
                queried_pc_labels,
        )

if __name__ == '__main__':
    dataset_path = '/home/song/datasets/Toronto_3D'
    dataset = Toronto3D(root=dataset_path, split='train', batch_size=4)

    for i in range(100):
        xyz, rgb, label = dataset[i]
        print(xyz.shape, rgb.shape, label.shape)
