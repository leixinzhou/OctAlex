from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from AugSurfSeg import *

SLICE_per_vol = 60

class OCTDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, img_np, label_np, surf, vol_list=None, transforms=None):
        """
        Args:
            img_np (string): Path to the image numpy file.
            label_np (string): Path to the label numpy file.
            vol_list (list): sample list.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sf = surf
        self.image = np.load(img_np, mmap_mode='r')
        self.label = np.load(label_np, mmap_mode='r')
        self.vol_list = vol_list
        self.trans = transforms

    def __len__(self):
        if self.vol_list is None:
            return self.image.shape[0]
        else:
            return len(self.vol_list)*SLICE_per_vol

    def __getitem__(self, idx):
        real_idx = self.vol_list[int(idx/SLICE_per_vol)]*SLICE_per_vol + idx % SLICE_per_vol
        image = self.image[real_idx, ]
        image_ts = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = self.label[real_idx,:, self.sf]
        label_ts = torch.from_numpy(label.astype(np.float32).reshape(-1))
        
        sample = {"img": image_ts, "gt": label_ts}
        
        return sample
       
if __name__ == "__main__":
    """
    test the class
    """
    vol_list = [10,3]
    vol_index = 1
    slice_index = 10
    patch_dir = "/home/leizhou/Documents/OCT/60slice/split_data_2D_400/test/patch.npy"
    truth_dir = "/home/leizhou/Documents/OCT/60slice/split_data_2D_400/test/truth.npy"
    dataset = OCTDataset(img_np=patch_dir, label_np=truth_dir, surf=[0,1,2], vol_list=vol_list)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    test_patch = np.load(patch_dir, mmap_mode='r')
    test_truth = np.load(truth_dir, mmap_mode='r')
    _, axes = plt.subplots(1,2)
    axes[0].imshow(np.transpose(test_patch[vol_list[vol_index]*SLICE_per_vol+slice_index,].astype(np.float32)), cmap='gray')
    axes[0].plot(test_truth[vol_list[vol_index]*SLICE_per_vol+slice_index,:,0])
    axes[0].plot(test_truth[vol_list[vol_index]*SLICE_per_vol+slice_index,:,1])
    axes[0].plot(test_truth[vol_list[vol_index]*SLICE_per_vol+slice_index,:,2])
    for i, batch in enumerate(loader):
        if i == vol_index*SLICE_per_vol+slice_index:
            img = batch['img'].squeeze().numpy()
            gt = batch['gt'].squeeze().numpy()
            break
    axes[1].imshow(np.transpose(img), cmap='gray')
    axes[1].plot(gt[:400])
    axes[1].plot(gt[400:800])
    axes[1].plot(gt[800:1200])
    plt.show()
    
