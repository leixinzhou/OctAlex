from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch


class OCTDataset(Dataset):
    """convert 3d dataset to Dataset."""

    def __init__(self, img_np, label_np, surf, g_label_np=None, batch_nb=None):
        """
        Args:
            img_np (string): Path to the image numpy file.
            label_np (string): Path to the label numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sf = surf
        self.image = np.load(img_np, mmap_mode='r')
        if g_label_np == None:
            self.g_label = None
        else:
            # the g lable is memmap file, not numpy file
            self.g_label = np.load(g_label_np, mmap_mode='r')
        self.label = np.load(label_np, mmap_mode='r')
        self.bn = batch_nb

    def __len__(self):
        if self.bn is None:
            return self.image.shape[0]
        else:
            return self.bn

    def __getitem__(self, idx):
        image = self.image[idx, ]
        image_ts = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = self.label[idx,:, self.sf]
        label_ts = torch.from_numpy(label.astype(np.float32).reshape(-1))
        
        sample = {"img": image_ts, "gt": label_ts}
        if self.g_label is None:
            return sample
        else:
            g_label_ts = torch.from_numpy(
                self.g_label[idx, ].astype(np.float32))
            sample["gaus_gt"] = g_label_ts
            # img: x 1 400 512, gt: x 400, g_gt: x 400 512
            return sample