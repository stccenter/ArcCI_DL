import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class IceTilesDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):
        self.img_list = glob.glob(f'{dataset_dir}/image/*')
        self.img_list.sort()
        self.msk_list = glob.glob(f'{dataset_dir}/mask/*')
        self.msk_list.sort()
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        msk = Image.open(self.msk_list[index]).convert('L')
        img = np.array(img)
        msk = np.array(msk)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=msk)
            img = transformed["image"]
            msk = transformed["mask"]
        msk = msk.long()
        return {'image': img, 'mask': msk}

    def __len__(self):
        return len(self.img_list)
