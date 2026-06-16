import os
import numpy as np
import rasterio
from torch.utils.data import Dataset


class OpenEarthMapDataset(Dataset):
    """PyTorch Dataset for OpenEarthMap flat layout.

    Split files may contain entries like "tyrolw/tyrolw_25" or plain "tyrolw_25".
    In both cases os.path.basename strips any leading path, giving the flat filename.

    Directory layout expected on Kaggle:
        ROOT_DIR/
        ├── images/train/*.tif
        ├── images/val/*.tif
        ├── labels/train/*.tif
        └── labels/val/*.tif
    """

    CLASSES = [
        'Background', 'Bareland', 'Rangeland', 'Developed',
        'Road', 'Tree', 'Water', 'Agriculture', 'Building',
    ]
    NUM_CLASSES = 9
    IGNORE_INDEX = 255

    def __init__(self, root_dir, img_dir, mask_dir, split_file, transform=None):
        self.root_dir  = root_dir
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

        assert os.path.exists(split_file), f"Split file not found: {split_file}"
        with open(split_file) as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            name = os.path.basename(line.strip())
            if not name:
                continue
            img_path  = os.path.join(root_dir, img_dir,  name + '.tif')
            mask_path = os.path.join(root_dir, mask_dir, name + '.tif')
            assert os.path.exists(img_path),  f"Image not found: {img_path}"
            assert os.path.exists(mask_path), f"Mask not found:  {mask_path}"
            self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        with rasterio.open(img_path) as src:
            image = src.read()                        # (C, H, W) uint8
        image = np.transpose(image, (1, 2, 0))        # (H, W, C)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        image = image.astype(np.uint8)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)       # (H, W)

        if self.transform:
            result = self.transform(image=image, mask=mask)
            image = result['image']                   # float tensor (C, H, W)
            mask  = result['mask'].long()             # int64 tensor (H, W)

        return image, mask
