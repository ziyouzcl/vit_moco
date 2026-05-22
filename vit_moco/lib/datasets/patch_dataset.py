# -*- coding: utf-8 -*-
"""
patch_dataset — 3D 医学影像 ROI Patch 数据集（PatchDataset）

作用
    将原始 3D CT/MRI 及其 ROI mask 转为模型可直接使用的固定尺寸 patch（默认 48³）。

能力概览
    - 数据来源：CSV 列表，或 Image/Mask 成对目录。
    - 几何信息：自动计算 ROI 质心与 bounding box（prep_row）。
    - 裁剪策略：big_roi_resize、all_roi_resize、crop、resize 等（见下）。
    - 可选项：强度归一化（normalize）、外部注入的数据增强（transform）、
              仅保留 ROI（only_roi）、mask 引导双通道（get_with_mask）、显式负样本（enable_negatives）。
    - 适用场景：3D ViT 微调/预后、特征提取；MoCo 需另配双视图 Dataset，不直接走 enable_negatives。

主要裁剪策略：
- crop：以 ROI 质心为中心裁剪固定大小 patch，保留原始尺度和瘤周信息，但大病灶可能裁不全。
- big_roi_resize：小 ROI 直接裁固定大小 patch，大 ROI 先完整裁出再 resize，是较折中的推荐方案。
- all_roi_resize：完整裁出 ROI 后统一 resize，保证包含病灶，但会弱化真实大小信息。
- all_roi_crop：完整裁出 ROI 但不 resize，保留原始尺度，但输出大小不固定。
- resize：直接将整张图 resize，不依赖 ROI，适合无 mask 场景，但可能丢失局部病灶细节。

肿瘤预后任务中，通常优先考虑 big_roi_resize 或 crop，并建议保留瘤周区域，即 only_roi=False。
@对外暴露接口:
    __all__ = ['PatchDataset']
"""
from math import e
import os
from pathlib import Path
from easydict import EasyDict as edict
from typing import Union, Tuple
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import scipy
import monai
monai.data.set_track_meta(False)
#医学影像强度域预处理与数据增强模块：NormalizeIntensityUnbiased
from vit_moco.lib.transforms.intensity import NormalizeIntensityUnbiased

def as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x).float()

def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

class ROIConv3d(torch.nn.Module):
    def __init__(self, kernel_size=5):
        super(ROIConv3d, self).__init__()
        self.padding = kernel_size // 2  # 计算填充大小以保持输入和输出形状一致
        self.weight = torch.ones(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        # 在前向传播中使用自定义卷积核进行卷积操作，并使用合适的填充来保持输入形状不变
        return torch.nn.functional.conv3d(x, self.weight, padding=self.padding)
    
    def cuda(self):
        self.weight = self.weight.cuda()
        return self


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 csv_path: Union[str, Path, None] = None, 
                 image_dir: Union[str, Path, None] = None,
                 mask_dir: Union[str, Path, None] = None,
                 method: str = "big_roi_resize",
                 size: Union[int, Tuple[int]] = 48,
                 label: Union[str, None] = None,
                 enable_negatives: bool = False,
                 negative_dir: Union[str, Path, None] = None,
                 transform: Union[torch.nn.Module, None] = None,
                 normalize: bool = True,
                 normalize_unbiased: bool = False,
                 only_roi: bool = False,
                 read_in_memory: bool = False,
                 get_with_mask: bool = False,
                 ):
        
        assert method in [None, "big_roi_resize", "all_roi_resize", "crop", "resize", "all_roi_crop"]
        if isinstance(size, int):
            self.size = (size, size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise TypeError(f"Unsupported type for size: {type(size)}")
        
        assert isinstance(csv_path, (str, Path)) or image_dir is not None

        self.csv_path = csv_path
        self.method = method
        self.label = label
        self.enable_negatives = enable_negatives
        self.negative_paths = list(Path(negative_dir).glob("*.nii.gz")) if negative_dir else None
        self.transform = transform
        self.normalize = normalize
        self._norm = NormalizeIntensityUnbiased() if normalize_unbiased else monai.transforms.NormalizeIntensity()
        self.only_roi = only_roi
        self.read_in_memory = read_in_memory
        self.cookies = {}
        self.get_with_mask = get_with_mask
        if self.get_with_mask:
            self.roi_conv3d = ROIConv3d(kernel_size=15).cuda()

        if csv_path is not None:
            self._df = pd.read_csv(csv_path)
            self._len = len(self._df)
            self._rows = [edict(_) for _ in self._df.to_dict(orient="index").values()]
        elif image_dir is not None:
            self.image_paths = sorted(Path(image_dir).iterdir())
            self.mask_paths = sorted(Path(mask_dir).iterdir()) if mask_dir is not None else None
            if self.mask_paths is not None:
                assert len(self.image_paths) == len(self.mask_paths)
            self._len = len(os.listdir(image_dir))
            self._rows = []
            for i in range(self._len):
                row = edict()
                row.image_path = self.image_paths[i]
                row.mask_path = self.mask_paths[i] if self.mask_paths is not None else None
                row.method = "resize"
                self._rows.append(row)
        else:
            raise ValueError("Either csv_path or image_dir should be provided.")

#返回数据集样本数
    def __len__(self):
        return self._len

#输入：一张 CT + 一个分割 Mask（病灶 / 器官）
#输出：这个病灶的 质心坐标 + bounding box（包围框）    
    @staticmethod
    def prep_row(image_path, mask_path):
        if isinstance(image_path, np.ndarray) or isinstance(mask_path, np.ndarray):
            image_np = image_path
            mask_np = mask_path
        else:
            image_path = Path(image_path)
            mask_path = Path(mask_path)
            image_np = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            mask_np = mask_np.astype("int32")
            mask_np[mask_np != 1] = 0

        if image_np.shape != mask_np.shape:
            print("Shape mismatch: \n    ", image_path, mask_path)
            return None

        row = edict()
        # =========================== calculate centroid ===========================
        try:
            centroid = scipy.ndimage.center_of_mass(mask_np)
            assert np.isnan(centroid).any() == False
        except:
            print("Empty mask: ", mask_path)
            return None
        x, y, z = centroid
        row["coordX"], row["coordY"], row["coordZ"] = x, y, z

        # =========================== calculate bounding box ===========================
        bbox = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask_np, axis=0), margin=1, allow_smaller=False)
        roi_start, roi_end = bbox

        if roi_start == roi_end:
            print(f"Empty mask: {mask_path}")
            return None

        row["roi_start"], row["roi_end"] = roi_start, roi_end
        for i, n in enumerate(["x", "y", "z"]):
            row[f"roi_start_{n}"], row[f"roi_end_{n}"] = row["roi_start"][i], row["roi_end"][i]

        return row
    
    def read_image(self, path: Union[str, Path]):
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if self.normalize:
            img = self._norm(img)
        return img
    
    def read_mask(self, path: Union[str, Path]):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype("int32")
        mask[mask > 0] = 1
        return mask
    
    def read_negative(self):
        img = self.read_image(np.random.choice(self.negative_paths))
        return monai.transforms.ToTensor(track_meta=False)(img)
    
    def all_roi_crop(self, img, roi_center, roi_size):
        crop_size = max(roi_size)
        cropper = monai.transforms.Compose([
            monai.transforms.ToTensor(track_meta=False),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return cropper(img)
    
    def big_roi_resize(self, img, roi_center, roi_size):
        crop_size = max(max(roi_size), self.size[0])
        cropper = monai.transforms.Compose([
            monai.transforms.ToTensor(track_meta=False),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
            monai.transforms.Resize(spatial_size=self.size) if crop_size > self.size[0] else monai.transforms.Lambda(lambda x: x),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return cropper(img)
    
    def all_roi_resize(self, img, roi_center, roi_size):
        crop_size = max(roi_size)
        cropper = monai.transforms.Compose([
            monai.transforms.ToTensor(track_meta=False),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
            monai.transforms.Resize(spatial_size=self.size),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return cropper(img)
    
    def crop_img(self, img, roi_center):
        cropper = monai.transforms.Compose([
            monai.transforms.ToTensor(track_meta=False),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=self.size),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return cropper(img)
    
    def resize_img(self, img):
        if img.shape == self.size:
            return img
        
        resizer = monai.transforms.Compose([
            monai.transforms.ToTensor(track_meta=False),
            monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
            monai.transforms.Resize(spatial_size=self.size),
            monai.transforms.SqueezeDim(dim=0),
        ])
        return resizer(img)
    
    def crop_negative_patch(self, img):
        valid_patch_size = monai.data.utils.get_valid_patch_size(img.shape, self.size)
        neg_patch = img[monai.data.utils.get_random_patch(img.shape, valid_patch_size)]
        return monai.transforms.ToTensor(track_meta=False)(neg_patch)
    
    def get_positive_patch(self, index, img, mask=None):
        row = self._rows[index]
        method = row.get("method", self.method)
        
        # Read ROI
        if method != "resize":
            try:
                centroid = (row["coordX"], row["coordY"], row["coordZ"])
                roi_start = [row.roi_start_x, row.roi_start_y, row.roi_start_z]
                roi_end = [row.roi_end_x, row.roi_end_y, row.roi_end_z]
            except:
                row.update(self.prep_row(row.image_path, row.mask_path))
                centroid = (row["coordX"], row["coordY"], row["coordZ"])
                roi_start = [row.roi_start_x, row.roi_start_y, row.roi_start_z]
                roi_end = [row.roi_end_x, row.roi_end_y, row.roi_end_z]

            roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
            roi_size = [roi_end[i] - roi_start[i] for i in range(3)]        

        # Notably, ROI bbox and patch are both cubes
        if method is None:
            ...
        elif method == "big_roi_resize": 
            # If ROI bbox is bigger than patch, resize ROI bbox to patch size
            # else, do nothing
            img = self.big_roi_resize(img, roi_center, roi_size)
            mask = self.big_roi_resize(mask, roi_center, roi_size) if mask is not None else None
        elif method == "all_roi_resize":
            # Resize ROI bbox to patch size
            img = self.all_roi_resize(img, roi_center, roi_size)
            mask = self.all_roi_resize(mask, roi_center, roi_size) if mask is not None else None
        elif method == "all_roi_crop":
            # Crop ROI bbox (cube) from image, do not resize
            img = self.all_roi_crop(img, roi_center, roi_size)
            mask = self.all_roi_crop(mask, roi_center, roi_size) if mask is not None else None
        elif method == "crop":
            # Crop patch from image, do not resize
            img = self.crop_img(img, centroid)
            mask = self.crop_img(mask, centroid) if mask is not None else None
        elif method == "resize":
            # Resize image to patch size, do not crop
            img = self.resize_img(img)
            mask = self.resize_img(mask) if mask is not None else None
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        pos_patch = as_numpy(img)
        mask_patch = as_numpy(mask) if mask is not None else None
        mask_patch = mask_patch.astype("int32") if mask is not None else None
        

        if mask is not None and self.only_roi:
            assert np.sum(mask_patch != 0) == np.sum(mask_patch == 1) # 0: background, 1: foreground
            # background = np.ones_like(pos_patch) * pos_patch.min()
            if pos_patch.min() < 0:
                background = np.full_like(pos_patch, -1024) # CT
            else:
                background = np.zeros_like(pos_patch) # MR
            pos_patch = np.where(mask_patch == 1, pos_patch, background)
        return pos_patch, mask_patch

    def __getitem__(self, index, transform=True):
        row = self._rows[index]
        if self.read_in_memory and index in self.cookies:
            pos_patch, mask_patch = self.cookies[index]
        else:
            # Read Image
            img = self.read_image(row.image_path)
            mask = self.read_mask(row.mask_path) if self.only_roi or self.get_with_mask else None

            # Get Positive Patch
            pos_patch, mask_patch = self.get_positive_patch(index, img, mask)
            if self.read_in_memory:
                self.cookies[index] = (pos_patch, mask_patch)
        
        if self.get_with_mask:
            with torch.no_grad():
                mask_patch_x = self.roi_conv3d(as_tensor(mask_patch).cuda().unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()
            pos_patch = np.stack([pos_patch, mask_patch_x], axis=0)

        if transform:
            pos_patch = self.transform(pos_patch) if self.transform else pos_patch
        # Get Label
        try:
            target = int(row[self.label]) if self.label is not None else False
        except:
            target = False
        
        # Get Negative Patch
        if self.enable_negatives:
            if self.negative_paths is None:
                neg_patch = self.crop_negative_patch(img)
            else:
                neg_patch = self.read_negative()
            neg_patch = self.transform(neg_patch) if self.transform else neg_patch
            return {"positive": pos_patch, "negative": neg_patch}, target
        
        return pos_patch, target

    def get_row(self, index):
        return self._rows[index]
    
    def get_mask(self, index):
        row = self._rows[index]
        mask = self.read_mask(row.mask_path)
        return mask
    
    def get_mask_patch(self, index):
        row = self._rows[index]

        # Read Mask
        img = self.read_image(row.image_path)
        mask = self.read_mask(row.mask_path)
        _, mask_patch = self.get_positive_patch(index, img, mask)
        mask_patch[mask_patch > 0] = 1
        return as_numpy(mask_patch)
    
    def save(self, csv_path):
        # self._rows
        new_df = pd.DataFrame(self._rows)
        new_df.to_csv(csv_path, index=False)
    

def test1():
    from pathlib import Path
    import shutil

    csv_path = "/home/wzt/src/fmcibx/data/bbox/ALL_pretrain_bbox.csv"
    enable_negatives = False
    size = 48
    label = "label"
    only_roi = False
    method = ["big_roi_resize", "crop", "resize"][0]
    idx = 58

    output_dir = Path("/mnt/tmp/0")
    output_dir.mkdir(exist_ok=True, parents=True)  

    dataset = PatchDataset(csv_path=csv_path, method=method, label="label", size=size, enable_negatives=enable_negatives, only_roi=only_roi)
    print("len: ", len(dataset))
    
    row = dataset.get_row(idx)
    print("image_path: ", row.get("image_path", "None"))
    print("mask_path: ", row.get("mask_path", "None"))
    
    x, label = dataset[idx]
    if enable_negatives:
        print(x["positive"].shape, x["negative"].shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x["positive"]), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(x["negative"]), str(output_dir / f"{idx:03d}_patch_neg.nii.gz"))
    else:
        print(x.shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))

    mask_patch = dataset.get_mask_patch(idx)
    sitk.WriteImage(sitk.GetImageFromArray(mask_patch), str(output_dir / f"{idx:03d}_patch_mask.nii.gz"))

def test2():
    from pathlib import Path
    import shutil

    image_dir = "/media/wzt/wd18t/ProcessedData/ROI_resize48/CRC_Shanxi/all_roi_resize/image"
    mask_dir = "/media/wzt/wd18t/ProcessedData/ROI_resize48/CRC_Shanxi/all_roi_resize/mask"
    enable_negatives = False
    size = 48
    only_roi = False
    method = ["big_roi_resize", "crop", "resize"][0]
    idx = 58

    output_dir = Path("/mnt/tmp/0")
    output_dir.mkdir(exist_ok=True, parents=True)  

    dataset = PatchDataset(image_dir=image_dir, mask_dir=mask_dir, method="resize", label="label", size=size, enable_negatives=enable_negatives, only_roi=only_roi)
    print("len: ", len(dataset))
    
    row = dataset.get_row(idx)
    print("image_path: ", row.get("image_path", "None"))
    print("mask_path: ", row.get("mask_path", "None"))
    
    x, label = dataset[idx]
    if enable_negatives:
        print(x["positive"].shape, x["negative"].shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x["positive"]), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(x["negative"]), str(output_dir / f"{idx:03d}_patch_neg.nii.gz"))
    else:
        print(x.shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))

    mask_patch = dataset.get_mask_patch(idx)
    sitk.WriteImage(sitk.GetImageFromArray(mask_patch), str(output_dir / f"{idx:03d}_patch_mask.nii.gz"))

if __name__ == '__main__':
    # test1()
    test2()
    # debug()
    ...