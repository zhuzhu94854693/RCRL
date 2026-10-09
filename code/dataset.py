# dataset.py
import os
import cv2
import glob
import random
import numpy as np
import torch
import torch.utils.data

# ============================================================
# 数据集路径
# ============================================================
dataset_1 = r'/home/user/storage/mainBack/zly/datasets/shuguang'
dataset_2 = r'/home/user/storage/mainBack/zly/datasets/Italy'
dataset_3 = r'/home/user/storage/mainBack/zly/datasets/yellow_C'
dataset_4 = r'/home/user/storage/mainBack/zly/datasets/yellow_D'
dataset_5 = r'/home/user/storage/mainBack/zly/datasets/Img17'
datasets = [dataset_1, dataset_2, dataset_3, dataset_4, dataset_5]

# ============================================================
# 工具函数
# ============================================================
def normalize_image(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def find_dataset_files(dataset):
    def _find(pattern):
        files = glob.glob(os.path.join(dataset, pattern))
        return files[0] if files else None

    T1 = _find('*_1.*')
    T2 = _find('*_2.*')
    GT = _find('*_gt.*')
    if T1 is None or T2 is None:
        raise FileNotFoundError(f"T1/T2 未找到: {dataset}")
    return T1, T2, GT


def find_pseudo_label(dataset, p_tra='IRGMcS', pseudo_name=None):
    if pseudo_name is not None:
        candidates = [
            os.path.join(dataset, 'Pseudo', pseudo_name),
            os.path.join(dataset, pseudo_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    cand_dirs = [
        os.path.join(dataset, 'Pseudo'),
        os.path.join(dataset, 'pseudo_USSD'),
    ]
    patterns = [
        'CM_map_iter_6.png',
        'USSD.tif',
        '*pseudo*.png',
        '*pseudo*.tif',
        '*.png',
    ]
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        for p in patterns:
            files = glob.glob(os.path.join(d, p))
            if files:
                return files[0]
    return None


def _read_pair_and_pseudo(dataset, p_tra='IRGMcS', pseudo_name=None):
    """返回 T1[T1,H,W,3] float32、T2、pseudo[H,W] 0/1、GT[H,W] 0/1 或 None"""
    t1_path, t2_path, gt_path = find_dataset_files(dataset)
    T1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
    T2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
    GT = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if gt_path else None

    if T1 is None or T2 is None:
        raise RuntimeError(f"读图失败: {t1_path}, {t2_path}")

    if T1.ndim == 2:
        T1 = np.stack([T1] * 3, axis=-1)
    if T2.ndim == 2:
        T2 = np.stack([T2] * 3, axis=-1)

    T1 = normalize_image(T1)
    T2 = normalize_image(T2)

    if GT is not None:
        GT = (GT > 127).astype(np.uint8)
        # 保证 GT 与 T1 尺寸一致
        if GT.shape != T1.shape[:2]:
            GT = cv2.resize(GT, (T1.shape[1], T1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    pseudo_path = find_pseudo_label(dataset, p_tra, pseudo_name)
    if pseudo_path is not None:
        pseudo = cv2.imread(pseudo_path, cv2.IMREAD_GRAYSCALE)
        if pseudo is None:
            raise RuntimeError(f"伪标签读取失败: {pseudo_path}")
        pseudo = (pseudo > 127).astype(np.uint8)
        if pseudo.shape != T1.shape[:2]:
            pseudo = cv2.resize(pseudo, (T1.shape[1], T1.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        print(f"[dataset] 伪标签: {pseudo_path}")
    else:
        diff = np.abs(T1 - T2).mean(axis=2)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        thr = np.quantile(diff, 0.9)
        pseudo = (diff > thr).astype(np.uint8)
        print(f"[dataset] 未找到伪标签，使用 CVA+分位数生成（thr={thr:.3f}）")

    return T1, T2, pseudo, GT


# ============================================================
# 整图 Dataset
# ============================================================
class HCD(torch.utils.data.Dataset):
    """
    整图模式：
      训练：返回 (img1, img2, pseudo)  —— 均整图
      测试：返回 (img1, img2, gt_or_pseudo, i, j)，i=j=0 表示整图
    属性：
      self.H, self.W       原图尺寸
      self.pseudo_label    整图伪标签 [H,W]
      self.gt              整图 GT [H,W] 或 None
    """
    def __init__(self, train=True, transform=None,
                 dataset_id=1, p_tra='IRGMcS',
                 patch_size=None, num=None,
                 pseudo_name=None, augment=False,
                 repeat=1):
        super().__init__()
        dataset = datasets[dataset_id - 1]
        self.train = train
        self.transform = transform
        self.repeat = repeat if train else 1

        T1, T2, pseudo, GT = _read_pair_and_pseudo(dataset, p_tra, pseudo_name)

        self.H, self.W = T1.shape[:2]
        self.T1 = torch.from_numpy(T1.transpose(2, 0, 1)).float()      # [3,H,W]
        self.T2 = torch.from_numpy(T2.transpose(2, 0, 1)).float()      # [3,H,W]
        self.pseudo_label = pseudo                                     # [H,W] uint8
        self.gt = GT

    def __len__(self):
        return self.repeat

    def __getitem__(self, index):
        img1 = self.T1.clone()
        img2 = self.T2.clone()
        pseudo = torch.from_numpy(self.pseudo_label.astype(np.float32)).unsqueeze(0)  # [1,H,W]

        if self.train and self.transform is not None:
            # 对整图做同样的几何变换（保持 img1/img2/pseudo 同步）
            seed = random.randint(0, 2 ** 31 - 1)
            random.seed(seed); np.random.seed(seed)
            img1_np = self.transform(img1.permute(1, 2, 0).numpy())
            random.seed(seed); np.random.seed(seed)
            img2_np = self.transform(img2.permute(1, 2, 0).numpy())
            img1 = torch.from_numpy(img1_np.transpose(2, 0, 1)).float()
            img2 = torch.from_numpy(img2_np.transpose(2, 0, 1)).float()

        if self.train:
            return img1, img2, pseudo
        else:
            if self.gt is not None:
                label = torch.from_numpy(self.gt.astype(np.float32)).unsqueeze(0)
            else:
                label = pseudo.clone()
            # 整图模式坐标占位
            return img1, img2, label, 0, 0


# ============================================================
# 测试/使用
# ============================================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_set = HCD(train=True, dataset_id=8, p_tra='IRGMcS')
    print(f"train size(H,W) = {train_set.H},{train_set.W}, len={len(train_set)}")
    x1, x2, p = train_set[0]
    print("train:", x1.shape, x2.shape, p.shape)

    test_set = HCD(train=False, dataset_id=8, p_tra='IRGMcS')
    print(f"test size(H,W) = {test_set.H},{test_set.W}, len={len(test_set)}")
    x1, x2, y, i, j = test_set[0]
    print("test:", x1.shape, x2.shape, y.shape, i, j)

    loader = DataLoader(train_set, batch_size=1, shuffle=True)
    for a, b, c in loader:
        print("batch:", a.shape, b.shape, c.shape)
        break
