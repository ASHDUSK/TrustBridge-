"""图像 IO：PNG/JPG/NIfTI/DICOM 读取与保存。"""
from __future__ import annotations

import os
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _to01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    if a.max() > 1.5:  # 8/12/16bit 整数图像
        a = a / max(a.max(), 1.0)
    return np.clip(a, 0.0, 1.0)


def load_image(path: str, slice_idx: int | None = None) -> tuple[np.ndarray, dict]:
    """读取图像文件 → ([0,1] float32 HxW, meta)。

    - PNG/JPG 等灰度图：直接读
    - .nii / .nii.gz：slice_idx 指定轴位切片（默认中间层）
    - .dcm：pydicom 读取像素
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXTS:
        from PIL import Image
        img = Image.open(path).convert("L")
        return _to01(np.array(img)), {"format": ext[1:], "orig_hw": img.size[::-1]}

    if ext in (".nii", ".gz"):
        import nibabel as nib
        vol = nib.load(path).get_fdata(dtype=np.float32)
        if vol.ndim == 2:
            return _to01(vol), {"format": "nifti", "orig_hw": vol.shape}
        k = vol.shape[2] // 2 if slice_idx is None else int(slice_idx)
        k = int(np.clip(k, 0, vol.shape[2] - 1))
        sl = np.rot90(np.asarray(vol[:, :, k]), k=1)  # 轴位方向习惯
        return _to01(sl), {"format": "nifti", "slice": k, "n_slices": int(vol.shape[2]),
                           "orig_hw": sl.shape}

    if ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return _to01(arr), {"format": "dicom", "orig_hw": arr.shape}

    raise ValueError(f"不支持的文件类型: {ext}")


def load_volume(path: str) -> tuple[np.ndarray, object]:
    """读取 NIfTI 整卷 → (float32 [H,W,Z] [0,1], nibabel 图像对象)。"""
    import nibabel as nib
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    vol = np.transpose(vol, (1, 0, 2))  # x,y,z -> 行列一致
    hi = vol.max()
    if hi > 1.5:
        vol = vol / hi
    return np.clip(vol, 0.0, 1.0), img


def save_volume(arr_hwxz: np.ndarray, ref_img, path: str):
    """以参考图像的仿射保存翻译后的整卷。"""
    import nibabel as nib
    vol = np.transpose(arr_hwxz, (1, 0, 2)).astype(np.float32)
    out = nib.Nifti1Image(vol, ref_img.affine, ref_img.header.copy())
    out.set_data_dtype(np.float32)
    nib.save(out, path)
