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
        # Rescale 斜率/截距（CT 等存储原始值，需换算到真实强度）
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        if slope != 1.0 or inter != 0.0:
            arr = arr * slope + inter
        # MONOCHROME1（白=0）→ 翻转为常规 MONOCHROME2（黑=0）
        if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                arr = hi - (arr - lo) + lo
        meta = {"format": "dicom", "orig_hw": arr.shape,
                "modality": str(getattr(ds, "Modality", "")),
                "patient_id": str(getattr(ds, "PatientID", "")),
                "study_date": str(getattr(ds, "StudyDate", ""))}
        return _to01(arr), meta

    raise ValueError(f"不支持的文件类型: {ext}")


def save_dicom(pred01: np.ndarray, path: str, source_path: str | None = None,
               source_mod: str = "", target_mod: str = "") -> str:
    """翻译结果 → DICOM Secondary Capture（16-bit 灰度，全量程线性映射）。

    source_path 给定时继承患者/检查标签便于追溯；翻译图像标记 OT 模态
    （符合 DICOM 对派生图像的规范，避免与真实采集模态混淆）。
    """
    import datetime

    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    pred = np.clip(np.asarray(pred01, dtype=np.float32), 0.0, 1.0)
    pixels = (pred * 65535.0).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"   # Secondary Capture
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"                       # 派生图像
    ds.SeriesDescription = f"TrustBridge translation {source_mod}->{target_mod}".strip()
    ds.PatientName = "ANONYMOUS"
    ds.PatientID = "ANONYMOUS"
    if source_path and os.path.exists(source_path):
        try:
            src = pydicom.dcmread(source_path, stop_before_pixels=True)
            for tag in ("PatientName", "PatientID", "StudyDate"):
                if tag in src:
                    ds[tag].value = src[tag].value
            if "StudyInstanceUID" in src:
                ds.StudyInstanceUID = src.StudyInstanceUID      # 归入原检查
        except Exception:
            pass
    ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
    ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
    ds.ImageComments = ("Derived by TrustBridge (diffusion-bridge translation); "
                        "research demonstration, NOT for clinical use")
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0                                  # unsigned
    ds.RescaleSlope = f"{1.0 / 65535.0:.9g}"   # DS 字符串最长 16 字符（指数形式）
    ds.RescaleIntercept = "0"
    ds.PixelData = pixels.tobytes()
    # 传输语法已在 file_meta（ExplicitVRLittleEndian）；pydicom 3 弃用 is_little_endian/is_implicit_VR
    ds.save_as(path, enforce_file_format=True)
    return path


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
