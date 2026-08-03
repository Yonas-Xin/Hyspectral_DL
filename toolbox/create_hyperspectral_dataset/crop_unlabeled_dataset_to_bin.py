"""
Generate an unlabeled patch dataset from hyperspectral imagery.
- Cropping logic mirrors toolbox/set_dataset.py (row-wise valid ranges + two-pass patching).
- Output patches can be saved as raw .bin (BHW order) or GeoTIFF.
- The script layout follows toolbox/split_and_clip_dataset.py: configuration at top, main() as entrypoint.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm
from osgeo import gdal

from core import Hyperspectral_Image
from gdal_utils import GDAL2NP_TYPE, write_data_to_tif, _write_bin_meta_file

# -----------------------------------------------------------------------------
# Configuration (edit to your data)
# -----------------------------------------------------------------------------
input_img_paths: List[str] = [
    r'',
]
output_root: str = r""
patch_size: int = 9
output_format: str = "bin"  # "bin" or "tif"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_row_valid_ranges(mask: np.ndarray, patch: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return row-wise valid x ranges for the mask."""
    h, _ = mask.shape
    coords = []
    valid_rows = np.where(np.any(mask, axis=1))[0]
    if len(valid_rows) == 0:
        return []
    y_start = valid_rows[0]
    for y in range(y_start, h, patch):
        if (y + patch // 2) > h:
            break
        row_data = mask[y, :]
        valid_x = np.where(row_data)[0]
        if len(valid_x) == 0:
            continue
        coords.append([(y, valid_x[0]), (y, valid_x[-1])])
    return coords


def _save_patch(
    full_data: np.ndarray,
    geotrans: Tuple[float, ...],
    projection: str,
    out_dir: str,
    prefix: str,
    idx: int,
    fmt: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "tif":
        out_path = os.path.join(out_dir, f"{prefix}_{idx}.tif")
        write_data_to_tif(out_path, full_data, geotrans, projection)
    else:
        out_path = os.path.join(out_dir, f"{prefix}_{idx}.bin")
        data_to_write = full_data if full_data.ndim == 3 else full_data[np.newaxis, :, :]
        data_to_write.tofile(out_path)
    return out_path


def crop_single_image(
    sr_img_path: str,
    mask: np.ndarray,
    out_dir: str,
    patch: int,
    fmt: str,
) -> List[str]:
    fmt = fmt.lower()
    if fmt not in {"bin", "tif"}:
        raise ValueError("output_format must be 'bin' or 'tif'")

    ds = gdal.Open(sr_img_path)
    if ds is None:
        raise RuntimeError(f"Failed to open image: {sr_img_path}")

    im_width = ds.RasterXSize
    im_height = ds.RasterYSize
    im_bands = ds.RasterCount
    im_geotrans = ds.GetGeoTransform()
    im_proj = ds.GetProjection()

    band = ds.GetRasterBand(1)
    dtype_name, numpy_dtype = GDAL2NP_TYPE.get(band.DataType, ("unknown", None))
    if numpy_dtype is None:
        raise ValueError(f"Unsupported GDAL datatype: {band.DataType}")

    left_top = patch // 2 - 1 if patch % 2 == 0 else patch // 2
    right_bottom = patch // 2

    idx_list = get_row_valid_ranges(mask, patch)
    if not idx_list:
        print(f"No valid mask rows for {sr_img_path}, skip.")
        return []

    saved_paths: List[str] = []
    current_idx = 0

    # Pass 1: center-based
    for i in tqdm(idx_list[1:], desc="Loop 1", leave=False):
        (y, x_start_), (_, x_end_) = i
        for x in range(x_start_ + patch, x_end_ + 1, patch):
            if not mask[y, x]:
                continue
            if (x + patch // 2) > x_end_:
                break

            x_start = x - left_top
            y_start = y - left_top
            x_end = x + right_bottom + 1
            y_end = y + right_bottom + 1

            read_x = max(0, x_start)
            read_y = max(0, y_start)
            read_width = min(x_end, im_width) - read_x
            read_height = min(y_end, im_height) - read_y

            if read_width <= 0 or read_height <= 0:
                continue

            if im_bands > 1:
                full_data = np.zeros((im_bands, patch, patch), dtype=numpy_dtype)
                data = ds.ReadAsArray(read_x, read_y, read_width, read_height)
                off_x = read_x - x_start
                off_y = read_y - y_start
                full_data[:, off_y:off_y + read_height, off_x:off_x + read_width] = data
            else:
                full_data = np.zeros((patch, patch), dtype=numpy_dtype)
                data = ds.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                off_x = read_x - x_start
                off_y = read_y - y_start
                full_data[off_y:off_y + read_height, off_x:off_x + read_width] = data

            new_geotrans = list(im_geotrans)
            new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
            new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]

            current_idx += 1
            saved_paths.append(
                _save_patch(full_data, tuple(new_geotrans), im_proj, out_dir, "img", current_idx, fmt)
            )

    # Pass 2: grid-based
    for i in tqdm(idx_list, desc="Loop 2", leave=False):
        (y, x_start_), (_, x_end_) = i
        if y + patch > im_height:
            break
        for x in range(x_start_, x_end_ + 1, patch):
            if y + left_top >= im_height or x + left_top >= im_width:
                continue
            if not mask[y + left_top, x + left_top]:
                continue
            if x + patch > x_end_:
                break

            x_start = x
            y_start = y
            x_end = x + patch
            y_end = y + patch

            read_x = max(0, x_start)
            read_y = max(0, y_start)
            read_width = min(x_end, im_width) - read_x
            read_height = min(y_end, im_height) - read_y

            if read_width <= 0 or read_height <= 0:
                continue

            if im_bands > 1:
                full_data = np.zeros((im_bands, patch, patch), dtype=numpy_dtype)
                data = ds.ReadAsArray(read_x, read_y, read_width, read_height)
                off_x = read_x - x_start
                off_y = read_y - y_start
                full_data[:, off_y:off_y + read_height, off_x:off_x + read_width] = data
            else:
                full_data = np.zeros((patch, patch), dtype=numpy_dtype)
                data = ds.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                off_x = read_x - x_start
                off_y = read_y - y_start
                full_data[off_y:off_y + read_height, off_x:off_x + read_width] = data

            new_geotrans = list(im_geotrans)
            new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
            new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]

            current_idx += 1
            saved_paths.append(
                _save_patch(full_data, tuple(new_geotrans), im_proj, out_dir, "img", current_idx, fmt)
            )

    ds = None

    if fmt == "bin":
        meta_path = os.path.join(out_dir, "raw_bip_meta.json")
        _write_bin_meta_file(
            meta_path=meta_path,
            bands=im_bands,
            height=patch,
            width=patch,
            dtype_name=dtype_name,
            count=len(saved_paths),
            layout="BHW",
            source_image_path=sr_img_path,
        )
    return saved_paths


def main() -> None:
    os.makedirs(output_root, exist_ok=True)
    all_patch_paths: List[str] = []
    for img_path in input_img_paths:
        print(f"Processing: {img_path}")
        mask = Hyperspectral_Image(img_path, True).backward_mask
        img_out_dir = os.path.join(output_root, Path(img_path).stem)
        patches = crop_single_image(img_path, mask, img_out_dir, patch_size, output_format)
        all_patch_paths.extend(patches)
        print(f"Saved {len(patches)} patches to {img_out_dir}")

    if all_patch_paths:
        list_path = os.path.join(output_root, ".datasets.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in all_patch_paths:
                f.write(f"{p}\n")
        print(f"Patch list saved to: {list_path}")


if __name__ == "__main__":
    main()
