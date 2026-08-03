import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gdal_utils import random_split_shp_by_area_and_clip

input_tif = r''
input_shp = r''
output_dir = r''

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

patch_size = 9
seed = 42
output_format = 'bin'


def main() -> None:
    random_split_shp_by_area_and_clip(
        tif_path=input_tif,
        shp_path=input_shp,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        patch_size=patch_size,
        seed=seed,
        output_format=output_format,
    )


if __name__ == '__main__':
    main()
