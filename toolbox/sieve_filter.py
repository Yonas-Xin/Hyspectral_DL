import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from core import Hyperspectral_Image

input_file = r'' 
output_file = r''
remove_size = 25
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_file)
    img.sieve_filtering(output_file, threshold_pixels=remove_size)