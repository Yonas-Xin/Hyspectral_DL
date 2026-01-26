import sys, os
sys.path.append('.')
from core import Hyperspectral_Image

input_file = r'' 
output_file = r''
remove_size = 64
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_file)
    img.sieve_filtering(output_file, threshold_pixels=remove_size)