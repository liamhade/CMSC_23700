from raster import *
from utils import *

vb_h, vb_w, im_h, im_w = [500.0, 500.0, 128, 128]
shape = read_svg("tests/test5.svg")[1]
shape_scaler = Scale(vb_h, vb_w, im_h, im_w)

a, b = viewbox_2_img_scales(vb_h, vb_w, im_h, im_w)
print(a, b)
circle_region_tester = CircleCoverage().create_circle_region_tester(shape, a, b)

print(circle_region_tester((0,0)))