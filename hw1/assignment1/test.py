from raster import *
from utils import *

vb_h, vb_w, im_h, im_w = [500.0, 500.0, 16, 16]
shape = read_svg("tests/test5.svg")[1]
shape_scaler = Scale(vb_h, vb_w, im_h, im_w)

a, b = viewbox_2_img_scales(vb_h, vb_w, im_h, im_w)
print(shape.center, shape.radius)
print(a, b)
shifted_center = (a*shape.center[0], b*shape.center[1])
circle_region_tester = CircleCoverage().create_circle_region_tester(shifted_center, shape.radius, a, b)

print(circle_region_tester((7.5, 5)))