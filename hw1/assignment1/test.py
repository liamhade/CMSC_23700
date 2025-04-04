from raster import *
from utils import *

line = read_svg("tests/test2.svg")[1] 

points = LineCoverage().get_line_corners(line)

l = points[0:2]
LineCoverage().create_point_in_line_region_function(l, points)
region_testers = LineCoverage().create_line_region_testers(line)
