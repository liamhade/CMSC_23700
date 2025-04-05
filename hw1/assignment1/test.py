from raster import *
from utils import *

line = read_svg("tests/custom.svg")[1] 
print(line)

# points = LineCoverage().get_line_corners(line)

# l = points[0:2]
# LineCoverage().create_point_in_line_region_function(l, points)
# line_corners = LineCoverage().get_line_corners(line)
# region_testers = LineCoverage().create_line_region_testers(line)

# # bb = bounding_box(line)
# # print(bb[0], bb[-1])
# # print(LineCoverage().check_if_point_in_line(192, 150, region_testers))

# print(line_corners)
# LineCoverage().get_line_coverage_for_pixel(line, 210, 143, region_testers)

# print(LineCoverage().check_if_point_in_line(210, 144, region_testers))
# '''
# 210 143 0.0
# 210 144 0.1111111111111111
# 210 145 0.7777777777777778
# '''