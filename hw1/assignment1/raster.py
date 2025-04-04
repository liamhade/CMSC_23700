'''
Remember that the coordinates for this program are centered
around (0,0) being in the upper-left hand corner.
'''

from typing import Optional, Callable, Tuple, List
import numpy as np
from numpy.typing import NDArray
from utils import *
from shapes import Shape, SVG, Triangle, Line, Circle

'''
Function that uses the point-slope formula to find the y-location of
a value on a line given an x.

Args:
    p1 : One of the points of the line
    p2 : Another point on the line
    x  : Value of the x-axis

Return:
    y : Y-value of the line
'''
def line_y(p1: List[float], p2: List[float], x: float):
    x1, y1 = p1
    x2, y2 = p2
    return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

'''
We don't want to have to look at every point in our viewbox
if our shape is just a small portion of the screen. Thus,
we use a bounding box to restrict the number of pixels
we actually need to examine.'

Args:
    shape : Shape we will be creating a bounding box for.

Return:
    bb : Upper-left and lower-right corner of our shapes bounding box.
'''
def bounding_box(shape: Shape) -> List[List[int]]:
    if shape.type in ["triangle", "line"]:
        xs = list(map(lambda p: int(p[0]), shape.pts))
        ys = list(map(lambda p: int(p[1]), shape.pts))

        # Remember that the points of our SVG can be fractions,
        # while pixel vales can only be integers, so we need to
        # convert float to int.
        return ((min(xs), min(ys)), (max(xs), max(ys)))
    
    elif shape.type == "circle":
        # Upper-left because on our pixel-grid, the origin
        # is in the upper-left.
        upper_left  = shape.center - shape.radius
        lower_right = shape.center + shape.radius
        return (upper_left, upper_left), (lower_right, lower_right)
    else:
        raise ValueError("Unknown shape type.")

'''
The viewbox of our SVG may (and likely does) have different
dimension then the final dimensions of our image. Thus,
we need to convert the (x, y) pixel coordinates of our viewbox
into (x', y') coords in our image, while still maintaing aspect rations.

This function assumes that the viewbox and image both have
their origin at (0,0).

Args:
    viewbox_h : Viewbox height
    viewbox_w : Viewbox width
    img_h     : Image height
    img_w     : Image width
    x         : X-coord for the viewbox pixel
    y         : Y-coord for the viewbox pixel

Return:
    (x', y') : Translated coordinates for the image pixel    
'''
def viewbox_coords_2_img(viewbox_h: int, 
                         viewbox_w: int, 
                         img_h    : int, 
                         img_w    : int, 
                         x        : int,
                         y        : int) -> Tuple[int]:
    x_img = int(x * (img_w / viewbox_w))
    y_img = int(y * (img_h / viewbox_h))
    return (x_img, y_img)

'''
For aliasing / supersampling, we don't just want to know if a pixel
lies inside a specific shape -- we also need to know what portion
of the pixel is inside the shape. This is where the idea of coverage
comes in handy. A pixel that has 50% coverage means that half of it
lies insides the shape.

Args:
    shape : Shape that we're testing the coverage for
    x     : X-coordinate of our pixel
    y     : Y-coordinate of our pixel 

Return:
    coverage : Fraction of the pixel that lies in our shape.
'''
def get_coverage_for_pixel(shape: Shape, x: int, y: int) -> float:
    
    if shape.type == "triangle":
        return get_triangle_coverage_for_pixel(shape, x, y)

    elif shape.type == "circle":
        pass

    elif shape.type == "line":
        pass

    else:
        raise ValueError("not sure what shape was given")

'''
Helper function for get_coverage_of_pixel(), but with a
specific application for checking what the coverage of a pixel is
for a given triangle.

We split each pixel up into 9 points, and check if each of those points
falls inside our outside the region of the triangle.

Args:
    triangle : Triangle object we're using as our boundary
    x        : X-coord of the upper-left corner of our pixel
    y        : Y-coord of the upper-left corner of our pixel

Return:
    coverage : Fraction of the pixel that is within our shape
'''
def get_triangle_coverage_for_pixel(triangle: Triangle, x: int, y: int) -> float: 
    # Splitting the pixel into 9 evenly spaced sample points.
    xs = [x+0.5*i for i in range(3)]
    ys = [y+0.5*i for i in range(3)]
    # Calculating the Cartesion product of our Xs and Ys
    points = [(x,y) for x in xs for y in ys]
    num_points_in_triangle = sum(map(lambda p: check_if_point_in_triangle(triangle, p[0], p[1]), points))

    return num_points_in_triangle / len(points)
'''
Helper function for get_triangle_coverage_of_pixel(). This function checks
if a specific (x, y) point (not pixel) is within the overlap
of our three triangle regions.
'''
def check_if_point_in_triangle(triangle: Triangle, x: float, y: float) -> bool:
    triangle_region_testers = create_triangle_region_testers(triangle)
    return all(map(lambda tester: tester((x,y)), triangle_region_testers))

'''
The bounded region of a triangle can be thought of as the overlap
between three other regions: the "inside" regions of the three sides
of the triangle. This function creates three functions, one for testing if
a point is in each of three "inside" regions.

Args:
    triangle : List of 4 points defining our triangle (4, because the last one is redundant) 

Return:
    region_testers : Three region testers for checking if a point is in each region
'''
def create_triangle_region_testers(triangle: Triangle) -> Callable[List[float], bool]:
    return [create_point_in_triangle_region_function(triangle.pts[i:i+2], triangle.pts) for i in range(3)]

'''
Helper function for create_triangle_region_testers(). This function
creates a funciton to test if a point is within the region 
defined by one of the sides of a triangle. When all three
regions are used in conjuction, we can tell if a point is within
a triangle.

We need the rest of the points (other_points) in addition to our line
so that we know which side of the triangle is "within" the region.

Note: this function has undefined behavaior if all three points of the triangle are in a line.

Args:
    line [(x1, y1), (x2, y2)] : Pair of points defining the line / side of triangle
    triangle_points           : All the points of the triangle.

Return:
    region_tester (List[float] -> bool) : Function that tests if a point is inside the line region
'''
def create_point_in_triangle_region_function(line: NDArray[NDArray[float]], triangle_points: NDArray[NDArray[float]]) -> Callable[NDArray[float], bool]:
    p1, p2 = line
    for x, y in filter(lambda p: ~(p == line).all(axis=1).any(), triangle_points):
        # Triangle point is above the line region,
        # so our "inside" region will be above the line to.
        if line_y(p1, p2, x) < y:
            return lambda p: line_y(p1, p2, p[0]) < p[1]
        # "Inside" region is below the line.
        else:
            return lambda p: line_y(p1, p2, p[0]) >= p[1]
    
    raise ValueError("No (x, y) to iterate over")

def rasterize(
    svg_file: str,
    im_w: int,
    im_h: int,
    output_file: Optional[str] = None,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    antialias: bool = True,
) -> np.ndarray:
    """
    :param svg_file: filename
    :param im_w: width of image to be rasterized
    :param im_h: height of image to be rasterized
    :param output_file: optional path to save numpy array
    :param background: background color, defaults to white (1, 1 ,1)
    :param antialias: whether to apply antialiasing, defaults to True
    :return: a numpy array of dimension (H,W,3) with RGB values in [0.0,1.0]
    """

    background_arr = np.array(background)
    shapes: List[Shape] = read_svg(svg_file)
    img = np.zeros((im_h, im_w, 3))
    img[:, :, :] = background_arr # Initializing the image with a background color
    svg = shapes[0]
    assert isinstance(svg, SVG)

    # the first shape in shapes is always the SVG object with the viewbox sizes
    for shape in shapes[1:]:
        for x,y in bounding_box(shape):
            a = get_coverage_for_pixel(shape, x, y)
            x_img, y_img = viewbox_coords_2_img(x,y)
            img[x,y] = (1-a)*img[x,y] + shape.color*a 

    if output_file:
        save_image(output_file, img)

    return img

if __name__ == "__main__":
    # print(read_svg("tests/test1.svg"))
    rasterize("tests/test1.svg", 128, 128, output_file="your_output.png", antialias=False)
