'''
Remember that the coordinates for this program are centered
around (0,0) being in the upper-left hand corner.

TODO: Be able to turn off anti-aliasing
TODO: Fix anti-aliasing
TODO: Make get_<shape>_coverage_for_pixel a single, generalizable function
'''

from typing import Optional, Callable, Tuple, List
import numpy as np
from numpy.typing import NDArray
from itertools import product
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
    bb : Pixels contained in our bounding box
'''
def bounding_box(shape: Shape) -> List[List[int]]:
    if shape.type in ["triangle", "line"]:
        if shape.type == "triangle":
            pts = shape.pts
        elif shape.type == "line":
            pts = LineCoverage().get_line_corners(shape)

        # Remember that the points of our SVG can be fractions,
        # while pixel vales can only be integers, so we need to
        # convert float to int.
        xs = list(map(lambda p: int(p[0]), pts))
        ys = list(map(lambda p: int(p[1]), pts))
        
        # Finding the extreme coordinates of our shape.
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        xs_bb = range(min_x, max_x + 1)
        ys_bb = range(min_y, max_y + 1)

        # Cartesion product of our X and Y values.
        return list(product(xs_bb, ys_bb))

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
    
    # Calculating aspected rations for viewbox and image
    vb_ratio  = viewbox_w / viewbox_h
    img_ratio = img_w / img_h

    # Image is wider than the viewbox 
    if img_ratio > vb_ratio:
        # Need to pad our x-values to keep everything centered
        scale = img_h / viewbox_h
        pad_x = (img_w - viewbox_w * scale) / 2
        pad_y = 0
    # Image (might be) taller than viewbox
    else:
        # Need to pad our y-values to keep everything centered
        scale = img_w / viewbox_w
        pad_x = 0
        pad_y = (img_h - viewbox_h * scale) / 2

    # Maintaining a uniform scale
    x_img = int(x * scale + pad_x)
    y_img = int(y * scale + pad_y)

    return (x_img, y_img)

'''
For aliasing / supersampling, we don't just want to know if a pixel
lies inside a specific shape -- we also need to know what portion
of the pixel is inside the shape. This is where the idea of coverage
comes in handy. A pixel that has 50% coverage means that half of it
lies insides the shape.

Args:
    shape     : Shape that we're testing the coverage for
    x         : X-coordinate of our pixel
    y         : Y-coordinate of our pixel 
    antialias : Whether to apply anti-aliasing to our coverage
    triange_region_testers : List of functions for checking if a point is inside our triangle
    line_region_testers    : List of functions for checking if a point is inside our line 

Return:
    coverage : Fraction of the pixel that lies in our shape.
'''
def get_coverage_for_pixel(shape: Shape, 
                        x: int, 
                        y: int, 
                        antialias: bool,
                        triangle_region_testers: Callable[List[float], bool],
                        line_region_testers: Callable[List[float], bool]) -> float:
    
    if shape.type == "triangle":
        return TriangleCoverage().get_triangle_coverage_for_pixel(x, y, antialias, triangle_region_testers)

    elif shape.type == "circle":
        pass

    elif shape.type == "line":
        return LineCoverage().get_line_coverage_for_pixel(x, y, antialias, line_region_testers)

    else:
        raise ValueError("not sure what shape was given")

class LineCoverage():
    '''
    Helper function for get_coverage_of_pixel(), but with a
    specific application for checking what the coverage of a pixel is
    for a given line.

    We split each pixel up into 9 points, and check if each of those points
    falls inside our outside the region of the line.

    Args:  
        x              : X-coord of the upper-left corner of our pixel
        y              : Y-coord of the upper-left corner of our pixel
        antialias      : Whether we want to apply anti-aliasing to our image generation
        region_testers : Function for checking if a point is within one of the three "regions" of a triangle
    Return:
        coverage : Fraction of the pixel that is within our shape
    '''
    def get_line_coverage_for_pixel(self, x: int, y: int, antialias: bool, region_testers: Callable[List[float], bool]) -> float: 
        if antialias:
            # Splitting the pixel into 9 evenly spaced sample points.
            xs = [x+0.5*i for i in range(3)]
            ys = [y+0.5*i for i in range(3)]
            # Calculating the Cartesion product of our Xs and Ys
            points = [(x,y) for x in xs for y in ys]

            num_points_in_line = sum(map(lambda p: self.check_if_point_in_line(p[0], p[1], region_testers), points))

            return num_points_in_line / len(points)
        else:
            # Checking if the upper-left corner of our pixel is contained in the region
            return float(self.check_if_point_in_line(x, y, region_testers))


    '''
    Helper function for get_line_coverage_of_pixel(). This function checks
    if a specific (x, y) point (not pixel) is within the overlap
    of our four line regions.
    '''
    def check_if_point_in_line(self, x: float, y: float, line_region_testers: Callable[List[float], bool]) -> bool:
        return all(map(lambda tester: tester((x,y)), line_region_testers))

    '''
    A Line() object is define by the location of its ends,
    and its width. To find the region bounded by the line, we need
    to figure out where the corners of the line are.

    Args:
        line : Line() object

    Return:
        corners : Set of four (x,y) coordinates for each corner of the line.
    '''
    def get_line_corners(self, line: Line) -> List[List[float]]:
        (x1, y1), (x2, y2) = line.pts
        dy, dx = y2-y1, x2-x1
        # Angle of our line
        theta = np.arctan(dy/dx)
        dx_prime = np.sin(theta) * (line.width / 2)
        dy_prime = np.cos(theta) * (line.width / 2)

        '''
        Now that we know how far each corner is from the middle of each of the
        endpoints of our line, we can begin solving for the locations of the 
        four corners of the line.

        Note: We want the corners of the line to rotate in 
        one direction, otherwise our boundaries will get messed up.
        '''

        # Line corners nearest the first edge point
        p1 = (x1 - dx_prime, y1 + dy_prime)
        p2 = (x1 + dx_prime, y1 - dy_prime)

        # Line corners nearest the second edge point
        p3 = (x2 + dx_prime, y2 - dy_prime)
        p4 = (x2 - dx_prime, y2 + dy_prime)

        return np.array([p1, p2, p3, p4])

    '''
    The bounded region of a line with a width (w) can be thought of as the overlap
    between four regions marcated by the 4 sides of our line. This function creates 
    four functions, one for testing if a point is in each of the "inside" regions.

    Args:
        line : Line object

    Return:
        region_testers : Four region testers for checking if a point is in each region
    '''
    def create_line_region_testers(self, line: Line) -> Callable[List[float], bool]:
        points = self.get_line_corners(line)
        # Adding the start point to the end for ease of computing
        # the final boundary edge of our line.
        points = np.append(arr=points, values=[points[0]], axis=0)
        return [self.create_point_in_line_region_function(points[i:i+2], points) for i in range(4)]

    '''
    Helper function for create_line_region_testers(). This function
    creates a funciton to test if a point is within the region 
    defined by one of the sides of a line.

    We need the rest of the points (other_points) in addition to our line
    so that we know which side of the line is "within" the region.

    Args:
        line         : Pair of points defining an edge of our "line" object
        line_points  : All four  points of the line

    Return:
        region_tester (List[float] -> bool) : Function that tests if a point is inside the line region
    '''
    def create_point_in_line_region_function(self, line: NDArray[NDArray[float]], line_points: NDArray[NDArray[float]]) -> Callable[NDArray[float], bool]:
        p1, p2 = line
        print(f"Creating line using points {p1} and {p2}")
        for x, y in filter(lambda p: ~(p == line).all(axis=1).any(), line_points):
            # Triangle point is above the line region,
            # so our "inside" region will be above the line to.
            if line_y(p1, p2, x) < y:
                return lambda p: line_y(p1, p2, p[0]) < p[1]
            # "Inside" region is below the line.
            else:
                return lambda p: line_y(p1, p2, p[0]) >= p[1]
        
        raise ValueError("No (x, y) to iterate over")


class TriangleCoverage:
    '''
    Helper function for get_coverage_of_pixel(), but with a
    specific application for checking what the coverage of a pixel is
    for a given triangle.

    We split each pixel up into 9 points, and check if each of those points
    falls inside our outside the region of the triangle.

    Args:  
        triangle       : Triangle object we're using as our boundary
        x              : X-coord of the upper-left corner of our pixel
        y              : Y-coord of the upper-left corner of our pixel
        antialias      : Whether to apply anti-aliasing to our image generator
        region_testers : Function for checking if a point is within one of the three "regions" of a triangle
    Return:
        coverage : Fraction of the pixel that is within our shape
    '''
    def get_triangle_coverage_for_pixel(self, x: int, y: int, antialias: bool, region_testers: Callable[List[float], bool]) -> float: 
        if antialias:
            # Splitting the pixel into 9 evenly spaced sample points.
            xs = [x+0.5*i for i in range(3)]
            ys = [y+0.5*i for i in range(3)]
            # Calculating the Cartesion product of our Xs and Ys
            points = [(x,y) for x in xs for y in ys]
            num_points_in_triangle = sum(map(lambda p: self.check_if_point_in_triangle(p[0], p[1], region_testers), points))

            return num_points_in_triangle / len(points)
        else:
            # Checking if the upper-left corner of our pixel is contained in the region
            return float(self.check_if_point_in_triangle(x, y, region_testers))
    
    '''
    Helper function for get_triangle_coverage_of_pixel(). This function checks
    if a specific (x, y) point (not pixel) is within the overlap
    of our three triangle regions.
    '''
    def check_if_point_in_triangle(self, x: float, y: float, triangle_region_testers: Callable[List[float], bool]) -> bool:
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
    def create_triangle_region_testers(self, triangle: Triangle) -> Callable[List[float], bool]:
        if triangle.type != "triangle": return [] # TODO: rework the code so you don't have to do this
        return [self.create_point_in_triangle_region_function(triangle.pts[i:i+2], triangle.pts) for i in range(3)]

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
    def create_point_in_triangle_region_function(self, line: NDArray[NDArray[float]], triangle_points: NDArray[NDArray[float]]) -> Callable[NDArray[float], bool]:
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
    vb_h = shapes[0].h  # Viewbox height
    vb_w = shapes[0].w  # Viewbox width
    img = np.zeros((im_h, im_w, 3))
    img[:, :, :] = background_arr # Initializing the image with a background color
    svg = shapes[0]
    assert isinstance(svg, SVG)

    # the first shape in shapes is always the SVG object with the viewbox sizes
    for shape in shapes[1:]:

        # Creating our region testers here, since otherwise we we would have to create them
        # on the fly for each pixel.
        if shape.type == "triangle":
            triangle_region_testers = TriangleCoverage().create_triangle_region_testers(shape)
            line_region_testers = []
        elif shape.type == "line":
            triangle_region_testers = []
            line_region_testers = LineCoverage().create_line_region_testers(shape)

        for x, y in bounding_box(shape):
            x_img, y_img = viewbox_coords_2_img(vb_h, vb_w, im_h, im_w, x,y)
            a = get_coverage_for_pixel(shape, x, y, antialias, triangle_region_testers, line_region_testers)
            # print(x, y, a)
            img[x_img, y_img] = (1-a)*img[x_img, y_img] + shape.color*a 

    # Rotating the image by 90 degrees,
    # since our origin shifts from the upper-left to the 
    # lower-left during our transformations.
    if shape.type == 'triangle': 
        img = np.rot90(img, k=-1)

    if output_file:
        save_image(output_file, img)

    return img

if __name__ == "__main__":
    # print(read_svg("tests/test1.svg"))

    # Triangle test
    # rasterize("tests/test1.svg", 128, 128, output_file="your_output.png", antialias=True)

    # Line test
    rasterize("tests/test2.svg", 128, 128, output_file="your_output.png", antialias=True)
