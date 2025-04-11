from typing import Sequence, Optional
import os
import numpy as np
from itertools import product
from PIL import Image


class TriangleMesh:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_colors: Optional[np.ndarray] = None,
        vertex_colors: Optional[np.ndarray] = None,
    ):
        """
        vertices: (n_vertices, 3) float array of vertex positions
        faces: (n_faces, 3) int array of indices into vertices, each row the verts of a triangle
        face_colors: (n_faces, 3) float array of a rgb color (in range [0,1]) per face
        vertex_colors: (n_vertices, 3) float array of a rgb color (in range [0,1]) per vertex
        """
        self.vertices = vertices
        self.faces = faces
        self.face_colors = face_colors
        self.vertex_colors = vertex_colors

def save_image(fname: str, arr: np.ndarray) -> np.ndarray:
    """
    fname: path of where to save the image
    arr: numpy array of shape (H,W,3), and should be between 0 and 1

    saves both the image and an .npy file of the original image array
    and returns back the original array
    """
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im.save(fname)
    np.save(os.path.splitext(fname)[0] + ".npy", arr)
    return arr


def read_image(fname: str) -> np.ndarray:
    """reads image file and returns as numpy array (H,W,3) rgb in range [0,1]"""
    return np.asarray(Image.open(fname)).astype(np.float64) / 255


def point_in_triangle(p: np.array, A: np.array, B: np.array, C: np.array) -> bool:
    """
    Checks if a point lies within a 2D triangle using a Barycentric coordinate
    system.

    Args:
        p : Point whose inclusion we are testing for
        A : 1st triangle vertex
        B : 2nd triangle vertex
        C : 3rd triangle vertex 
    
    Return:
        inside : Boolean indicating whether the point is inside our triangle
    """
    v1 = A - C
    v2 = B - C
    v3 = p - C

    # (2,2) matrix for our triangle
    m = np.column_stack((v1, v2))
    # Solving for the alpha and beta values of our triangle
    try:
        alpha, beta = np.linalg.solve(m, v3) 
    # Triangle with 0 area 
    except np.linalg.LinAlgError:
        return False

    gamma = 1 - alpha - beta

    return (alpha >= 0) and (beta >= 0) and (gamma >= 0)

def triangle_bounding_box(img_h: int, img_w: int, vertices: np.array) -> np.array:
    """
    Finds the binding box for a triangle with vertices v1, v2, and v3.

    Args:
        img_h    : Height of the image
        img_w    : Width of the image
        vertices : Numpy array of the triangle vertices
    Return:
        Lower-left and upper-right corner of the triangle
    """

    min_x = int(np.clip(vertices.min(axis=0)[0], 0, img_w))
    min_y = int(np.clip(vertices.min(axis=0)[1], 0, img_h))
    max_x = int(np.clip(vertices.max(axis=0)[0], 0, img_w))
    max_y = int(np.clip(vertices.max(axis=1)[1], 0, img_h))

    return (min_x, min_y), (max_x, max_y)

# P1
def render_viewport(obj: TriangleMesh, im_w: int, im_h: int):
    """
    Render out just the vertices of each triangle in the input object.
    TIP: Pad the vertex pixel out in order to visualize properly like in the
    handout pdf (but turn that off when you submit your code)
    """
    # Viewport matrix
    m_vp = np.array([
        [im_w/2, 0, 0, im_w/2],
        [0, im_h/2, 0, im_h/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Initializing our empty image
    img = np.zeros((im_h, im_w, 3))
    
    # Each vertex as length 3
    for v in obj.vertices:
        # Viewport matrix has shape (4,4), so we need to add
        # an extra homogenous datapoint for the shapes to line up.
        # Since a vertex is a point, we append c=1.
        v = np.append(v, [1])
        # Applying out transformation matrix to v to get a length 4 array.
        vp = m_vp @ v
        # Dividing by our homogenous coordinate
        vp = vp / vp[-1]
        # Unpacking the image coordinates, and dropping the homogenous coordinate
        x_img, y_img, _, _ = vp
        
        # Rounding to the nearest pixel value
        x_img = round(x_img)
        y_img = round(y_img)

        # TODO: Remove this padding for the final step
        for x in range(x_img-5, x_img+5):
            for y in range(y_img-5, y_img+5):
                img[y, x] = [1,1,1]

    return save_image("my_p1.png", img)

# P2
def render_ortho(obj: TriangleMesh, im_w: int, im_h: int):
    """Render the orthographic projection of the cube"""
    l = 0
    r = 12
    b = 0
    t = 12
    f = 0
    n = 12

    # Orthogonal projection matrix
    m_ortho = [
        [1/6, 0, 0, -1],
        [0, 1/6, 0, -1],
        [0, 0, 1/6, -1],
        [0, 0, 0, 1]
    ]

    # Matric for converting our cube into the scale of our
    # image.
    m_img = [
        [im_w, 0, 0, 0],
        [0, im_h, 0, 0],
        [0, 0, 0, 0],
        [im_w, im_h, 0, 0]
    ]

    # Initializing our empty image
    img = np.zeros((im_h, im_w, 3))

    # Each vertex as length 3
    projected_vertices = []
    for v in obj.vertices:
        # Viewport matrix has shape (4,4), so we need to add
        # an extra homogenous datapoint for the shapes to line up.
        # Since a vertex is a point, we append c=1.
        v = np.append(v, [1])
        # Applying our ortho matrix to v to get a length 4 array.
        vp = m_ortho @ v
        # Dividing by our homogenous coordinate (still a length 4 vector)
        vp = vp / vp[-1]
        
        projected_vertices.append(vp)

    # Converting the projected vertices into a numpy array
    p_v = np.array(projected_vertices)

    for (i1, i2, i3), c in zip(obj.faces, obj.face_colors):
        # Grabbing the three vertices of our triangle in their orthonormal form.
        # Each one is a numpy array of length 4.
        tri_v1_o, tri_v2_o, tri_v3_o = p_v[i1], p_v[i2], p_v[i3]
        # Using the image width and height to translate our vertices
        # into points (not pixels yet) in our image
        v1 = ((tri_v1_o @ m_img) / 2)[:2]
        v2 = ((tri_v2_o @ m_img) / 2)[:2]
        v3 = ((tri_v3_o @ m_img) / 2)[:2]

        triangle_image_verties = np.array([v1, v2, v3])

        (min_x, min_y), (max_x, max_y) = triangle_bounding_box(im_h, im_w, triangle_image_verties)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Checking the middle of the pixel
                if point_in_triangle(np.array([x + 0.5, y + 0.5]), v1, v2, v3):
                    img[y, x] = c
                
    return save_image("my_p2.png", img)
    
# P3
def render_camera(obj: TriangleMesh, im_w: int, im_h: int):
    """Render the orthographic projection of the cube with the specific camera settings"""
    return save_image("p3.png", YOUR_IMAGE_ARRAY_HERE)
    

# P4
def render_perspective(obj: TriangleMesh, im_w: int, im_h: int):
    """Render the perspective projection with perspective divide"""
    return save_image("p4.png", YOUR_IMAGE_ARRAY_HERE)
    


# P5
def render_zbuffer_with_color(obj: TriangleMesh, im_w: int, im_h: int):
    """Render the input with z-buffering and color interpolation enabled"""
    return save_image("p5.png", YOUR_IMAGE_ARRAY_HERE)
    


# P6
def render_big_scene(objlist: Sequence[TriangleMesh], im_w: int, im_h: int):
    """Render a big scene with multiple shapes"""
    return save_image("p6.png", YOUR_IMAGE_ARRAY_HERE)
    


# P7
def texture_map(obj: TriangleMesh, img: np.ndarray, im_w: int, im_h: int):
    """Render a cube with the texture img mapped onto its faces"""
    return save_image("p7.png", YOUR_IMAGE_ARRAY_HERE)
    


def get_big_scene():
    # Cube
    vertices = np.array(
        [
            [-0.35, -0.35, -0.15],
            [-0.15, -0.35, -0.15],
            [-0.35, -0.15, -0.15],
            [-0.15, -0.15, -0.15],
            [-0.35, -0.35, -0.35],
            [-0.15, -0.35, -0.35],
            [-0.35, -0.15, -0.35],
            [-0.15, -0.15, -0.35],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.45, 0.5, 0.35], [0.4, 0.4, 0.45], [0.4, 0.35, 0.25], [0.4, 0.45, 0.3]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    tet1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.0, 0.0, 0.0], [-0.1, -0.3, -0.25], [-0.1, 0.1, 0.3], [-0.1, -0.15, 0.4]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    tet2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    vertices = np.array(
        [
            [-0.4, -0.4, 0.2],
            [-0.5, -0.4, 0.2],
            [-0.4, -0.5, 0.2],
            [-0.5, -0.5, 0.2],
            [-0.4, -0.4, 0.3],
            [-0.5, -0.4, 0.3],
            [-0.4, -0.5, 0.3],
            [-0.5, -0.5, 0.3],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    cube2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    return [cube1, tet1, tet2, cube2]


if __name__ == "__main__":
    im_w = 800
    im_h = 600
    vertices = np.array(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    triangle_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    cube = TriangleMesh(vertices, triangles, triangle_colors)

    # NOTE for your own testing purposes:
    # Uncomment and run each of these commented-out functions after you've filled them out
    # render_viewport(cube, im_w, im_h)

    ortho_vertices = np.array(
        [
            [1.0, 1.0, 1.5],
            [11.0, 1.0, 1.5],
            [1.0, 11.0, 1.5],
            [11.0, 11.0, 1.5],
            [1.0, 1.0, -1.5],
            [11.0, 1.0, -1.5],
            [1.0, 11.0, -1.5],
            [11.0, 11.0, -1.5],
        ]
    )
    ortho_cube = TriangleMesh(ortho_vertices, triangles, triangle_colors)
    render_ortho(ortho_cube, im_w, im_h)
    # render_camera(ortho_cube, im_w, im_h)
    # render_perspective(cube, im_w, im_h)
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube.vertex_colors = vertex_colors
    # render_zbuffer_with_color(cube, im_w, im_h)

    objlist = get_big_scene()
    # render_big_scene(objlist, im_w, im_h)
    img = read_image("bored_ape.jpeg")
    # texture_map(cube, img, im_w, im_h)
