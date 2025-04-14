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
    max_y = int(np.clip(vertices.max(axis=0)[1], 0, img_h))

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

    # Viewport matrix
    m_vp = np.array([
        [im_w/2, 0, 0, im_w/2],
        [0, im_h/2, 0, im_h/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Orthogonal projection matrix
    m_ortho = np.array([
        [2/(r-l), 0, 0, -(r+l)/(r-l)],
        [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, 2/(n-f), -(n+f)/(n-f)],
        [0, 0, 0, 1]
    ])

    # Initializing our empty image
    img = np.zeros((im_h, im_w, 3))

    # Converting each of the vertices into our image viewbox
    for (i1, i2, i3), c in zip(obj.faces, obj.face_colors):
        # Grabbing the three vertices of our triangle in their orthonormal form.
        # Each one is a numpy array of length 3, but we need to add our homogenous
        # coordinate w.
        v1, v2, v3 = obj.vertices[i1], obj.vertices[i2], obj.vertices[i3]
        # print(v1, v2, v3)
        # Adding homegenous coordinates
        v1 = np.append(v1, [1])
        v2 = np.append(v2, [1])
        v3 = np.append(v3, [1])
        # print("space")
        # print(v1, v2, v3)

        # Translating our coordinates using the viewport and image matrices.
        # The output will be a (4,1) array, as seen on page 141 of the textbook.
        v1 = m_vp @ m_ortho @ v1
        v2 = m_vp @ m_ortho @ v2
        v3 = m_vp @ m_ortho @ v3

        # Only keeping the x and y coordinates of our triangle vertices.
        # Now the array is of shape (2,)
        v1 = v1[:2]
        v2 = v2[:2]
        v3 = v3[:2]

        triangle_image_vertices = np.array([v1, v2, v3])

        (min_x, min_y), (max_x, max_y) = triangle_bounding_box(im_h, im_w, triangle_image_vertices)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                # Checking the middle of the pixel
                if point_in_triangle(np.array([x + 0.5, y + 0.5]), v1, v2, v3):
                    img[im_h - y, x] = c
                
    return save_image("my_p2.png", img)
    
# P3
def render_camera(obj: TriangleMesh, im_w: int, im_h: int):
    """Render the orthographic projection of the cube with the specific camera settings"""
    # Calculating the distance of our eye from the image plane
    e = np.array([0.2, 0.2, 1])
    lookat = np.array([0, 0, 0]) 
    g = lookat - e
    t = np.array([0, 1, 0])

    # Calculating u, v, and w values
    w = - g / np.linalg.norm(g)
    u = np.cross(t, w) / np.linalg.norm(np.cross(t, w))
    v = np.cross(w, u)

    # Calculating camera matrix using two (4,4) matrices from the textbook
    m1 = np.array([
        [u[0], u[1], u[2], 0],
        [v[0], v[1], v[2], 0],
        [w[0], w[1], w[2], 0],
        [0, 0 ,0 , 1]
    ])
    m2 = np.array([
        [1, 0, 0, -e[0]],
        [0, 1, 0, -e[1]],
        [0, 0, 1, -e[2]],
        [0, 0, 0, 1],
    ])
    # Camera matrix is a (4,4) array
    m_cam = m1 @ m2

    # Viewbox values
    l = 0
    r = 12
    b = 0
    t = 12
    f = 0
    n = 12

    # Viewport matrix
    m_vp = np.array([
        [im_w/2, 0, 0, im_w/2],
        [0, im_h/2, 0, im_h/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    # Orthogonal projection matrix
    m_ortho = np.array([
        [2/(r-l), 0, 0, -(r+l)/(r-l)],
        [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, 2/(n-f), -(n+f)/(n-f)],
        [0, 0, 0, 1]
    ])

    print("camera")
    print(m_cam)
    print("ortho")
    print(m_ortho)
    print("viewport")
    print(m_vp)

    # Initializing our empty image
    img = np.zeros((im_h, im_w, 3))

    # Converting each of the vertices into our image viewbox
    for (i1, i2, i3), c in zip(obj.faces, obj.face_colors):
        # Grabbing the three vertices of our triangle in their orthonormal form.
        # Each one is a numpy array of length 3, but we need to add our homogenous
        # coordinate w.
        v1, v2, v3 = obj.vertices[i1], obj.vertices[i2], obj.vertices[i3]
        # Adding homegenous coordinates
        v1 = np.append(v1, [1])
        v2 = np.append(v2, [1])
        v3 = np.append(v3, [1])

        # Translating our coordinates using the viewport and image matrices.
        # The output will be a (4,1) array, as seen on page 141 of the textbook.
        v1 = m_vp @ m_ortho @ m_cam @ v1
        v2 = m_vp @ m_ortho @ m_cam @ v2
        v3 = m_vp @ m_ortho @ m_cam @ v3

        # Only keeping the x and y coordinates of our triangle vertices.
        # Now the array is of shape (2,).
        # Also, we need to divide by w here.
        v1 = v1[:2]
        v2 = v2[:2]
        v3 = v3[:2]

        triangle_image_vertices = np.array([v1, v2, v3])

        (min_x, min_y), (max_x, max_y) = triangle_bounding_box(im_h, im_w, triangle_image_vertices)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # Checking the middle of the pixel
                if point_in_triangle(np.array([x + 0.5, y + 0.5]), v1, v2, v3):
                    img[(im_h - 1) - y, x] = c
                
    return save_image("my_p3.png", img)

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
    # render_ortho(ortho_cube, im_w, im_h)
    render_camera(ortho_cube, im_w, im_h)
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
