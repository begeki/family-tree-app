"""
Helper functions.

These functions have not been designed to be used in other use-cases; they
are merely here to stop main.py from becoming too bloated.
"""

import cv2
import math
import mediapipe as mp
import numpy as np
import os
from   PIL import Image, ImageDraw
import streamlit as st


def swap_R_and_B(image_arr):
    """
    Assuming RGB or RGBA, swap the R and B elements. This is done in-place.
    """
    for i in range(0, len(image_arr)):
        for j in range(0, len(image_arr[0])):
            if len(image_arr[i][j]) == 3:  # if it's a red-green-blue image
                _1, _2, _3 = image_arr[i][j]  # get the 1st, 2nd, 3rd colours
                image_arr[i][j] = _3, _2, _1  # swap the 1st and 3rd colours
            if len(image_arr[i][j]) == 4:  # if it's a red-green-blue-alpha image
                _1, _2, _3, _4 = image_arr[i][j]  # get the 1st, 2nd, 3rd, 4th colours
                image_arr[i][j] = _3, _2, _1, _4  # swap the 1st and 3rd colours


def calculate_facial_tilt(image, face_landmarks):
    """
    Given a set of facial landmarks, calculate the angle by which the
    face is tilted, with zero meaning that the face is level.

    This works by taking two reference points of the face - the LEFT_TEMPLE
    and the RIGHT_TEMPLE. Each point has an (x, y) coordinate. When the face
    is level, the y-coordinates of the LEFT_TEMPLE and RIGHT_TEMPLE will be
    equal. As the face tilts, the y-coordinates will differ.

    By knowing the delta-x and delta-y between the LEFT_TEMPLE and the
    RIGHT_TEMPLE, it is trivial to calculate the angle (with respect to the
    horizintal).

    Note: any symetrical landmarks could feasibly be used for this. The
    reason that the TEMPLE is used as default is because these are the most
    separated points and so gives the least fractional uncertainty.

    Reading
    -------
        Iris centres from
            https://stackoverflow.com/a/74530918/2660406

        Annotated landmark indices
            https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    Parameters
    ----------
        image : cv2.image
            image

        face_landmarks : array of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
            set of 'landmarks' that represent the face. Each value must have
            an `x`, `y`, and `z` attribute (although `z` is not used in the
            calculations).

    Returns
    -------
        facial_tilt : float
            inferred angle (degrees) that face is tilted by. Zero means face
            is level; positive means the right-hand-side of face is higher
            than the left-hand-side (i.e., face is tilted to their left, or
            from 'our' perspective, to the right of image))
    """

    # Get the width of image. This is required to scale the landmark points
    # which are given in a normalised way (fraction of image).
    h, w, __ = image.shape

    # Define the 'landmark' indices that represent key points on the face
    RIGHT_TEMPLE = 127
    LEFT_TEMPLE = 356

    # Retrieve (x, y) of face's left temple
    X_left = face_landmarks.landmark[LEFT_TEMPLE].x * w
    Y_left = face_landmarks.landmark[LEFT_TEMPLE].y * h

    # Retrieve (x, y) of face's right temple
    X_right = face_landmarks.landmark[RIGHT_TEMPLE].x * w
    Y_right = face_landmarks.landmark[RIGHT_TEMPLE].y * h

    # Assuming a face-on view of the head, we now have a right-angled
    # triangle where the hypotemuse is the line that connects the two
    # temples. It follows that the two shorter sides of the triangle is the
    # delta-x and delta-z distances.
    opposite = Y_left - Y_right
    adjacent = X_left - X_right
    facial_tilt = math.degrees(math.atan(opposite / adjacent))

    return facial_tilt


def calculate_facial_rotation(image, face_landmarks):
    """
    Given a set of facial landmarks, calculate the angle by which the
    face is rotated, with zero being full-frontal and negative meaning the
    face is pointed towards the left of the image.

    This works by taking two reference points of the face - the LEFT_TEMPLE
    and the RIGHT_TEMPLE. Each point has an (x, y) coordinate, but also a
    z-coordinate which represents depth 'in/out' of the face.

    When the face is pointed directly ahead, the depth (z) of the LEFT_TEMPLE
    and RIGHT_TEMPLE are the same. As the face starts rotating to the left
    of the image, the distance of the LEFT TEMPLE will increase, while the
    RIGHT_TEMPLE will decrease.

    The difference in depth and the difference in x can be used to calculate
    the angle of rotation.

    Reading
    -------
        Iris centres from
            https://stackoverflow.com/a/74530918/2660406

        Annotated landmark indices
            https://github.com/google/mediapipe/blob/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    Parameters
    ----------
        image : cv2.image
            image

        face_landmarks : array of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
            set of 'landmarks' that represent the face. Each value must have
            an `x`, `y`, and `z` attribute (although `y` is not used in the
            calculations).

    Returns
    -------
        facial_rotation : float
            inferred angle (degrees) that face is rotated by. Zero means face
            is pointed directly forwards; positive means face is pointed to
            the person's left (or from 'our' perspective, to the right of
            image) meaning the right-hand-side of their head is more visible.
    """

    # Get the width of image. This is required to scale the landmark points
    # which are given in a normalised way (fraction of image).
    h, w, __ = image.shape

    # Define the 'landmark' indices that represent key points on the face
    RIGHT_TEMPLE = 127
    LEFT_TEMPLE = 356

    # Retrieve (x, z) of left temple
    X_left = face_landmarks.landmark[LEFT_TEMPLE].x * w
    Z_left = face_landmarks.landmark[LEFT_TEMPLE].z * w

    # Retrieve (x, z) of right temple
    X_right = face_landmarks.landmark[RIGHT_TEMPLE].x * w
    Z_right = face_landmarks.landmark[RIGHT_TEMPLE].z * w

    # Assuming a bird's-eye-view of the head, we now have a right-angled
    # triangle where the hypotemuse is the line that connects the two
    # temples. It follows that the two shorter sides of the triangle is the
    # delta-x and delta-z distances.
    opposite = Z_left - Z_right
    adjacent = X_left - X_right
    facial_rotation = math.degrees(math.atan(opposite / adjacent))

    return facial_rotation


def extract_face_landmarks(image):
    """
    """

    # Initialise
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,  # only detect one face
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return results.multi_face_landmarks[0]  # there's only one face, so take the first set of results


def rotate_image_by_angle(image, angle):
    """
    """
    # code smell:

    # Convert CV2 image to Pillow image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Rotate the PIL image
    rotated_image = img.rotate(angle, expand=1)

    # code smell: need to convert the PIL image back to CV2 image, but currently doing it by saving and reading an image (see below)
    # rotated_image = np.asarray(rotated_image)

    # Save the PIL image locally
    os.makedirs('data/.temp', exist_ok=True)
    rotated_image.convert('RGB').save('data/.temp/_.jpeg')

    # Reload the image using cv2
    rotated_image = cv2.imread('data/.temp/_.jpeg')

    return rotated_image


def parameterise_head(image, face_landmarks, facial_rotation):
    """
    Find the (x,y,z) coordinate of the centre of the head (as opposed to the
    centre of the face). This is important for centreing the entire head in
    a thumbnail image (while centreing the face risks having some of the
    head cropped out). The calculation assume that the head is a sphere, by
    finding the (x,y) coordinate of the NOSE_TIP, and then going 'into' the
    head by the angle by which the face is rotated, we can arrive at the
    centre of the head. The (x,y) coordinates define the centre of the head in
    the image (the z-component can be ignored).
    """

    # Get the width of image. This is required to scale the landmark points
    # which are given in a normalised way (fraction of image).
    h, w, __ = image.shape

    # Define the 'landmark' indices that represent key points on the face
    RIGHT_FACE_SIDE = 127  #234
    LEFT_FACE_SIDE = 356  #454

    # Retrieve (x, y, z) of left side
    X_left = face_landmarks.landmark[LEFT_FACE_SIDE].x * w
    Y_left = face_landmarks.landmark[LEFT_FACE_SIDE].y * h
    Z_left = face_landmarks.landmark[LEFT_FACE_SIDE].z * w

    # Retrieve (x, z) of right side
    X_right = face_landmarks.landmark[RIGHT_FACE_SIDE].x * w
    Y_right = face_landmarks.landmark[RIGHT_FACE_SIDE].y * h
    Z_right = face_landmarks.landmark[RIGHT_FACE_SIDE].z * w

    # Find mid-point between the two sides
    X_mid = (X_left + X_right) / 2
    Y_mid = (Y_left + Y_right) / 2
    Z_mid = (Z_left + Z_right) / 2

    # From the mid-point, we now want to move towards the centre of the head.
    # This distance is approximately 40% of the distance between the two
    # sides.
    Z_delta = 0.2 * (X_left - X_right)
    Y_delta = 0  # TODO: calculation only looks at x- and z-axes which implicitly assumes the face is pointing level. It would be better to also include y-axis (i.e., if face is stooped or looking upwards)
    X_delta = 0.2 * (Z_left - Z_right)

    X_centre = int(X_mid - X_delta)
    Y_centre = int(Y_mid + Y_delta)
    Z_centre = int(Z_mid + Z_delta)

    # Get approximate radius of head.
    r = 6.0 * math.sqrt(X_delta**2 + Z_delta**2)

    return r, X_centre, Y_centre, Z_centre


def frame_head(image, r, centre_head, frame_shape='circle', portrait_style='face', expand=1.0):
    """
    Frame (mask) image to keep just the head.
    """
    cx, cy, cz = centre_head

    # Define whether image will be circle or oval
    assert frame_shape in ('circle', 'oval'), "Must be 'circle' or 'oval'"
    if frame_shape == 'circle':
        # Don't scale the size of circle
        a = 1.0
    elif frame_shape == 'oval':
        # Scale the x-dimension smaller, and the y-dimension bigger
        a = 1.1

    # Define by how much to expand the frame
    r = r * expand

    # Define whether face or bust should appear in image. If 'bust' then bring
    # the centre of the image down a bit, but correct the radius so that the
    # top of image stays in the same place.
    if portrait_style == 'face':
        pass
    elif portrait_style == 'bust':
        cy = cy + 0.4*r
        r = r + 0.4*r

    # Read image in PIL
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Make an empty mask with same dimensions as image
    height, width = img.size
    mask = Image.new('L', [height, width], 0)

    # Draw mask. The mask is an ellipse centred on (cx, cy)
    draw = ImageDraw.Draw(mask)
    draw.pieslice([(cx - r / a, cy - r * a), (cx + r / a, cy + r * a)], 0, 360,
                  fill=255, outline="white")
    img_array = np.array(img)
    mask_array = np.array(mask)

    # Apply mask to original image
    framed_img_arr = np.dstack((img_array, mask_array))

    # Crop image to size of mask
    x1 = max(int(cy - r * a), 0)  # either top of mask, or top of image
    x2 = min(int(cy + r * a)+1, len(framed_img_arr))  # either bottom of mask, or bottom of image
    y1 = max(int(cx - r / a), 0)  # either left of mask, or left of image
    y2 = min(int(cx + r / a)+1, len(framed_img_arr[0]))  # either right of mask, or right of image
    framed_img_arr = framed_img_arr[x1:x2,y1:y2]

    # Code smell! TODO: ensure that cv2.COLOR_BGR2RGB works on deployed streamlit app
    # Each pixel (the i'th, j'th element of the array) consists of a 4-element
    # array - these are the red-green-blue-alpha values. For some reason,
    # cv2.COLOR_BGR2RGB does not convert these properly (in some cases), and
    # so the following section does it 'manually' - i.e., it swaps the 1st and
    # 3rd colours in each pixel.
    # for i in range(0, len(framed_img_arr)):
    #     for j in range(0, len(framed_img_arr[0])):
    #         _1, _2, _3, _4 = framed_img_arr[i][j]  # get the 1st, 2md, 3rd, 4th colours
    #         framed_img_arr[i][j] = _3, _2, _1, _4  # swap the 1st and 3rd colours

    return framed_img_arr


def show_mesh(image, face_landmarks):
    """
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    annotated_image = image.copy()
    swap_R_and_B(annotated_image)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGBA2RGB)

    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,  # mesh of all 468 points
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=1)
        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,  # outlines of facial features
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=3)
        #connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    st.image(Image.fromarray(annotated_image), caption=f'"facemesh"')#, width=300)