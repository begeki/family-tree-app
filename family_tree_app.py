from helpers import *

st.set_page_config(
     page_title="Frame yourself | @begeki",
     page_icon=":mirror:",
 )
st.title("Frame yourself")

@st.cache
def read_image(RAW_IMAGE_FILEPATH):
    image = Image.open(RAW_IMAGE_FILEPATH)
    return image


@st.cache
def detect_face(image):

    # Read image
    image = np.asarray(image)
    image = image.copy()
    swap_R_and_B(image)

    # Extract the face landmarks (face mesh)
    face_landmarks = extract_face_landmarks(image)

    # Overlay facemesh on top of image
    detected_face_image = apply_facemesh(image, face_landmarks)

    return image, face_landmarks, detected_face_image


@st.cache
def level_face(image, face_landmarks):
    # Calculate by how much the face is tilted (leaning left/right).
    # This will determine by how much to correct the image in order
    # to make the face level.
    facial_tilt = calculate_facial_tilt(image, face_landmarks)
    # st.metric('Tilt', f'{facial_tilt:.0f}°')

    # Rotate image to make eyes level.
    image = rotate_image_by_angle(image, facial_tilt)

    # Re-extract the face landmarks (face mesh) now that the image has been
    # corrected to make eyes level.
    face_landmarks = extract_face_landmarks(image)

    # Overlay facemesh on top of image
    levelled_image = apply_facemesh(image, face_landmarks)

    return image, face_landmarks, levelled_image


with st.sidebar:

    # Image
    # TODO: upload image from camera.
    RAW_IMAGE_FILEPATH = st.file_uploader('Upload a photo of a single face', type=None, accept_multiple_files=False)

if RAW_IMAGE_FILEPATH is None:
    st.info(':information_source: Upload a photo containing a single face using the sidebar.')

else:

    # TODO: if input image has been processed, it does not need to be re-processed whenever frame parameters are
    #  changed. Ensure that col1, col2, col3, etc. can still be populated, despite these processes running within
    #  a decorated function.
    # @st.cache
    # def process_input_image(image):
    #     # run all processes up to (but not including) when image is framed
    #     return image, r, (cx, cy, cz)

    # Sequence of images showing how calculations/processing progresses
    st.header('Face detection')
    col1, col2, col3, col4, col5, col6, col7 = st.columns([5,1,5,1,5,1,5])

    with col1:

        # Read image
        image = read_image(RAW_IMAGE_FILEPATH)

        # Display image on app
        st.image(image, caption=f'Image uploaded')

    with col2:
        st.text('>')

    # # Read image
    # image = np.asarray(image)
    # image = image.copy()
    # swap_R_and_B(image)

    with col3:
        with st.spinner(text="Detecting face..."):

            # Detect face
            image, face_landmarks, detected_face_image = detect_face(image)

            # Display image on app
            st.image(detected_face_image, caption=f'Face detected')

    with col4:
        st.text('>')

    with col5:
        with st.spinner(text="Leveling face..."):

            # Make the face level
            image, face_landmarks, levelled_image = level_face(image, face_landmarks)

            # Display image on app
            st.image(levelled_image, caption=f'Eyes levelled')

    with col6:
        st.text('>')

    with col7:
        with st.spinner(text="Locating centroid of head..."):

            # Calculate by how much the face is rotated (pointing left/right)
            # This will determine how we frame the portrait to ensure *head* is
            # centred in the frame, not the face.
            facial_rotation = calculate_facial_rotation(image, face_landmarks)
            # st.metric('Rotation', f'{facial_rotation:.0f}°')

            # Infer parameters of head: centre and radius. The centre will be used as
            # the focal point of the picture frame.
            r, cx, cy, cz = parameterise_head(image, face_landmarks, facial_rotation)

            img = image.copy()
            swap_R_and_B(img)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=None, outline='#00ff00', width=10)
            draw.line([(cx, cy-r/5), (cx, cy+r/5)], fill='#00ff00', width=10)
            draw.line([(cx-r/5, cy), (cx+r/5, cy)], fill='#00ff00', width=10)
            draw.line([(cx-r, cy-r/3), (cx+r, cy-r/3)], fill='#00ff00', width=2)
            draw.line([(cx-r, cy+r/3), (cx+r, cy+r/3)], fill='#00ff00', width=2)
            draw.line([(cx-r/3, cy-r), (cx-r/3, cy+r)], fill='#00ff00', width=2)
            draw.line([(cx+r/3, cy-r), (cx+r/3, cy+r)], fill='#00ff00', width=2)
            st.image(img, caption=f'Head centroid located')

    st.success('Subject has been successfully detected!')

    # TODO: fill/blur around the edge of the image. This is important for when
    #  the mask is applied and 'null' space is left uncovered.

    # Show framed image on app
    st.header('Now customise your frame')
    colA, colB, colC = st.columns([2,1,5])

    with colA:

        # Size of frame
        EXPAND = st.number_input('Size', min_value=0.5, max_value=2.5, value=1.0, step=0.1, format='%.1f')

        # Shape of frame
        FRAME_SHAPE = st.radio('Frame shape', ['circle', 'oval'], index=0)

        # Style of frame
        PORTRAIT_STYLE = st.radio('Portrait style', ['face', 'bust'], index=0)


    with colC:

        # Frame the image: apply a mask and crop
        framed_img_arr = frame_head(image, r, (cx, cy, cz), FRAME_SHAPE, PORTRAIT_STYLE, EXPAND)

        # Display image on app
        st.image(Image.fromarray(framed_img_arr), caption=f'{PORTRAIT_STYLE} framed in {FRAME_SHAPE}')

    # Option to download image
    os.makedirs('data/processed_images', exist_ok=True)
    final_img = Image.fromarray(framed_img_arr)
    basename = os.path.basename(RAW_IMAGE_FILEPATH.name)
    filename, extension = os.path.splitext(basename)
    final_img.save(f'data/processed_images/{filename}_framed.png')
    with open(f'data/processed_images/{filename}_framed.png', "rb") as file:
        button = st.download_button(
            label='Download framed image',
            data=file,
            mime='image/png',
            file_name=f'{filename}_framed.png',
        )

    # Warn user if there are 'null' values in the framed image
    null_pixels_flag = False
    for i in range(0, len(framed_img_arr)):
        for j in range(0, len(framed_img_arr[0])):
            if framed_img_arr[i][j][:3].sum() == 0\
                    and framed_img_arr[i][j][3].sum() == 255:  # if 1) colour is black and 2) within the frame
                null_pixels_flag = True
                break
    if null_pixels_flag is True:
        st.warning("""
        :warning: Frame includes 'null' pixels (i.e., pixels that were off \
        the edge of the original image). You may want to:\n * make the \
        frame smaller using the options in the sidebar\n * use a different \
        photo where the face is not so close to the edge
        """)
