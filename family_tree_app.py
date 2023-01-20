from helpers import *

st.set_page_config(
     page_title="Frame yourself | @begeki",
     page_icon=":mirror:",
 )
st.title("Frame yourself")

with st.sidebar:

    # Image
    RAW_IMAGE_FILEPATH = st.file_uploader('Upload a photo of a single face', type=None, accept_multiple_files=False)

    st.header('Frame parameters:')
    col1, col2, col3 = st.columns([4, 3, 3])

    with col1:
        # Size of frame
        EXPAND = st.number_input('Size', min_value=0.5, max_value=2.5, value=1.0, step=0.1, format='%.1f')

    with col2:
        # Shape of frame
        FRAME_SHAPE = st.radio('Frame shape', ['circle', 'oval'], index=0)

    with col3:
        # Style of frame
        PORTRAIT_STYLE = st.radio('Portrait style', ['face', 'bust'], index=0)

    # Show workings/intermediate steps
    SHOW_WORKINGS = st.checkbox('Show workings?')


if RAW_IMAGE_FILEPATH is None:
    st.info(':information_source: Upload a photo containing a single face using the sidebar.')

else:

    # TODO: reorganise to have raw_image > face_mesh > rotated_image > centre of head. And then have the final image underneath.
    col1, col2, col3, col4, col5, col6, col7 = st.columns([5,1,5,1,5,1,5])

    image = Image.open(RAW_IMAGE_FILEPATH)
    with col1:
        st.image(image, caption=f'Input')#, width=300)

    with col2:
        st.text('>')

    # Read image
    image = np.asarray(image)
    image = image.copy()
    swap_R_and_B(image)

    # Extract the face landmarks (face mesh)
    face_landmarks = extract_face_landmarks(image)

    with col3:
        with st.spinner(text="Detecting face..."):
            # TODO: change image to greyscale, and make facemesh more prominent (green?)
            show_mesh(image, face_landmarks)
    with col4:
        st.text('>')

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
    with col5:
        with st.spinner(text="Leveling face..."):
            # TODO: show image of rotated image
            show_mesh(image, face_landmarks)
            pass
    with col6:
        st.text('>')

    # Calculate by how much the face is rotated (pointing left/right)
    # This will determine how we frame the portrait to ensure *head* is
    # centred in the frame, not the face.
    facial_rotation = calculate_facial_rotation(image, face_landmarks)
    # st.metric('Rotation', f'{facial_rotation:.0f}°')

    # Infer parameters of head: centre and radius. The centre will be used as
    # the focal point of the picture frame.
    r, cx, cy, cz = parameterise_head(image, face_landmarks, facial_rotation)

    with col7:
        with st.spinner(text="Inferring centre of head..."):
            # TODO: add image of centre of head's location
            #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = image.copy()
            swap_R_and_B(img)
            img = Image.fromarray(img)
            #image = Image.new('RGBA', (200, 200))
            draw = ImageDraw.Draw(img)
            #draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=None, outline='#00ff00', width=10)
            #draw.line([(cx-r/5, cy-r/5), (cx+r/5, cy+r/5)], fill='#00ff00', width=10)
            #draw.line([(cx-r/5, cy+r/5), (cx+r/5, cy-r/5)], fill='#00ff00', width=10)
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=None, outline='#00ff00', width=10)
            draw.line([(cx, cy-r/5), (cx, cy+r/5)], fill='#00ff00', width=10)
            draw.line([(cx-r/5, cy), (cx+r/5, cy)], fill='#00ff00', width=10)
            draw.line([(cx-r, cy-r/3), (cx+r, cy-r/3)], fill='#00ff00', width=2)
            draw.line([(cx-r, cy+r/3), (cx+r, cy+r/3)], fill='#00ff00', width=2)
            draw.line([(cx-r/3, cy-r), (cx-r/3, cy+r)], fill='#00ff00', width=2)
            draw.line([(cx+r/3, cy-r), (cx+r/3, cy+r)], fill='#00ff00', width=2)
            st.image(img, caption=f'Head centre')  # , width=300)
            pass

    # TODO: fill/blur around the edge of the image. This is important for when
    #  the mask is applied and 'null' space is left uncovered.

    # Frame the image: apply a mask and crop
    # TODO: when frame parameters are changed, only run script from here - the
    #  facemesh does not need to be recalculated. For further details, see:
    #  https://docs.streamlit.io/library/api-reference/performance
    framed_img_arr = frame_head(image, r, (cx, cy, cz), FRAME_SHAPE, PORTRAIT_STYLE, EXPAND)

    # Show framed image on app
    st.success('Subject has successfully detected!')
    st.image(Image.fromarray(framed_img_arr), caption=f'{PORTRAIT_STYLE} framed in {FRAME_SHAPE}', width=300)

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
        :warning: Frame includes 'null' pixels. You may want to:\n * make the \
        frame smaller using the options in the sidebar\n * use a different \
        photo where the face is not so close to the edge
        """)


    # TODO: change colours of landmarks: https://stackoverflow.com/questions/71860084/how-can-i-change-the-color-of-the-lip-that-got-its-landmarks-without-disturbing
