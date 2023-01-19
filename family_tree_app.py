import os

from helpers import *

st.title("Frame yourself")

with st.sidebar:

    # Image
    RAW_IMAGE_FILEPATH = st.file_uploader('Upload a photo of a single face', type=None, accept_multiple_files=False)

    #with st.container:

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
    image = Image.open(RAW_IMAGE_FILEPATH)
    st.image(image, caption=f'Raw image: {RAW_IMAGE_FILEPATH.name}', width=300)

    # Read image
    image = np.asarray(image)
    image = image.copy()
    swap_R_and_B(image)


    # Extract the face landmarks (face mesh)
    face_landmarks = extract_face_landmarks(image)

    show_mesh(image, face_landmarks)

    # Calculate by how much the face is tilted (leaning left/right).
    # This will determine by how much to correct the image in order
    # to make the face level.
    facial_tilt = calculate_facial_tilt(image, face_landmarks)
    st.metric('Tilt', f'{facial_tilt:.0f}°')

    # Rotate image to make eyes level.
    image = rotate_image_by_angle(image, facial_tilt)

    # Re-extract the face landmarks (face mesh) now that the image has been
    # corrected to make eyes level.
    face_landmarks = extract_face_landmarks(image)

    # Calculate by how much the face is rotated (pointing left/right)
    # This will determine how we frame the portrait to ensure *head* is
    # centred in the frame, not the face.
    facial_rotation = calculate_facial_rotation(image, face_landmarks)
    st.metric('Rotation', f'{facial_rotation:.0f}°')

    # Infer parameters of head: centre and radius. The centre will be used as
    # the focal point of the picture frame.
    r, cx, cy, cz = parameterise_head(image, face_landmarks, facial_rotation)

    # TODO: fill/blur around the edge of the image. This is important for when the mask is applied and 'null' space is left uncovered.

    # Frame the image: apply a mask and crop
    framed_img_arr = frame_head(image, r, (cx, cy, cz), FRAME_SHAPE, PORTRAIT_STYLE, EXPAND)

    # Show framed image on app
    st.image(Image.fromarray(framed_img_arr), caption=f'{PORTRAIT_STYLE} framed in {FRAME_SHAPE}', width=300)

    # Option to download image
    os.makedirs('data/processed_images', exist_ok=True)
    final_img = Image.fromarray(framed_img_arr)
    basename = os.path.basename(RAW_IMAGE_FILEPATH.name)
    filename, extension = os.path.splitext(basename)
    final_img.save(f'data/processed_images/{filename}_framed.png')
    with open(f'data/processed_images/{filename}_framed.png', "rb") as file:
        button = st.download_button(
            label     =  'Download framed image',
            data      =   file,
            mime      =  'image/png',
            file_name = f'{filename}_framed.png',
        )

    # Warn user if there are 'null' values in the framed image
    null_pixels_flag = False
    for i in range(0, len(framed_img_arr)):
        for j in range(0, len(framed_img_arr[0])):
            if framed_img_arr[i][j][:3].sum() == 0 and framed_img_arr[i][j][3].sum() == 255:  # if 1) colour is black and 2) within the frame
                null_pixels_flag = True
                break
    if null_pixels_flag == True:
        st.warning(":warning: Frame includes 'null' pixels. You may want to:\n * make the frame smaller using the options in the sidebar\n * use a different photo where the face is not so close to the edge")

    # st.image(framed_img_arr, caption='Sunrise by the mountains', channels='RGB')
    # imageRGB = cv2.cvtColor(framed_img_arr, cv2.COLOR_BGR2RGB)
    # st.image(imageRGB, caption='Sunrise by the mountains')
    # img = Image.fromarray(imageRGB)
    # st.image(img, caption='Sunrise by the mountains')

    #img.save('fromArray.jpg')



    # @st.cache
    # def load_data(nrows):
    #     data = pd.read_csv(DATA_URL, nrows=nrows)
    #     lowercase = lambda x: str(x).lower()
    #     data.rename(lowercase, axis='columns', inplace=True)
    #     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    #     return data
    #
    # data_load_state = st.text('Loading data...')
    # data = load_data(10000)
    # data_load_state.text("Done! (using st.cache)")
    #
    # if st.checkbox('Show raw data'):
    #     st.subheader('Raw data')
    #     st.write(data)
    #
    # st.subheader('Number of pickups by hour')
    # hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
    # st.bar_chart(hist_values)
    #
    # # Some number in the range 0-23
    # hour_to_filter = st.slider('hour', 0, 23, 17)
    # filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    #
    # st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    # st.map(filtered_data)