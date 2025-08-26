# app.py - Streamlit OpenCV Image Effects (fully commented line-by-line)

# import streamlit for building the web UI
import streamlit as st  # main Streamlit library for UI
# import OpenCV for image processing
import cv2  # OpenCV for image processing operations
# import numpy for array manipulation
import numpy as np  # numeric arrays and conversions
# import PIL Image for easy conversion between OpenCV and displayable images
from PIL import Image  # PIL Image for display / conversion
# import io for in-memory byte streams (for download buttons)
import io  # handle in-memory bytes for downloads
# import a sample image from scikit-image to use when no file is uploaded
from skimage import data  # scikit-image sample datasets (Chelsea etc.)

# configure the Streamlit page title and layout
st.set_page_config(page_title="OpenCV Image Effects", layout="wide")  # set page title and wide layout
# show the main title of the app
st.title("OpenCV Image Effects (Grayscale, Blur & Edge Detection)")  # app heading
# short instruction for the user
st.markdown("Upload an image or use the sample image (Chelsea) to try effects.")  # short description

# file uploader widget for user to upload images (jpg/png)
uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])  # file uploader

# slider to control blur kernel size (must be odd to work correctly with GaussianBlur)
blur_kernel = st.slider("Blur kernel size (odd)", min_value=1, max_value=51, value=15, step=2)  # blur kernel slider
# slider to control the minimum threshold for Canny edge detector
canny_min = st.slider("Canny min threshold", 0, 500, 100)  # Canny min threshold
# slider to control the maximum threshold for Canny edge detector
canny_max = st.slider("Canny max threshold", 0, 500, 200)  # Canny max threshold

# checkbox to toggle grayscale conversion
apply_gray = st.checkbox("Grayscale", value=True)  # toggle grayscale
# checkbox to toggle blur
apply_blur = st.checkbox("Blur", value=True)  # toggle blur
# checkbox to toggle Canny edge detection
apply_canny = st.checkbox("Edge Detection (Canny)", value=True)  # toggle Canny

# create two columns: one for original image, one for processed results
col1, col2 = st.columns(2)  # two-column layout

# helper function: read uploaded file into an OpenCV BGR numpy array
def read_image(file) -> np.ndarray:  # function signature returns numpy array
    image = Image.open(file).convert("RGB")  # open file with PIL and ensure RGB
    arr = np.array(image)[:, :, ::-1]  # convert RGB (PIL) to BGR (OpenCV) by reversing channels
    return arr  # return BGR numpy array

# helper function: convert BGR numpy array back to PIL Image for Streamlit display
def to_pil(img_bgr: np.ndarray) -> Image.Image:  # convert OpenCV BGR array to PIL Image
    img_rgb = img_bgr[:, :, ::-1]  # convert BGR to RGB by reversing channels
    return Image.fromarray(img_rgb)  # create and return a PIL Image

# helper function: create an in-memory PNG bytes object from a PIL Image (for download)
def to_bytes(pil_img):
    buf = io.BytesIO()  # create a BytesIO buffer
    pil_img.save(buf, format="PNG")  # save PIL image into buffer as PNG
    buf.seek(0)  # rewind buffer to start
    return buf  # return the buffer

# load uploaded image if provided, otherwise use scikit-image sample (cat image)
if uploaded is not None:  # if user uploaded a file
    img_bgr = read_image(uploaded)  # read uploaded file into BGR numpy array
else:  # if no file uploaded, use the sample image
    st.info("No image uploaded. Using sample cat image from scikit-image.")  # inform user
    sample = data.chelsea()  # load sample cat image (RGB numpy array)
    img_bgr = sample[:, :, ::-1]  # convert sample RGB -> BGR for OpenCV consistency

# Display the original image in the left column using container-width (no deprecation warning)
col1.header("Original")  # header for original image section
col1.image(to_pil(img_bgr), use_container_width=True)  # show the original image (PIL) in column 1

# dictionary to store processed results (name -> BGR image)
results = {}  # prepare a dict to collect outputs

# Apply grayscale if checkbox is selected
if apply_gray:  # if user wants grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # convert BGR to gray
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # convert gray back to BGR so display is consistent
    results["Grayscale"] = gray_bgr  # store grayscale image in results dict

# Apply blur if checkbox is selected
if apply_blur:  # if user wants blur
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1  # ensure kernel size is odd
    blur_img = cv2.GaussianBlur(img_bgr, (k, k), 0)  # apply Gaussian blur with the chosen kernel
    results["Blur"] = blur_img  # store blurred image

# Apply Canny edge detection if checkbox is selected
if apply_canny:  # if user wants edge detection
    gray_for_canny = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # convert to gray for Canny
    edges = cv2.Canny(gray_for_canny, canny_min, canny_max)  # compute edges using Canny thresholds
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # convert edges (gray) to BGR for consistent display
    results["Canny Edges"] = edges_bgr  # store edges image

# Loop through results and display each processed image in the right column with a download button
for name, img in results.items():  # iterate over result name and BGR image
    col2.subheader(name)  # subheader for each processed result
    pil = to_pil(img)  # convert BGR image to PIL Image for display and download
    col2.image(pil, use_container_width=True)  # display the processed image using container width
    # create a download button that serves the image PNG bytes to the user
    col2.download_button(
        label=f"Download {name}",  # button label
        data=to_bytes(pil),  # PNG bytes of the image
        file_name=f"{name.replace(' ','_').lower()}.png",  # suggested filename
        mime="image/png"  # mime type for PNG
    )  # end download_button

# If no effects selected, give a small hint to the user
if not results:  # if results dict is empty (no checkboxes selected)
    st.warning("No effects selected. Please tick Grayscale, Blur, or Edge Detection to see results.")  # user hint

