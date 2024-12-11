import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper
import os
import numpy as np
from io import BytesIO
from image import MyImage
import cv2

# Menu and state management
menu = st.sidebar.selectbox("Menu", ["Edit image", "About", "Contact"])

if "active_section" not in st.session_state:
    st.session_state.active_section = "Edit image"
if "operation" not in st.session_state:
    st.session_state.operation = None
if "processed_image" not in st.session_state:
    # To store results of each operation
    st.session_state.processed_image = None
if "operation_type" not in st.session_state:
    st.session_state.operation_type = None

def switch_section(section_name):
    st.session_state.active_section = section_name

# Utility functions
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

def flip_image(image, direction):
    if direction == "Horizontal":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "Vertical":
        return image.transpose(Image.FLIP_TOP_BOTTOM)

def resize_image(image, width, height):
    return image.resize((width, height))

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

if menu == "Edit image":
    st.title("Upload an image")
    filename = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if filename:
        pil_image = Image.open(filename)
        st.image(pil_image, caption="Uploaded Image")
        st.session_state.original_image = pil_image

        save_dir = "uploads"
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Save the file
        file_path = os.path.join(save_dir, filename.name)
        with open(file_path, "wb") as f:
            f.write(filename.getbuffer())

        # Get the absolute path
        absolute_path = os.path.abspath(file_path)
        myImage = MyImage(file_path)
    else:
        st.warning("Please upload an image to start editing.")
        st.stop()

    st.write("### Select an Edit Option:")

    col1, col2, col3 = st.columns(3)
    if col1.button("Adjust"):
        switch_section("Adjust")
    if col2.button("Filter"):
        switch_section("Filter")
    if col3.button("Advanced"):
        switch_section("Advanced")

    if st.session_state.active_section == "Adjust":
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        if col1.button("Crop"):
            st.session_state.operation = "crop"
        elif col2.button("Rotate"):
            st.session_state.operation = "rotate"
        elif col3.button("Flip"):
            st.session_state.operation = "flip"
        elif col4.button("Resize"):
            st.session_state.operation = "resize"
        elif col5.button("Brightness"):
            st.session_state.operation = "brightness"
        elif col6.button("Contrast"):
            st.session_state.operation = "contrast"

        # Reset the processed image for new operation
        st.session_state.processed_image = None

        if st.session_state.operation == "crop":
            st.write("### Crop Image")
            cropped_image = st_cropper(st.session_state.original_image)
            st.image(cropped_image, caption="Cropped Image")

            if st.button("Finish Cropping"):
                st.session_state.processed_image = cropped_image

        elif st.session_state.operation == "rotate":
            st.write("### Rotate Image")
            angle = st.slider("Rotation angle", 0, 360, 0)
            
            result = myImage.rotate(angle)
            cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            rotated_image = Image.fromarray(cv2_image_rgb)
            st.image(rotated_image, use_container_width=True)

            if st.button("Apply Rotation"):
                st.session_state.processed_image = rotated_image

        elif st.session_state.operation == "flip":
            st.write("### Flip Image")
            flip_direction = st.selectbox("Select direction", ["Horizontal", "Vertical"])
            flipped_image = flip_image(st.session_state.original_image, flip_direction)
            st.image(flipped_image, caption="Flipped Image")

            if st.button("Apply Flip"):
                st.session_state.processed_image = flipped_image

        elif st.session_state.operation == "resize":
            st.write("### Resize Image")
            width = st.number_input("Width", min_value=1, value=st.session_state.original_image.width)
            height = st.number_input("Height", min_value=1, value=st.session_state.original_image.height)
            resized_image = resize_image(st.session_state.original_image, int(width), int(height))
            st.image(resized_image, caption="Resized Image")

            if st.button("Apply Resize"):
                st.session_state.processed_image = resized_image

        elif st.session_state.operation == "brightness":
            st.write("### Adjust Brightness")
            brightness_factor = st.slider("Brightness factor", 0.1, 3.0, 1.0)
            brightened_image = adjust_brightness(st.session_state.original_image, brightness_factor)
            st.image(brightened_image, caption="Brightened Image")

            if st.button("Apply Brightness"):
                st.session_state.processed_image = brightened_image

        elif st.session_state.operation == "contrast":
            st.write("### Adjust Contrast")
            contrast_factor = st.slider("Contrast factor", 0.1, 3.0, 1.0)
            contrasted_image = adjust_contrast(st.session_state.original_image, contrast_factor)
            st.image(contrasted_image, caption="Contrasted Image")

            if st.button("Apply Contrast"):
                st.session_state.processed_image = contrasted_image

        # Provide a download button for the processed image
        if st.session_state.processed_image:
            buffer = BytesIO()
            _, file_extension = os.path.splitext(filename.name)
            file_extension = file_extension.lstrip(".")
            st.session_state.processed_image.save(buffer, format="PNG")
            buffer.seek(0)

            st.download_button(
                label="Download Image",
                data=buffer,
                file_name=filename.name,
                mime="image/" + file_extension,
            )
    elif st.session_state.active_section == "Filter":
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Apply filters"):
            st.session_state.operation = "apply filters"
        elif col2.button("Grayscale"):
            st.session_state.operation = "grayscale"
        elif col3.button("Equalize"):
            st.session_state.operation = "equalize"
        elif col4.button("Unblur"):
            st.session_state.operation = "unblur"
        
        st.session_state.processed_image = None

        if st.session_state.operation == "apply filters":
            col1, col2, col3, col4 = st.columns(4)
            if col1.button("Blur"):
                st.session_state.operation_type = "blur"
            elif col2.button("Median Blur"):
                st.session_state.operation_type = "median blur"
            elif col3.button("Sharpen"):
                st.session_state.operation_type = "sharpen"
            elif col4.button("Edge"):
                st.session_state.operation_type = "edge"

            if st.session_state.operation_type == "blur":
                kernel_size = st.slider("Kernel size", 1, 50, step=2)
                sigmaX = st.slider("Standard deviation for X", 1, 50, step=2)
                result = myImage.apply_filter("BLUR", kernel_size=kernel_size, sigmaX=sigmaX)
                if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    cv2_image_rgb = result
                filtered_image = Image.fromarray(cv2_image_rgb)
                st.image(filtered_image, use_container_width=True)

                st.session_state.processed_image = filtered_image
            if st.session_state.operation_type == "median blur":
                kernel_size = st.slider("Kernel size", 1, 50, step=2)
                result = myImage.apply_filter("MEDIAN BLUR", kernel_size=kernel_size)
                if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    cv2_image_rgb = result
                filtered_image = Image.fromarray(cv2_image_rgb)
                st.image(filtered_image, use_container_width=True)

                st.session_state.processed_image = filtered_image
            if st.session_state.operation_type == "sharpen":
                result = myImage.apply_filter("SHARPEN", 0, 0)
                if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    cv2_image_rgb = result
                filtered_image = Image.fromarray(cv2_image_rgb)
                st.image(filtered_image, use_container_width=True)

                st.session_state.processed_image = filtered_image
            if st.session_state.operation_type == "edge":
                result = myImage.apply_filter("EDGE", 0, 0)
                if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    cv2_image_rgb = result
                filtered_image = Image.fromarray(cv2_image_rgb)
                st.image(filtered_image, use_container_width=True)

                st.session_state.processed_image = filtered_image
        elif st.session_state.operation == "grayscale":
            result = myImage.gray_scale()
            if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                cv2_image_rgb = result
            grayscale_image = Image.fromarray(cv2_image_rgb)
            st.image(grayscale_image, use_container_width=True)

            st.session_state.processed_image = grayscale_image
        elif st.session_state.operation == "equalize":
            result = myImage.equalize()
            equalize_image = Image.fromarray(result)
            st.image(equalize_image, use_container_width=True)
            st.session_state.processed_image = equalize_image
        elif st.session_state.operation == "unblur":
            result = myImage.unblur()
            if len(result.shape) == 3:
                    cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            else:
                cv2_image_rgb = result
            unblur_image = Image.fromarray(cv2_image_rgb)
            st.image(unblur_image, use_container_width=True)

            st.session_state.processed_image = unblur_image
        if st.session_state.processed_image:
            buffer = BytesIO()
            _, file_extension = os.path.splitext(filename.name)
            file_extension = file_extension.lstrip(".")
            st.session_state.processed_image.save(buffer, format="PNG")
            buffer.seek(0)

            st.download_button(
                label="Download Image",
                data=buffer,
                file_name=filename.name,
                mime="image/" + file_extension,
            )
            
    

elif menu == "About":
    st.title("About Us")
    st.write("Information about this app.")
elif menu == "Contact":
    st.title("Contact Us")
    st.write("Here's how you can reach us!")
