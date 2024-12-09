import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import os
import cv2
import numpy as np
from io import BytesIO

# Menu and state management
menu = st.sidebar.selectbox("Menu", ["Edit image", "About", "Contact"])

if "active_section" not in st.session_state:
    st.session_state.active_section = "Edit image"
if "crop_finished" not in st.session_state:
    st.session_state.crop_finished = False
if "rotated_image" not in st.session_state:
    st.session_state.rotated_image = None
if "cropped_image" not in st.session_state:
    st.session_state.cropped_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "operation" not in st.session_state:
    st.session_state.operation = None  # Track the current operation

def switch_section(section_name):
    st.session_state.active_section = section_name

# Function to rotate image
def rotate_image(image, angle):
    # Rotate the image by the given angle
    rotated_image = image.rotate(angle, expand=True)  # `expand=True` ensures the image size adjusts
    return rotated_image

if menu == "Edit image":
    st.title("Upload an image")
    st.write("Upload the desired image for processing!")
    filename = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if filename:
        pil_image = Image.open(filename)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        st.image(image, caption="Uploaded Image", channels="BGR")

        save_path = os.path.join("uploads", filename.name)

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        with open(save_path, "wb") as f:
            f.write(filename.getbuffer())
        st.session_state["original_image"] = pil_image  # Save the original image in session state
    else:
        st.warning("Please upload an image to start editing.")
        st.stop()

    st.write("### Select an Edit Option:")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Adjust"):
        switch_section("Adjust")
    if col2.button("Filter"):
        switch_section("Filter")
    if col3.button("Utils"):
        switch_section("Utils")
    if col4.button("Advanced"):
        switch_section("Advanced")

    if st.session_state.active_section == "Adjust":
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        if col1.button("Crop"):
            st.session_state.operation = "crop"
        elif col2.button("Rotate"):
            st.session_state.operation = "rotate"
        # Crop functionality
        if st.session_state.operation == "crop":
            st.session_state.cropped_image = None
            st.session_state.crop_finished = False
            st.session_state.rotated_image = None

        if st.session_state.operation == "crop" and not st.session_state.crop_finished:
            if "original_image" in st.session_state and st.session_state.original_image is not None:
                st.write("### Select cropped area")
                cropped_image = st_cropper(
                    st.session_state.original_image,
                    aspect_ratio=None,
                )

                st.write("### Live Cropped Preview")
                st.image(cropped_image, use_container_width=True)

                if st.button("Finish Cropping"):
                    st.session_state.cropped_image = cropped_image
                    st.session_state.crop_finished = True

        if st.session_state.crop_finished and st.session_state.cropped_image is not None:
            st.write("### Final Cropped Image")
            st.image(st.session_state.cropped_image, use_container_width=True)

            buffer = BytesIO()
            _, file_extension = os.path.splitext(filename.name)
            file_extension = file_extension.lstrip(".")

            st.session_state["cropped_image"].save(buffer, format=file_extension.upper())
            buffer.seek(0)

            st.download_button(
                label="Download Cropped Image",
                data=buffer,
                file_name="cropped_image." + file_extension,
                mime="image/" + file_extension,
            )
        
        # Rotate functionality
        if st.session_state.operation == "rotate":
            st.session_state.cropped_image = None  # Reset crop content
            st.session_state.crop_finished = False  # Reset crop status
            st.session_state.rotated_image = None  # Clear any previous rotated image

            st.write("### Rotate Image")
            rotate_angle = st.slider(
                "Select rotation angle (in degrees)",
                min_value=0,
                max_value=360,
                value=0,
                step=1
            )

            if "cropped_image" in st.session_state and st.session_state["cropped_image"] is not None:
                rotated_image = rotate_image(st.session_state["cropped_image"], rotate_angle)
                st.session_state.rotated_image = rotated_image
            else:
                rotated_image = rotate_image(pil_image, rotate_angle)
                st.session_state.rotated_image = rotated_image

            # Display rotated image preview
            st.write("### Rotated Image Preview")
            st.image(st.session_state.rotated_image, use_container_width=True)

            # Provide a download button for the rotated image
            buffer = BytesIO()
            _, file_extension = os.path.splitext(filename.name)
            file_extension = file_extension.lstrip(".")
            st.session_state.rotated_image.save(buffer, format=file_extension.upper())
            buffer.seek(0)

            st.download_button(
                label="Download Rotated Image",
                data=buffer,
                file_name="rotated_image." + file_extension,
                mime="image/" + file_extension,
            )


elif menu == "About":
    st.title("About Us")
    st.write("Information about this app.")
elif menu == "Contact":
    st.title("Contact Us")
    st.write("Here's how you can reach us!")
