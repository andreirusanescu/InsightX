import streamlit as st
from PIL import Image, ImageEnhance
from streamlit_cropper import st_cropper
import os
import numpy as np
from io import BytesIO
from image import MyImage
import cv2
import matplotlib

# Menu and state management
menu = st.sidebar.selectbox("Menu", ["Edit image", "Advanced ML", "Blend"])

if "active_section" not in st.session_state:
	st.session_state.active_section = "Edit image"
if "operation" not in st.session_state:
	st.session_state.operation = None
if "processed_image" not in st.session_state:
	# To store results of each operation
	st.session_state.processed_image = None
if "operation_type" not in st.session_state:
	st.session_state.operation_type = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

def switch_section(section_name):
	st.session_state.active_section = section_name

def reset_uploader():
    st.session_state.file_uploaded = False

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
	if st.session_state.file_uploaded:
		reset_uploader()
	filename = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg"])
	if filename:
		st.session_state.file_uploaded = True
		st.success("File uploaded successfully!")
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

	col1, col2 = st.columns(3)
	if col1.button("Adjust"):
		switch_section("Adjust")
	if col2.button("Filter"):
		switch_section("Filter")

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
				intensity = st.slider("Intensity", 0.0, 3.0, step=0.1)
				denoise_strength = st.slider("Denoise strength", 0, 30, step=1)
				result = myImage.apply_filter("SHARPEN", intensity=intensity, denoise_strength=denoise_strength)
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
elif menu == "Advanced ML":
	st.title("Advanced Machine Learning Algorithms")
	st.write("Choose an algorithm form the list:")
	col1, col2, col3, col4 = st.columns(4)

	if col1.button("SIFT"):
		switch_section("sift")
	elif col2.button("RANSAC"):
		switch_section("ransac")
	elif col3.button("Palm"):
		switch_section("palm")
	elif col4.button("Detect faces"):
		switch_section("detect faces")

	if st.session_state.active_section == "sift":
		st.write("### Upload two images")
		if st.session_state.file_uploaded:
			reset_uploader()
		filename = st.file_uploader("Upload image 1", type=["jpeg", "png"])
		filename2 = st.file_uploader("Upload image 2", type=["jpeg", "png"])
		if filename and filename2:
			st.session_state.file_uploaded = True
			st.success("File uploaded successfully!")
			pil_image1 = Image.open(filename)
			st.image(pil_image1, caption="Uploaded Image 1")
			st.session_state.original_image = pil_image1

			save_dir = "uploads"
			os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

			# Save the file
			file_path1 = os.path.join(save_dir, filename.name)
			with open(file_path1, "wb") as f:
				f.write(filename.getbuffer())

			myImage = Image(file_path1)
			pil_image2 = Image.open(filename2)
			st.image(pil_image2, caption="Uploaded Image 2")
			st.session_state.second_image = pil_image2

			save_dir = "uploads"
			os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

			# Save the file
			file_path2 = os.path.join(save_dir, filename2.name)
			with open(file_path2, "wb") as f:
				f.write(filename2.getbuffer())

			st.session_state.processed_image = None
			nr_matches = st.slider("Number of matches", 0, 100, step=1)
			myImage.sift(file_path1, file_path2, nr_matches, "sift_result.jpeg")
			result = Image.open("sift_result.jpeg")
			st.image(result, caption="SIFT image (auto-save)")
		else:
			st.warning("Please upload two valid images to start editing.")
		
	elif st.session_state.active_section == "ransac":
		st.write("### Upload two images")
		filename = st.file_uploader("Upload image 1", type=["jpeg", "png"])
		filename2 = st.file_uploader("Upload image 2", type=["jpeg", "png"])
		if filename and filename2:
			st.session_state.file_uploaded = True
			pil_image1 = Image.open(filename)
			st.image(pil_image1, caption="Uploaded Image 1")
			st.session_state.original_image = pil_image1

			save_dir = "uploads"
			os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

			# Save the file
			file_path1 = os.path.join(save_dir, filename.name)
			with open(file_path1, "wb") as f:
				f.write(filename.getbuffer())
			myImage1 = MyImage(file_path1)

			pil_image2 = Image.open(filename2)
			st.image(pil_image2, caption="Uploaded Image 2")
			st.session_state.second_image = pil_image2

			save_dir = "uploads"
			os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

			# Save the file
			file_path2 = os.path.join(save_dir, filename2.name)
			with open(file_path2, "wb") as f:
				f.write(filename2.getbuffer())
			myImage2 = MyImage(file_path2)

			_, file_extension = os.path.splitext(filename.name)
			file_extension = file_extension.lstrip(".")
			myImage1.ransac(myImage1.image, myImage2.image, "ransac." + file_extension)
			result = Image.open("ransac." + file_extension)
			st.image(result, caption="Ransac image (auto-save)")
			st.session_state.processed_image = result

			# Provide a download button for the processed image
			if st.session_state.processed_image:
				buffer = BytesIO()
				_, file_extension = os.path.splitext(filename.name)
				file_extension = file_extension.lstrip(".")
				st.session_state.processed_image.save(buffer, format=file_extension.upper())
				buffer.seek(0)

				st.download_button(
					label="Download Image",
					data=buffer,
					file_name=filename.name,
					mime="image/" + file_extension,
				)
		else:	
			st.warning("Please upload two valid images to start editing.")

	elif st.session_state.active_section == "palm":
		st.write("Upload a photo")
		filename = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg"])
		if filename:
			st.session_state.file_uploaded = True
			st.success("File uploaded successfully!")
			pil_image = Image.open(filename)
			st.image(pil_image, caption="Uploaded Image")
			st.session_state.original_image = pil_image

			save_dir = "uploads"
			os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

			# Save the file
			file_path = os.path.join(save_dir, filename.name)
			with open(file_path, "wb") as f:
				f.write(filename.getbuffer())

			myImage = MyImage(file_path)

			st.session_state.processed_image = None
			myImage.find_palm_lines(file_path, "palm_lines.jpeg")
			result = Image.open("palm_lines.jpeg")
			st.image(result, caption="Palm lines image (auto-save)")
			st.session_state.processed_image = result

			# Provide a download button for the processed image
			if st.session_state.processed_image:
				buffer = BytesIO()
				_, file_extension = os.path.splitext(filename.name)
				file_extension = file_extension.lstrip(".")
				st.session_state.processed_image.save(buffer, format="JPEG")
				buffer.seek(0)

				st.download_button(
					label="Download Image",
					data=buffer,
					file_name=filename.name,
					mime="image/" + file_extension,
				)
		else:
			st.warning("Please upload an image to start editing.")
			st.stop()
	elif st.session_state.active_section == "detect faces":
		st.title("Instructions for proper face detection")
		st.write("### Upload multiple images of the same person. These will be used for training.")
		st.write("### Then, upload a single image. The algorithm will check if this person is the same as the people you entered.")
		name = st.text_area("Enter the name of the person below:", 
							value="",
							placeholder="Type something...",
							height=80)

		if name:
			uploaded_files = st.file_uploader("Upload multiple image files",
										type=[ "jpeg", "png"], 
										accept_multiple_files=True)
			if uploaded_files:
				save_directory = name
				if not os.path.exists(save_directory):
					os.makedirs(save_directory)

				for uploaded_file in uploaded_files:
					save_path = os.path.join(save_directory, uploaded_file.name)
					with open(save_path, "wb") as f:
						f.write(uploaded_file.getbuffer())
				st.success("All files processed and saved!")
				filename = st.file_uploader("Upload a file", type=["png", "jpeg"])
				if filename:
					st.session_state.file_uploaded = True
					pil_image = Image.open(filename)
					st.image(pil_image, caption="Uploaded Image")
					st.session_state.original_image = pil_image

					save_dir = "uploads"
					os.makedirs(save_dir, exist_ok=True)

					file_path = os.path.join(save_dir, filename.name)
					with open(file_path, "wb") as f:
						f.write(filename.getbuffer())

					myImage = MyImage(file_path)

					st.session_state.processed_image = None
					result = myImage.detect_face(save_directory, filename.name, file_path)
					if result is None:
						st.write("### No known faces detected in this photo")
					else:
						cv2_image_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
						detect_image = Image.fromarray(cv2_image_rgb)
						st.image(detect_image, caption="Detect faces image (auto-save)")
						st.session_state.processed_image = detect_image

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
				else:
					st.warning("Please upload an image to start.")
					st.stop()
			else:
				st.write("No files uploaded yet.")

elif menu == "Blend":
	st.title("Blend two images")
	st.write("### Upload two images")
	filename = st.file_uploader("Upload image 1", type=["jpeg", "png"])
	filename2 = st.file_uploader("Upload image 2", type=["jpeg", "png"])
	if filename and filename2:
		st.session_state.file_uploaded = True
		st.success("File uploaded successfully!")
		pil_image1 = Image.open(filename)
		st.image(pil_image1, caption="Uploaded Image 1")
		st.session_state.original_image = pil_image1

		save_dir = "uploads"
		os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

		# Save the file
		file_path = os.path.join(save_dir, filename.name)
		with open(file_path, "wb") as f:
			f.write(filename.getbuffer())

		myImage1 = MyImage(file_path)

		pil_image2 = Image.open(filename2)
		st.image(pil_image2, caption="Uploaded Image 2")
		st.session_state.second_image = pil_image2

		save_dir = "uploads"
		os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

		# Save the file
		file_path = os.path.join(save_dir, filename2.name)
		with open(file_path, "wb") as f:
			f.write(filename2.getbuffer())

		st.session_state.processed_image = None
		alpha = st.slider("Alpha", 0.0, 1.0, value=0.5, step=0.1)
		if not (0.0 < alpha < 1.0) or round(alpha + 1.0 - alpha, 5) != 1.0:
			st.warning("Alpha should be strictly between 0.0 and 1.0.")
		else:
			result = myImage1.blend(file_path, alpha, 1.0 - alpha)
			result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
			blend_image = Image.fromarray(result_rgb)
			st.image(blend_image, use_container_width=True)

			st.session_state.processed_image = blend_image
		if st.session_state.processed_image:
			buffer = BytesIO()
			_, file_extension = os.path.splitext(filename.name)
			file_extension = file_extension.lstrip(".")
			st.session_state.processed_image.save(buffer, format="JPEG")
			buffer.seek(0)

			st.download_button(
				label="Download Image",
				data=buffer,
				file_name=filename.name,
				mime="image/" + file_extension,
			)
	else:
		st.warning("Please upload two valid images to start editing.")
		st.stop()

