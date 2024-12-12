### MAXIM Anca-Stefania 324CC
### RUSANESCU Andrei-Marian 323CC

## [GitHub link](https://github.com/andreirusanescu/PythonProject)

# **Image Editor Application**

## **Introduction**

This Image Editor Application offers a comprehensive set of image processing
and machine learning features. It combines the power of traditional image
manipulation with cutting-edge techniques like facial recognition, object
detection, and image registration. The application provides both a
**graphical user interface (GUI)** and a **text-based interface**, ensuring
flexibility for all types of users. The GUI can be run using
```streamlit run menu.py```, whereas the text-based interface can be run
with the command ```python3 main.py <filename>```.

This software is inspired by the functionalities of well-known photo editing
tools like **Adobe Photoshop**, **GIMP**, and **Paint.NET**, while also
incorporating advanced machine learning-based features that are rarely found
in traditional editors to explore the world of Computer vision.

---

## **Features**

### **1. Image Adjustment Commands**

- **CROP**: Crop the image to a specified region using (x1, y1, x2, y2) coordinates.
- **ROTATE**: Rotate the image by a given angle (in degrees).
- **FLIP**: Flip the image vertically, horizontally, or both.
- **RESIZE**: Resize the image to custom dimensions.
- **BRIGHTNESS**: Adjust the image’s brightness (-100 to 100).
- **CONTRAST**: Adjust the image’s contrast (0.5 to 3.0).

### **2. Filter Commands**

- **APPLY**: Apply a custom filter to the image. Supported filters:
  - **BLUR**: Blur the image with a specified kernel size and standard deviation value.
  - **MEDIAN\_BLUR**: Apply median blur for noise reduction.
  - **SHARPEN**: Sharpen the image with configurable intensity and denoise strength.
  - **EDGE**: Apply edge detection to highlight image boundaries.
- **GRAYSCALE**: Convert the image to grayscale.
- **EQUALIZE**: Apply histogram equalization to enhance image contrast.
- **UNBLUR**: Attempt to restore blurry images using wiener deconvolution.

### **3. Advanced Machine Learning Features**

- **BLEND**: Blend multiple images together.
- **PALM**: Detect and identify palm lines in images or lines in general.
- **SIFT (Scale-Invariant Feature Transform)**: Detect and match key points between two images.
- **RANSAC (Random Sample Consensus)**: Perform image registration to align two images.
- **DETECT\_FACES**: Detect faces in the image using OpenCV’s face detection algorithms.
- **RECOGNIZE\_FACES**: Recognize and identify faces using the LBPH (Local Binary Patterns Histograms) face recognizer.

### **4. Utility Commands**

- **SHOW**: Display the current state of the image.
- **SAVE**: Save the edited image to a specified filename.
- **EXIT**: Exit the application.

---

## **How to Use**

### **Text Interface**

1. Run the application with the following command:
```bash
python3 main.py <filename>
```
2. The program will display a list of available commands.

3. Type the command you wish to execute and follow the prompts to provide any necessary parameters (e.g., angles, dimensions, or filenames).

### **Graphical Interface**

If you prefer an intuitive, visual approach, launch the graphical user interface. Here, you can access all the features using buttons and input fields, similar to tools like GIMP or Paint.NET.

---

## **System Requirements**

- **Python 3.x**
- Required libraries:
  - **OpenCV** (cv2) for image processing and facial recognition.
  - **NumPy** for numerical computations.
  - **Pillow** (PIL) for image manipulation.
  - **Streamlit** for the graphical user interface.
  - **Matplotlib** for image plotting.

* __menu.py__
    - The graphical interface was implemented using the **streamlit** library.
    - The application opens with a selectbox menu located on the left side, presenting three options: __Edit image, Advanced ML, and Blend.__
    - __Edit image__ : A `file_uploader` appears, allowing the user to upload the desired image. This image is saved in a local folder named `uploads`.
    - After uploading the image, two buttons appear on the screen: **Adjust** and **Filter**. These buttons process the current image. At the end of each operation (described below), the user will be able to download the processed image locally (via a download button) while preserving the original format (.jpeg, .png).
    - __Adjust__: This option presents six buttons: **crop, rotate, flip, resize, brightness, contrast**. These perform the respective transformations on the uploaded image using functions from the Pillow library, the `ImageEnhance` module, and custom functions implemented in the `adjust.py` module, attached to the `MyImage` class (`image.py`). The **intensity of the filters** can be adjusted using a `st.slider`.
    - __Filter__: This option presents four buttons: **Apply filters, Grayscale, Equalize, and Unblur**. The functionalities are implemented in `filters.py`. To preserve the original format of the image (RGB or grayscale), `cv2.cvtColor` was used. The **intensity of the filters** can also be adjusted using a `st.slider`.
    - __Advanced ML__ : This section offers four advanced Machine Learning algorithms: **Sift, RANSAC, Palm, Facial Detection**.
    - __Sift__: Detects similarities between two uploaded images and displays them. This feature uses two `st.file_uploader` components and allows users to download the output image. Additionally, the number of matches can be selected using a slider.
    - __RANSAC__ : Takes two images as input and outputs a result that combines the two images optimally. Similar to Sift, it uses two `st.file_uploader` components and includes a download option.
    - __Palm__: Detects contours in the uploaded image. This is particularly useful for detecting palm lines, which appear slightly thickened, but it can also highlight other contours in an image.
    - __Detect faces__: Detects faces in an image. The algorithm implementation is in `detect_faces.py`. It allows users to upload a "database" of photos containing the same person. Afterward, a single image can be uploaded, and the algorithm will decide whether the person in the database is present in the uploaded image.
    A limitation of this algorithm is the relatively small size of the database that the user manually uploads, which may lead to suboptimal results. However, in the final version of the code, the **detect faces** functionality focuses solely on detecting faces in a photo and works correctly.
    - __Blend__: Takes two images as input and outputs the result of blending them based on a parameter called **alpha**, which determines the proportion of the first image in the final result (automatically, `1 - alpha` is the proportion of the second image).
---

<!-- Anca: -->
## detect_faces.py
- The code consists in three functions: __detect_face_rect, detect_face_coord and prepare_training_data__. 
- __Detect_face_rect__ detects a face and draws a rectangle around it. It's not called directly, but its' implementation is used in both detect_face_coord and prepare_train_data.
- __Detect_face_coord__ detects a face and returns its coordinates. The code loads the [__Haar Cascade Frontal Face__](https://github.com/udacity/dog-project/blob/master/haarcascades/haarcascade_frontalface_alt.xml) algorithm in a variable and Then, using that algorithm, it tries to detect the faces and draw a circle over the
detected face. scaleFactor is used to take care of large and small faces. minNeighbors looks at the face detected inside a rectangle
and decides what to include and what to reject.
- __Prepare_train_data__ performs facial recognition. The user provides a folder, containing pictures of the same person.
We first prepare the training data by storing the faces and their coordinates.
Then we begin training the model using face coordinates and labels.
``` python3
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
```
For GUI simplicity, it is implemented to support only one face and one directory
of faces, but it can be extended.
Then, we use
``` python3
label, confidence = face_recognizer.predict(face)
```
to predict as accurately as possible whether the input photo contains the face in the folder of files. The function outputs a confidence level. The greater the "confidence", the smaller the odds that the picture contains the desired face. Unfortunately, for small sets of data, these predictions are faulty.

---

## Implementation Challenges
- **Isolating different states of image processing**: Transitioning between functionalities, "clearing" the changes made to the image, and "resetting" the screen required using `st.session_state.*`. This ensures that the program's current state is global (accessible to all components).
- **Ensuring compatibility** between the manually implemented `MyImage` class and the image format used by Streamlit, achieved by converting between arrays and images.

---

<!-- Andrei -->
## adjust.py & filters.py
Most of the logic in adjust.py and filters.py has already been implemented without using high level libraries like cv2 [here](https://github.com/andreirusanescu/Image-Editor) in C using math.
I decided to embark on this journey because I wanted to learn this very powerful library, openCV.

```ROTATE``` functionality is a bit more high level, due to its ability of rotating the matrix at any angle, irrespective of whether or not it fits the whole original image (i.e. can be rotated at angles like 6 degrees not just 0, 45, 90, 180 and combinations of those angles). It uses a rotation matrix and an affine transformation.
The application features two types of blur ```BLUR```, a
Gaussian Blur and a Median Blur, but we recommend the first mentioned, due to its customizable ability (it uses a kernel and standard deviation). Basically, sigmaX controls the degree of blur added to the image. If the standard deviation is big => the Gaussian distribution will be more spreaded out (G(x, y) = 1 / (2pi * sigma^2) * e ^ (-(x^2+y^2)/2sigma^2)). If not parameter is given to sigmaX, then openCV gives it a value using this formula: ```σ = 0.3 * ((ksize − 1)/2 − 1) + 0.8```.

```SHARPEN``` imrpoves edge clarity and the overall image details by increasing the contrast between adjacent pixels.The sharpen kernel used causes some noise that is afterwards reduced using denoise_strength.

```EDGE``` uses Canny method from openCV with fixed sizes for the thresholds. Basically, what it does is a Gaussian blur to reduce the noise, then it computes the gradient of each pixel using the Sobel operator and the pixels that are not following the direction of the gradient are removed (i.e. the pixels that are not maximum points). The second threshold is for the strong edges, the middle pixels are held only if they are connected to a strong edge, and the pixels under the first threshold are removed.

```Equalize``` uses the Histogram equalization. Basically if the intensity of some pixels is extremely low, it increases it and if the intensity of other pixels is high it decreases it. 

```Pad kernel``` creates a padded array with the same shape as the image (target_shape). Places the kernel in the center of this new padded array. Shifts the kernel's center to the correct position for frequency domain convolution using np.fft.ifftshift(). ```Wiener Deconvolution``` tries to balance between unblurring the image and not amplifying the noise. It achieves this using a control parameter K to regularize the noise.

## main.py & image.py
The text-interface logic of the application is handled by the first file, while the monkey patching to link the methods declared in other files with ```MyImage```is done in the latter.

## sift.py
SIFT (Scale-Invariant Feature Transform) algorithm detects, extracts, and matches key points between two images. It creates a SIFT detector using cv2.SIFT_create(). It uses detectAndCompute() to identify key points (important features) and their corresponding descriptors (feature descriptions) in the image. It returns the key points and descriptors, which are essential for matching features between images. Uses Brute-Force Matcher (BFMatcher) to match the key points between the two images. Matches are sorted by distance (shorter distances = better matches). Only the top nr_matches are kept.

## ransac.py
RANSAC (RANdom SAmple Consensus) based image alignment algorithm. It takes two images as input (img1 and img2), finds their key features using SIFT, matches the features, and estimates an affine transformation to align the images. Finally, it warps one image to align with the other and blends them for visualization. The affine transformation formula is as follows: t = A*s + T, where A is a 2x2 matrix (rotation and scaling) and T is a 2x1 translation vector. Calculates the residual error for each point pair after applying the affine transformation. Randomly selects k point pairs to estimate an affine transformation. Computes the residual error for all points and counts inliers (points with error < threshold).

## Implementation Challenges
- **Understanding the math and logic behind these powerful methods**: there is not much control over the high level methods, if something fails and you do not know why it is hard to debug.
