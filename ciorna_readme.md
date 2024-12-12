# README

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

## detect_faces.py
- The code consists in three functions: __detect_face_rect, detect_face_coord and prepare_training_data__. 
- __Detect_face_rect__ detects a face and draws a rectangle around it. It's not called directly, but its' implementation is used in both detect_face_coord and prepare_train_data.
- __Detect_face_coord__ detects a face and returns its coordinates. The code loads the __Haar Cascade Frontal Face__ algorithm in a variable and Then, using that algorithm, it tries to detect the faces and draw a circle over the
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