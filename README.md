# Sign-Language-Learning-Application-with-Real-Time-Feedback

This application is designed for learning sign language using real-time feedback. The user stands in the center of the camera's view, and the entire interface operates through hand gestures, allowing users to switch between different modes without physical buttons. The application is built in Python, utilizing OpenCV and scikit-learn libraries.
# Project Overview
We developed an interactive application to facilitate learning Israeli Sign Language. Sign language plays an important role in integrating the deaf community into society, and knowing a basic vocabulary can be beneficial in various scenarios. This system is designed to recognize hand gestures the user performs and to determine whether they correspond to a given word in sign language, using exclusively image processing and machine learning techniques rather than deep learning. The application handles a wide range of vocabulary, user demographics, skin tones, body sizes, clothing types, and different environments.

## Features

The application offers two primary modes: Learning and Game.
•	Learning Mode: The user is shown a word and its corresponding sign (via an image and text on the screen). The user must perform the gesture within a limited time frame. If successful, the system provides positive feedback, and the next word is presented. If unsuccessful, the user receives negative feedback before the next word is displayed.
•	Game Mode: The computer randomly selects a word from the learned vocabulary and asks the user to demonstrate it. Like the learning mode, if the user successfully signs the word, they receive positive feedback; if not, negative feedback is given before moving to the next word.
The user can navigate between the main screen and each of the modes by "pressing" dedicated gesture-based "buttons." A separate gesture allows the user to return to the main screen.
The system uses a single camera (such as a webcam) to capture the user from the waist up, as well as the surrounding space, to sample frames of the required gestures.
Dataset Creation
We built our dataset by capturing images of different people performing our selected vocabulary words in various environments, without restrictions on attire or appearance, to create a diverse dataset. To further expand it, we applied image augmentation techniques (e.g., rotating, mirroring, scaling) to create approximately 100 images per label.

## Algorithms and Models

•	Haar Feature-based Cascade Classifier: This algorithm detects faces and returns the location of the top-left corner of the head if a face is detected.
•	Harris Corner Detection: An operator for detecting corners, defined as points with high intensity variation in both horizontal and vertical directions. Corners are considered robust and unique features for reliably matching between different images.
•	Histogram of Oriented Gradients (HOG): Extracts features from the images for classification.
•	Logistic Regression: A supervised linear machine learning model used to classify labels. After extracting features with HOG, we trained a Logistic Regression model on eight labels. The model takes feature vectors and corresponding labels as inputs and returns probabilities for each class.

## Authors

- [@adishireto](https://www.github.com/adishireto)
- [@RachelBonen](https://www.github.com/RachelBonen)

