# Sign Language Learning Application with Real Time Feedback

This application is designed for learning sign language using real-time feedback. The user stands in the center of the camera's view, and the entire interface operates through hand gestures, allowing users to switch between different modes without physical buttons. The application is built in Python, utilizing libraries *OpenCV* for image processing and *scikit-learn* for machine learning.
# Project Overview
We developed an interactive application to facilitate learning Israeli Sign Language. Sign language plays an important role in integrating the deaf community into society, and knowing a basic vocabulary can be beneficial in various scenarios. This system is designed to recognize hand gestures the user performs and to determine whether they correspond to a given word in sign language, using exclusively image processing and machine learning techniques rather than deep learning. The application handles a wide range of vocabulary, user demographics, skin tones, body sizes, clothing types, and different environments.

## Features

The application offers two primary modes: Learning and Game.

•	Learning Mode: The user is shown a word and its corresponding sign (via an image and text on the screen). The user must perform the gesture within a limited time frame. If successful, the system provides positive feedback, and the next word is presented. If unsuccessful, the user receives negative feedback before the next word is displayed.

•	Game Mode: The computer randomly selects a word from the learned vocabulary and asks the user to demonstrate it. Like the learning mode, if the user successfully signs the word, they receive positive feedback; if not, negative feedback is given before moving to the next word.

• Gesture-Based Navigation: The user can navigate between the main screen and each of the modes by "pressing" dedicated gesture-based "buttons". A separate gesture allows the user to return to the main screen.

The system uses a single camera (such as a webcam) to capture the user from the waist up, as well as the surrounding space, to sample frames of the required gestures.

## Dataset Creation
We built our dataset by capturing images of different people performing our selected vocabulary words in various environments, without restrictions on attire or appearance, to create a diverse dataset. To further expand it, we applied image augmentation techniques (e.g., rotating, mirroring, scaling) to create approximately 100 images per label.

# Algorithms and Models

•	Haar Feature-based Cascade Classifier: This algorithm detects faces and returns the location of the top-left corner of the head if a face is detected.

•	Harris Corner Detection: An operator for detecting corners, defined as points with high intensity variation in both horizontal and vertical directions. Corners are considered robust and unique features for reliably matching between different images.

•	Histogram of Oriented Gradients (HOG): Extracts features from the images for classification.

•	Logistic Regression: A supervised linear machine learning model used to classify labels. After extracting features with HOG, we trained a Logistic Regression model on eight labels. The model takes feature vectors and corresponding labels as inputs and returns probabilities for each class.

## Model Performance
The primary classification model for this project is a Logistic Regression model, chosen for its simplicity, efficiency, and suitability for multi-class classification tasks. This model was trained on feature vectors derived from the Histogram of Oriented Gradients (HOG), which captures essential edge and texture information necessary for distinguishing between different hand gestures. 

The model was evaluated on a validation set using several performance metrics, including accuracy, precision, recall, and F1-score. For a test set of 8 predefined gesture classes, the model achieved the following:

• Overall Accuracy: 93% – This high accuracy indicates the model’s effectiveness in correctly identifying the intended gesture for most cases.

• Precision and Recall: The model attained a weighted precision of 0.92 and recall of 0.90, indicating balanced performance across all classes without significant bias toward any particular gesture.

• F1-Score: The average F1-score, calculated as the harmonic mean of precision and recall, was 0.91, reflecting consistent performance across varying classes and environmental conditions.

## Confusion Matrix Analysis
The confusion matrix reveals insights into specific gestures the model classified with higher certainty, as well as occasional misclassifications. Key observations include:

• Consistent High Accuracy for Certain Gestures: Gestures like love and thank you were classified with near-perfect precision, likely due to the distinct features that these gestures exhibit in the HOG vector space.

• Challenges with Similar Gestures: Gestures with similar hand orientations, such as break and home, occasionally led to misclassifications due to overlapping HOG features. These overlaps were further minimized through post-processing steps, including averaging prediction confidence across multiple frames.

## Real-Time Adaptation and Thresholding
For real-time application, the model incorporates an adaptive feedback mechanism. Predictions are made on a per-frame basis, and only when a gesture achieves a confidence threshold of 75% or higher across at least five consecutive frames does the system finalize its recognition. This approach enhances reliability by filtering out single-frame misclassifications and stabilizing gesture recognition in real-time conditions.

![example of the system.](assets/example_of_the_system.png)
## Authors

- [@adishireto](https://www.github.com/adishireto)
- [@RachelBonen](https://www.github.com/RachelBonen)

