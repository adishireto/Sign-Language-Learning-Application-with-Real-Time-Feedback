import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time
import pygame
import cv2
import joblib
from skimage.feature import hog
import random


def displaying_the_modes():
    # Displaying buttons on the screen to select options: game or learn on the main screen
    # and an exit button on the screens of the game and learn modes.
    global frame, stat_flag
    if stat_flag == 0:
        frame[0:200, -200:] = cv2.addWeighted(frame[0:200, -200:], 0.6, learn_mood_logo, 0.4, 0)
        frame[0:200, 0:200] = cv2.addWeighted(frame[0:200, 0:200], 0.6, game_mood_logo, 0.4, 0)

    if stat_flag == 2:
        frame[0:200, 0:200] = cv2.addWeighted(frame[0:200, 0:200], 0.6, exit_logo, 0.4, 0)

    if stat_flag == 1:
        frame[0:200, 0:200] = cv2.addWeighted(frame[0:200, 0:200], 0.6, exit_logo, 0.4, 0)


def check_buttons():
    # Checking the mode that the participant wants to move to.
    # The participant must press the requested button by moving a hand in the area of the button.
    # A comparison is made between the pixel value in the designated area before and after the participant's selection
    # and accordingly a transition between the modes is made.

    global frame, init_mean_game, init_mean_learn, stat_flag, init_mean_exit, run_menu_flag, cap, window
    global curr_mean_game, curr_mean_learn, curr_mean_exit

    if stat_flag == 0:  # menu mode
        game_button_area = frame[0:200, 0:200]
        learn_button_area = frame[0:200, -200:]
        curr_mean_game = np.array(cv2.mean(game_button_area)[:-1])  # mean over the game button area
        curr_mean_learn = np.array(cv2.mean(learn_button_area)[:-1])  # mean over the learn button area

        if np.linalg.norm(curr_mean_game - init_mean_game) > 40:
            run_menu_flag = False
            stat_flag = 2  # move to game mode
            clear_window(window)
            print('move to game mode')
            main_menu(window)

        if np.linalg.norm(curr_mean_learn - init_mean_learn) > 40:
            run_menu_flag = False
            stat_flag = 1  # move to learn mode
            clear_window(window)
            print('move to learn mode')
            main_menu(window)

    if stat_flag == 1 or stat_flag == 2:  # learn or game mode
        exit_button_area = frame[0:200, 0:200]
        curr_mean_exit = np.array(cv2.mean(exit_button_area)[:-1])  # mean over the exit button area
        if np.linalg.norm(curr_mean_exit - init_mean_exit) > 40:
            run_menu_flag = False
            stat_flag = 0  # move to menu
            back2menu.play()
            clear_window(window)
            print('move to menu mode')
            main_menu(window)


def init_button(frame):
    # Calculation of the pixel value in the button area in the initial state
    global init_mean_game, init_mean_learn, init_mean_exit
    global curr_mean_learn, curr_mean_game, curr_mean_exit

    exit_button_area = frame[0:200, 0:200]
    init_mean_exit = np.array(cv2.mean(exit_button_area)[:-1])
    curr_mean_exit = init_mean_exit

    game_button_area = frame[0:200, 0:200]
    init_mean_game = np.array(cv2.mean(game_button_area)[:-1])
    curr_mean_game = init_mean_game

    learn_button_area = frame[0:200, -200:]
    init_mean_learn = np.array(cv2.mean(learn_button_area)[:-1])
    curr_mean_learn = init_mean_learn


def read_word_and_pre_pros(image_path, factor):
    # Import the image and its logo according to the requested word
    # Adaptation to the computer screen display
    image = cv2.imread(image_path)
    w = int(image.shape[0] * factor)
    h = int(image.shape[1] * factor)
    image = cv2.resize(image, (h, w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


def extract_hog_features(image):
    # Define HOG parameters
    orientations = 12
    pixels_per_cell = (16, 16)
    cells_per_block = (4, 4)
    # Compute HOG features
    hog_features, hog_image = hog(image, channel_axis=-1, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, visualize=True, block_norm='L2-Hys',
                                  transform_sqrt=True)
    return hog_features, hog_image


def predict_func(frame_copy):
    # Calculation of the movement performed by a participant.
    # First step: finding the face space using "Cascade Classifier" algorithm and adjusting it to the chest space
    gray_img = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2GRAY)
    haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)
    try:
        (x, y, w, h) = faces_rect[0]
        h = h * 4
        x = x - 2 * w
        w = 5 * w
        z = max(1, x + 2)
        frame_copy = frame_copy[y + 2:y + h - 2, z:x + w - 2]
    except:
        print('do not found a human in img!')

    # Second step: Finding significant corners using "corne rHarris" Algorithm.
    image_gray = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2GRAY)
    dst = cv2.cornerHarris(image_gray, blockSize=4, ksize=5, k=0.06)
    frame_copy[dst > 0.01 * dst.max()] = [0, 0, 255]
    normalized_img = cv2.normalize(frame_copy, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    normalized_img = cv2.resize(normalized_img, (256, 256))

    # Third step: Extract hog features
    [hog_features, hog_image] = extract_hog_features(normalized_img)

    # Fourth step: predicting the movement performed by the participant using the trained model.
    # Determining the predicting movement if the result is significant.
    y_pred_hog = model.predict(np.array([hog_features]))[0]
    pred_pro_hog = model.predict_proba(np.array([hog_features]))[0]
    sorted_indices = np.argsort(pred_pro_hog)
    pred_pro_hog_sorted = pred_pro_hog[sorted_indices]
    max_val = pred_pro_hog_sorted[-1]
    max_val_next = pred_pro_hog_sorted[-2]
    if max_val * 0.75 < max_val_next:
        return None
    else:
        return y_pred_hog


def main_menu(window):
    global frame, init_mean_game, init_mean_learn, stat_flag
    global run_menu_flag, cap, init_mean_exit, frame_copy
    center_frame = tk.Label(window)
    center_frame.pack()

    right_frame = tk.Label(window)
    right_frame.pack(side=tk.LEFT)
    left_label = tk.Label(window)
    left_label.pack(side=tk.RIGHT)

    ret, frame = cap.read()
    w = int(frame.shape[0] * 1.8)
    h = int(frame.shape[1] * 1.8)
    frame = cv2.resize(frame, (h, w))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    init_button(frame)
    displaying_the_modes()
    run_menu_flag = True
    move_timer = time.time()

    if stat_flag == 1:  # learning mode
        print('in to learning mode', time.time())
        do_the_wav_in_right.play()
        time.sleep(3)
        tik = time.time()
        pred_tik = time.time()
        predicted_labels = []
        ind_to_predict = 0      # Selection according to the order of a word for learning mode

    if stat_flag == 2:  # game mode
        print('in to game mode', time.time())
        in_to_game_mode.play()
        time.sleep(3)
        tik = time.time()
        pred_tik = time.time()
        predicted_labels = []
        random_case = random.choice(list_of_logos_ImageTk)   # Random selection of a word for the game mode

    while run_menu_flag:
        try:
            ret, frame = cap.read()

        except:
            print('cap in not valid!')
            break

        frame = cv2.resize(frame, (h, w))
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if abs(time.time() - move_timer) > 3.5:
            frame_copy = frame.copy()

            check_buttons()
            displaying_the_modes()
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)

            if stat_flag == 0:  # menu
                center_frame.config(image=img)
                center_frame.image = img
                center_frame.after(10)

            if stat_flag == 1:  # learning mode
                right_frame.config(image=img)
                right_frame.image = img
                right_frame.after(10)
                word_img = list_of_words_ImageTk[ind_to_predict % len(list_of_words_ImageTk)]
                word_true_label = list_of_words_paths[ind_to_predict % len(list_of_words_ImageTk)][1]

                left_label.config(image=word_img)
                left_label.image = word_img
                left_label.after(10)

                if abs(pred_tik - time.time()) > 0.25 and abs(time.time() - tik) > 4:
                    y_pred = predict_func(frame_copy)
                    if y_pred != None:
                        predicted_labels += [y_pred]
                    else:
                        print('Do not into predicted_labels list')
                    pred_tik = time.time()

                # Checking the participant's word performance
                # The test is performed under the condition that 10 seconds have passed or that the following two
                # conditions are met:
                # 1. There are at least 5 words in the "predicted_labels" array.
                # 2. The median value of the array is equal to the real word.
                med_label = np.median(np.array(predicted_labels))
                if abs(time.time() - tik) > 10 or ((len(predicted_labels) >= 5) and med_label == word_true_label):
                    print(predicted_labels)
                    med_label = np.median(np.array(predicted_labels))

                    if med_label == word_true_label:  # The participant was able to perform the word well
                        if med_label == 3:
                            Friendship_yes.play()
                        elif med_label == 4:
                            Home_yes.play()
                        elif med_label == 5:
                            love_yes.play()
                        elif med_label == 6:
                            OK_yes.play()
                        elif med_label == 8:
                            Thanku_yes.play()
                        else:
                            nice.play()
                    else:                            # The participant could not perform the word well
                        if word_true_label == 4:
                            Home_no.play()
                        elif word_true_label == 5:
                            Love_no.play()
                        elif word_true_label == 6:
                            Not_ok.play()
                        elif word_true_label == 7:
                            Sorry_not.play()
                        else:
                            mistake.play()

                    ind_to_predict += 1
                    tik = time.time()
                    predicted_labels = []
                    time.sleep(3)

            if stat_flag == 2:  # game mode

                right_frame.config(image=img)
                right_frame.image = img
                right_frame.after(10)

                word_img = random_case
                word_true_label = list_of_words_paths[list_of_logos_ImageTk.index(random_case)][1]

                left_label.config(image=word_img)
                left_label.image = word_img
                left_label.after(10)

                if abs(pred_tik - time.time()) > 0.5:
                    y_pred = predict_func(frame_copy)
                    if y_pred != None:
                        predicted_labels += [y_pred]
                    pred_tik = time.time()

                # Checking the participant's word performance
                # The test is performed under the condition that 8 seconds have passed or that the following two
                # conditions are met:
                # 1. There are at least 5 words in the "predicted_labels" array.
                # 2. The median value of the array is equal to the real word.
                med_label = np.median(np.array(predicted_labels))
                if abs(time.time() - tik) > 8 or ((len(predicted_labels) >= 5) and med_label == word_true_label):
                    print(predicted_labels)
                    med_label = np.median(np.array(predicted_labels))

                    if med_label == word_true_label:   # The participant was able to perform the word well
                        if med_label == 3:
                            Friendship_yes.play()
                        elif med_label == 4:
                            Home_yes.play()
                        elif med_label == 5:
                            love_yes.play()
                        elif med_label == 6:
                            OK_yes.play()
                        elif med_label == 8:
                            Thanku_yes.play()
                        else:
                            nice.play()
                    else:                             # The participant could not perform the word well
                        if word_true_label == 4:
                            Home_no.play()
                        elif word_true_label == 5:
                            Love_no.play()
                        elif word_true_label == 6:
                            Not_ok.play()
                        elif word_true_label == 7:
                            Sorry_not.play()
                        else:
                            mistake.play()

                    time.sleep(3)
                    random_case = random.choice(list_of_logos_ImageTk)
                    tik = time.time()
                    predicted_labels = []

        else:
            init_button(frame)
            displaying_the_modes()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        window.update()

    window.mainloop()


def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()


if __name__ == "__main__":  # real time only, this open the camera!

    stat_flag = 0  # 0 - menu , 1 - learning , 2 - game
    window = tk.Tk()
    window.title("Sign Language Recognition App")
    window.bind('<Escape>', lambda e: window.quit())

    welcome_label = tk.Label(window, text="Welcome to the Sign Language Recognition App!")
    welcome_label.pack()
    pygame.init()

    # Import sound clips into the application:
    # vocabulary:
    Friendship_yes = pygame.mixer.Sound(r'voice\Friendship yes.wav')
    Home_no = pygame.mixer.Sound(r'voice\Home no.wav')
    Home_yes = pygame.mixer.Sound(r'voice\Home yes.wav')
    Love_no = pygame.mixer.Sound(r'voice\Love no.wav')
    Not_ok = pygame.mixer.Sound(r'voice\Not ok.wav')
    OK_yes = pygame.mixer.Sound(r'voice\OK yes.wav')
    Sorry_not = pygame.mixer.Sound(r'voice\Sorry not.wav')
    Thanku_yes = pygame.mixer.Sound(r'voice\Thanku yes.wav')
    love_yes = pygame.mixer.Sound(r'voice\love-yes.wav')
    kiss_yes = pygame.mixer.Sound(r'voice\kiss-yes.wav')

    # feedbacks:
    nice = pygame.mixer.Sound(r'voice\nice.wav')
    mistake = pygame.mixer.Sound(r'voice\mistake.wav')
    do_the_wav_in_right = pygame.mixer.Sound(r'voice\do_the_wav_in_right.wav')
    in_to_game_mode = pygame.mixer.Sound(r'voice\in_to_game_mode.wav')
    back2menu = pygame.mixer.Sound(r'voice\back2menu.wav')

    # Import images and logos into the application:
    list_of_words_paths = [(r'presenting_the_words\break1.jpg',      1,   r'presenting_the_words\bracklogo.jpg'),
                           (r'presenting_the_words\baby1.jpg',       2,   r'presenting_the_words\babylogo.jpg'),
                           (r'presenting_the_words\friendship1.jpg', 3,   r'presenting_the_words\friendshiplogo.jpg'),
                           (r'presenting_the_words\home1.jpg',       4,   r'presenting_the_words\homelogo.jpg'),
                           (r'presenting_the_words\love1.jpg',       5,   r'presenting_the_words\lovelogo.jpg'),
                           (r'presenting_the_words\ok1.jpg',         6,   r'presenting_the_words\oklogo.jpg'),
                           (r'presenting_the_words\sorry1.jpg',      7,   r'presenting_the_words\sorrylogo.jpg'),
                           (r'presenting_the_words\thank1.jpg',      8,   r'presenting_the_words\thanklogo.jpg')]

    logo_list = []
    list_of_words_ImageTk = [read_word_and_pre_pros(img_path[0], 0.5) for img_path in list_of_words_paths]
    list_of_logos_ImageTk = [read_word_and_pre_pros(img_path[2], 1.5) for img_path in list_of_words_paths]

    game_mood_path = r'buttons_pic\game_mode.png'
    game_mood_logo = cv2.imread(game_mood_path)
    game_mood_logo = cv2.resize(game_mood_logo, (200, 200))
    learn_mood_path = r'buttons_pic\learn_mode.png'
    learn_mood_logo = cv2.imread(learn_mood_path)
    learn_mood_logo = cv2.resize(learn_mood_logo, (200, 200))
    exit_path = r'buttons_pic\exit.jpg'
    exit_logo = cv2.imread(exit_path)
    exit_logo = cv2.resize(exit_logo, (200, 200))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    model_path = r'LogisticRegression_12_(16, 16)_(4, 4).pkl'
    model = joblib.load(model_path)
    main_menu(window)
