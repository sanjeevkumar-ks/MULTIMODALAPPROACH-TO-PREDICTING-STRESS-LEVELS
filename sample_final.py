from tkinter import *
from tkinter import messagebox as mb
from PIL import Image, ImageTk
import json
import os
import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
import keras.utils as image
import threading

class Quiz:
    def __init__(self):
        self.q_no = 0
        self.display_title()
        self.display_question()
        self.opt_selected = IntVar()
        self.opts = self.radio_buttons()
        self.display_options()
        self.buttons()
        self.data_size = len(question)
        self.output = []
        self.stress_score = 0
        self.correct = 0
        self.start_camera()

    def start_camera(self):
        self.model = model_from_json(open("fer.json", "r").read())
        self.model.load_weights('fer.h5')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.count_anx = 0
        self.count_desp = 0
        self.count_normal = 0
        self.update_camera()

    def update_camera(self):
        ret, test_img = self.cap.read()
        if ret:
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            faces_detected = self.face_cascade.detectMultiScale(gray_img, 1.32, 5)
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = self.model.predict(img_pixels)
                max_index = np.argmax(predictions[0])
                emotions = ('angry', 'disgust', 'Anxiety', 'happy', 'Depressed', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]
                if predicted_emotion == 'Anxiety':
                    self.count_anx += 1
                elif predicted_emotion == 'Depressed':
                    self.count_desp += 1
                else:
                    self.count_normal += 1
                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(test_img, (1500, 800))
            cv2.imshow('Facial emotion analysis ', resized_img)
            cv2.setWindowProperty("Facial emotion analysis ", cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                return

            gui.after(10, self.update_camera)

    def stop_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

        if self.count_anx > self.count_desp:
            if self.count_anx > self.count_desp:
                res = 'Anxious'
            else:
                res = 'Normal'
        else:
            if self.count_normal > self.count_desp:
                res = 'Normal'
            else:
                res = 'Depressed'

        print("Facial stress result : ", res)
        mb.showinfo("Facial Emotion based stress Result", res)
        self.calculate_result(res)

    def calculate_result(self, res):
        wrong_count = self.data_size - self.correct
        correct = f"Correct: {self.correct}"
        wrong = f"Wrong: {wrong_count}"
        score = int(self.correct / self.data_size * 100)
        result = f"Score: {score}%"
        score2 = self.stress_score
        stress_percentage = f"Stress Percentage: {score2}%"

        if score2 <= 4:
            stress_level = "Normal"
        elif score2 >= 5 and score2 <= 9:
            stress_level = "mild Stress"
        elif score2 >= 10 and score2 <= 14:
            stress_level = "moderate Stress"
        elif score2 >= 15 and score2 <= 19:
            stress_level = "moderately severe"
        else:
            stress_level = "severe Stress"

        mb.showinfo("Result", f"Stress Prediction: {stress_level}\n{stress_percentage}")

        if res == "Normal" and stress_level == "Normal":
            mb.showinfo("Facial cum Question Response Result", "Normal")
        elif res == "Normal" and stress_level == "mild Stress":
            mb.showinfo("Facial cum Question Response Result", "Mild Stress")
        elif res == "Normal" and stress_level == "moderate Stress":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == "Normal" and stress_level == "moderately severe":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == "Normal" and stress_level == "severe Stress":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == 'Anxious' and stress_level == "Normal":
            mb.showinfo("Facial cum Question Response Result", "Normal")
        elif res == 'Anxious' and stress_level == "mild Stress":
            mb.showinfo("Facial cum Question Response Result", "mild Stress")
        elif res == 'Anxious' and stress_level == "moderate Stress":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == 'Anxious' and stress_level == "moderately severe":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == 'Anxious' and stress_level == "severe Stress":
            mb.showinfo("Facial cum Question Response Result", "severe Stress")
        elif res == 'Depressed' and stress_level == "Normal":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == 'Depressed' and stress_level == "mild Stress":
            mb.showinfo("Facial cum Question Response Result", "moderate Stress")
        elif res == 'Depressed' and stress_level == "moderately severe":
            mb.showinfo("Facial cum Question Response Result", "severe Stress")
        elif res == 'Depressed' and stress_level == "severe Stress":
            mb.showinfo("Facial cum Question Response Result", "severe Stress")

        fig = plt.figure(figsize=(10, 10))
        plt.bar("Random forest", 94.0, color='green', width=0.4)
        plt.bar("SVM", 85.0, color='orange', width=0.4)
        plt.bar("Logistic Regression", 82.0, color='pink', width=0.4)
        plt.bar("KNN", 82.0, color='red', width=0.4)
        plt.bar("Naive Bayes", 81.0, color='blue', width=0.4)
        plt.xlabel("Algorithms")
        plt.ylabel("Accuracy")
        plt.title("Algorithm Accuracy Scores")
        plt.show()

    def check_ans(self, q_no):
        out = self.opt_selected.get() - 1
        self.stress_score += out
        self.output.append(out)
        if self.opt_selected.get() == answer[q_no]:
            return True

    def next_btn(self):
        if self.check_ans(self.q_no):
            self.correct += 1
        self.q_no += 1
        if self.q_no == self.data_size:
            self.stop_camera()
        else:
            self.display_question()
            self.display_options()

    def buttons(self):
        next_button = Button(gui, text="Next", command=self.next_btn,
                             width=10, bg="blue", fg="white", font=("ariel", 16, "bold"))
        next_button.place(x=350, y=380)
        quit_button = Button(gui, text="Quit", command=gui.destroy,
                             width=5, bg="black", fg="white", font=("ariel", 16, " bold"))
        quit_button.place(x=700, y=50)

    def display_options(self):
        val = 0
        self.opt_selected.set(0)
        for option in options[self.q_no]:
            self.opts[val]['text'] = option
            val += 1

    def display_question(self):
        q_no = Label(gui, text=question[self.q_no], width=60,
                     font=('ariel', 14, 'bold'), anchor='w')
        q_no.place(x=70, y=100)

    def display_title(self):
        title = Label(gui, text="Stress Prediction",
                      width=50, bg="green", fg="white", font=("ariel", 20, "bold"))
        title.place(x=0, y=2)

    def radio_buttons(self):
        q_list = []
        y_pos = 150
        while len(q_list) < 4:
            radio_btn = Radiobutton(gui, text=" ", variable=self.opt_selected,
                                    value=len(q_list) + 1, font=("ariel", 14))
            q_list.append(radio_btn)
            radio_btn.place(x=100, y=y_pos)
            y_pos += 40
        return q_list

gui = Tk()
gui.state("zoomed")
gui.title("Stress Prediction")

with open('data.json') as f:
    data = json.load(f)

question = (data['question'])
options = (data['options'])
answer = (data['answer'])

c1 = Canvas(gui, bg="white", height=800, width=1500)
college1 = Image.open(r"img/New Project 202 [23D983A].png")
resize1 = college1.resize((1280, 832))
college_photo1 = ImageTk.PhotoImage(resize1)
background_image_label1 = Label(gui, image=college_photo1)
background_image_label1.place(x=0, y=0)

select_button = Button(gui, text="Start Analysing myself", bg="red", fg="white", command=lambda: run(),
                       font=("times new roman", 30), cursor="hand2")
select_button.place(x=450, y=460, width=450, height=50)

def run():
    select_button.destroy()
    quiz = Quiz()

gui.mainloop()
