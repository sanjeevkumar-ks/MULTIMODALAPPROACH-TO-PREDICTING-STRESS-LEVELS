from tkinter import *
from tkinter import messagebox as mb
from PIL import Image, ImageTk
import json
import os
import cv2
import numpy as np
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class StartPage:
    def __init__(self, master):
        self.master = master
        self.display_start_page()

    def display_start_page(self):
        start_button = Button(self.master, text="Start Analyzing Myself", command=self.start_analysis,
                              width=20, bg="blue", fg="white", font=("ariel", 16, "bold"))
        start_button.pack(pady=50)

    def start_analysis(self):
        self.master.destroy()  # Close the StartPage GUI
        analyze_gui = Tk()
        analyze_gui.state("zoomed")
        analyze_gui.title("Stress Prediction")

        with open('data.json') as f:
            data = json.load(f)

        global question, options, answer
        question = data['question']
        options = data['options']
        answer = data['answer']

        quiz_frame = Frame(analyze_gui, bg='white')
        quiz_frame.grid(row=0, column=0, padx=20, pady=20)
        quiz = Quiz(quiz_frame)

        camera_frame = Frame(analyze_gui, bg='white')
        camera_frame.grid(row=0, column=1, padx=20, pady=20)
        camera = Camera(camera_frame)

        quit_button = Button(analyze_gui, text="Quit", command=analyze_gui.destroy,
                             width=5, bg="black", fg="white", font=("ariel", 16, " bold"))
        quit_button.grid(row=1, column=0, columnspan=2, pady=10)

        analyze_gui.mainloop()


class Quiz:
    def __init__(self, master):
        self.master = master
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
        self.rf_model = self.train_random_forest()

    def display_title(self):
        title = Label(self.master, text="Stress Prediction",
                      width=50, bg="green", fg="white", font=("ariel", 20, "bold"))
        title.grid(row=0, column=0, columnspan=2)

    def display_question(self):
        q_no = Label(self.master, text=question[self.q_no], width=60,
                     font=('ariel', 14, 'bold'), anchor='w')
        q_no.grid(row=1, column=0, columnspan=2)

    def display_options(self):
        val = 0
        self.opt_selected.set(0)
        for option in options[self.q_no]:
            self.opts[val]['text'] = option
            val += 1

    def radio_buttons(self):
        q_list = []
        y_pos = 150
        while len(q_list) < 4:
            radio_btn = Radiobutton(self.master, text=" ", variable=self.opt_selected,
                                    value=len(q_list) + 1, font=("ariel", 14))
            q_list.append(radio_btn)
            radio_btn.grid(row=len(q_list) + 1, column=0, sticky=W)
            y_pos += 40
        return q_list

    def buttons(self):
        next_button = Button(self.master, text="Next", command=self.next_btn,
                             width=10, bg="blue", fg="white", font=("ariel", 16, "bold"))
        next_button.grid(row=6, column=0, pady=10)

    def check_ans(self):
        out = self.opt_selected.get() - 1
        self.stress_score += out
        self.output.append(out)

    def next_btn(self):
        self.check_ans()
        self.q_no += 1
        if self.q_no == self.data_size:
            self.display_result()
        else:
            self.display_question()
            self.display_options()

    def display_result(self):
        answered_questions = self.q_no  # Number of questions answered
        if answered_questions == 0:  # If no questions were answered
            mb.showerror("Error", "No questions were answered.")
            return

        max_score = (len(options[0]) - 1) * answered_questions
        stress_score = self.stress_score if self.stress_score >= 0 else 0  # Ensure not negative
        stress_percentage = round((stress_score / max_score) * 100, 2)  # Calculate with 2 decimal points
        stress_percentage = max(0, stress_percentage)  # Ensure not negative

        stress_level = self.calculate_stress_level()
        mb.showinfo("Result", f"Stress Level: {stress_level}\nStress Percentage: {stress_percentage}%")
        self.master.master.destroy()  # Close the Analyze GUI after displaying the result

    def calculate_stress_level(self):
        if self.stress_score <= 20:
            return "Normal"
        elif self.stress_score >= 5 and self.stress_score <= 30:
            return "Mild Stress"
        elif self.stress_score >= 10 and self.stress_score <= 50:
            return "Moderate Stress"
        elif self.stress_score >= 15 and self.stress_score <= 80:
            return "Moderately Severe Stress"
        else:
            return "Severe Stress"

    def train_random_forest(self):
        # Load dataset (replace this with your dataset)
        X = np.random.rand(100, 10)  # Sample feature matrix
        y = np.random.randint(0, 2, size=100)  # Sample target vector

        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Initialize and train Random Forest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        return rf_model


class Camera:
    def __init__(self, master):
        self.master = master
        self.start_camera()

    def start_camera(self):
        self.model = self.load_cnn_model()
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
                predicted_emotion = self.predict_emotion(roi_gray)
                if predicted_emotion == 'Anxiety':
                    self.count_anx += 1
                elif predicted_emotion == 'Depressed':
                    self.count_desp += 1
                else:
                    self.count_normal += 1
                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(test_img, (640, 480))
            img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            label = Label(self.master, image=imgtk)
            label.imgtk = imgtk
            label.grid(row=0, column=0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                return

            self.master.after(10, self.update_camera)

    def load_cnn_model(self):
        # Load CNN model architecture and weights
        model = model_from_json(open("fer.json", "r").read())
        model.load_weights('fer.h5')
        return model

    def predict_emotion(self, roi_gray):
        # Preprocess the image
        roi_gray = roi_gray / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        # Predict the emotion
        predictions = self.model.predict(roi_gray)
        emotions = ['angry', 'disgust', 'Anxiety', 'happy', 'Depressed', 'surprise', 'neutral']
        predicted_emotion = emotions[np.argmax(predictions)]
        return predicted_emotion

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


def main():
    start_gui = Tk()
    start_gui.state("zoomed")
    start_gui.title("Start Page")
    start_page = StartPage(start_gui)
    start_gui.mainloop()


if __name__ == "__main__":
    main()
