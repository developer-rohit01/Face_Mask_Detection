import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading
import csv
import os
from datetime import datetime
import winsound
import pygame

pygame.mixer.init()
accepted_sound = pygame.mixer.Sound("acc.mp3")
alert_sound = pygame.mixer.Sound("rejected.mp3")

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== LOAD MODEL =====
model = load_model("mask_mobilenet.h5")

# ===== LOAD FACE DETECTOR =====
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ===== SETUP FILES =====
if not os.path.exists("violations"):
    os.makedirs("violations")

if not os.path.exists("log.csv"):
    with open("log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Status"])

# ===== GLOBAL =====
running = False
cap = None
alert_cooldown = 0

# ===== CAMERA FUNCTIONS =====
def start_camera():
    global running, cap
    if running:
        return
    running = True
    cap = cv2.VideoCapture(0)
    threading.Thread(target=video_loop, daemon=True).start()

def stop_camera():
    global running, cap
    running = False
    if cap:
        cap.release()

# ===== MAIN LOOP =====
def video_loop():
    global running, cap, alert_cooldown

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224))
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                pred = model.predict(face, verbose=0)[0][0]

                label = "Mask" if pred < 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                cv2.putText(frame, f"{label} ({pred:.2f})",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), color, 2)

               # ===== LOGGING =====
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")
                with open("log.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([date, time, label])

# ===== IMAGE SAVE FOR NO MASK =====
                if label == "No Mask":
                  img_name = f"violations/{date}_{time.replace(':','-')}.jpg"
                  cv2.imwrite(img_name, frame)

# ===== SOUND SYSTEM (FIXED) =====
                if alert_cooldown == 0:
                  if label == "Mask":
                    accepted_sound.play()
                    alert_cooldown = 30
                  else:
                    alert_sound.play()
                    alert_cooldown = 30

                  

        if alert_cooldown > 0:
            alert_cooldown -= 1

        # ===== SHOW IN TKINTER =====
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(img)

        video_label.config(image=img)
        video_label.image = img

# ===== GUI =====
root = tk.Tk()
root.title("Model Pehchan")
root.geometry("900x650")
root.configure(bg="#1e1e2f")

title = tk.Label(root, text="MODEL PEHCHAN", font=("Arial", 24, "bold"),
                 fg="#00c3ff", bg="#1e1e2f")
title.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

btn_frame = tk.Frame(root, bg="#1e1e2f")
btn_frame.pack(pady=20)

def styled_btn(text, command, color):
    return tk.Button(btn_frame, text=text, command=command,
                     font=("Arial", 12, "bold"),
                     bg=color, fg="white",
                     width=10, bd=0)

styled_btn("Start", start_camera, "#28a745").grid(row=0, column=0, padx=10)
styled_btn("Stop", stop_camera, "#dc3545").grid(row=0, column=1, padx=10)
styled_btn("Exit", root.quit, "#6c757d").grid(row=0, column=2, padx=10)

root.mainloop()