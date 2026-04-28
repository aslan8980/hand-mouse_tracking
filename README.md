# 🖐️ AI Gesture Mouse

> Control your computer using only hand gestures via webcam.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![Platform](https://img.shields.io/badge/macOS-supported-lightgrey)

---

## 📌 Overview

AI Gesture Mouse is a computer vision project that allows you to control your system cursor without a physical mouse.

Using a webcam and hand tracking, the system detects gestures in real time and maps them to mouse actions such as movement, click, drag, and double click.

This project demonstrates how **AI + computer vision** can be used to build natural human-computer interaction systems.

---

## ✨ Features

- 🖱️ Cursor movement using hand position  
- 👌 Click gesture (thumb + index finger)  
- ✊ Drag & drop (pinch hold)  
- ☝️ Double click using gesture recognition  
- ✋ Cursor freeze (left-hand fist)  
- ⚡ Real-time performance  

---

## 🧠 Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Quartz (macOS system-level mouse control)  

---

## ⚙️ How It Works

1. Webcam captures real-time frames  
2. MediaPipe detects 21 hand landmarks  
3. Gesture logic analyzes finger positions  
4. Gestures are translated into mouse actions  
5. Quartz API sends events to macOS  

---

## 🎮 Controls

| Gesture | Action |
|--------|--------|
| Move right hand | Move cursor |
| Thumb + Index | Click |
| Pinch hold | Drag |
| ☝️ Index finger up | Double click |
| Left hand fist | Freeze cursor |

---

## 🛠 Installation

```bash
git clone https://github.com/aslan8980/hand-mouse_tracking
cd hand-mouse_tracking
pip install -r requirements.txt
python hand_tracking.py
