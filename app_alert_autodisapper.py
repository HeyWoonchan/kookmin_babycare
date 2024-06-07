import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk
import threading
import time
import requests

# 모델 불러오기
model_head = YOLO('models/head.pt')
model_posture = YOLO('models/posture.pt')
model_around = YOLO('models/around.pt')

# class 받아오기
class_names_head = model_head.names
class_names_posture = model_posture.names
class_names_around = model_around.names

# Pushover 알림 전송 함수
def send_pushover_notification(user_key, api_token, message):
    url = 'https://api.pushover.net/1/messages.json'
    payload = {
        'token': api_token,
        'user': user_key,
        'message': message
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print('Pushover Success')
        messagebox.showinfo("Pushover", "Pushover 알림 전송 성공!")
    else:
        print('Pushover Fail')
        messagebox.showerror("Pushover", "Pushover 알림 전송 실패!")

class BabyMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Baby Monitor App")

        # 알림 설정 변수
        self.pushover_user_key = tk.StringVar(value="u3s6ujq92zo5vy6cm6x4wv46dksxm1")
        self.pushover_api_key = tk.StringVar(value='aw9zx586p9rikjgqnsk1dahg2ccwp6')
        self.param_nohead_sec = tk.IntVar(value=5)  # 기본값 5초로 설정
        self.param_prone_sec = tk.IntVar(value=5)  # 아기가 배로 누운 시간 기본값 5초로 설정
        self.param_danger_dist = tk.DoubleVar(value=300.0)  # 아기 주변 위험요소 거리 기본값 1m로 설정
        self.enable_pushover = tk.BooleanVar(value=False)
        self.playback_speed = tk.DoubleVar(value=1.0)  # 기본 배속 설정

        self.setup_ui()

        self.cap = None
        self.running = False
        self.no_head_start_time = None
        self.last_posture_detected_time = None
        self.alert_triggered = False
        self.last_prone_time = None
        self.last_danger_time = None

        # 알림 조건 영역
        self.dangerous_object_alert_triggered = False
        self.prone_alert_triggered = False
        self.no_head_alert_triggered = False
        self.last_back_time = None

    def setup_ui(self):
        # 메인 화면 프레임
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 설정 화면 프레임
        self.settings_frame = tk.Frame(self.root)

        # 객체 검출 화면 프레임
        self.detect_frame = tk.Frame(self.root)

        # 메인 화면 구성
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(expand=True)

        tk.Button(self.button_frame, text="설정하기", command=self.show_settings).pack(pady=5)
        tk.Button(self.button_frame, text="영상 파일 열기", command=self.open_file).pack(pady=5)
        tk.Button(self.button_frame, text="동영상 파일 열기", command=self.open_video_file).pack(pady=5)
        tk.Button(self.button_frame, text="실시간 카메라 시작", command=self.start_camera).pack(pady=5)

        self.root.geometry("300x300")  # 초기 창 크기 설정

        # 설정 화면 구성
        settings_grid_frame = tk.Frame(self.settings_frame)
        settings_grid_frame.pack(expand=True, padx=10, pady=10)

        tk.Label(settings_grid_frame, text="Pushover User Key:").grid(row=0, column=0, padx=5, pady=5)
        tk.Entry(settings_grid_frame, textvariable=self.pushover_user_key).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(settings_grid_frame, text="Pushover API Key:").grid(row=1, column=0, padx=5, pady=5)
        tk.Entry(settings_grid_frame, textvariable=self.pushover_api_key).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(settings_grid_frame, text="No Head Detection Alert (seconds):").grid(row=2, column=0, padx=5, pady=5)
        tk.Entry(settings_grid_frame, textvariable=self.param_nohead_sec).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(settings_grid_frame, text="아기가 배로 누운지 몇 초 지나서 알람 발송할지:").grid(row=3, column=0, padx=5, pady=5)
        tk.Entry(settings_grid_frame, textvariable=self.param_prone_sec).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(settings_grid_frame, text="아기 주변에 위험요소가 얼마나 가까이 있을 때 알림 발송할지:").grid(row=4, column=0, padx=5, pady=5)
        tk.Entry(settings_grid_frame, textvariable=self.param_danger_dist).grid(row=4, column=1, padx=5, pady=5)

        tk.Checkbutton(settings_grid_frame, text="Pushover 알림 전송", variable=self.enable_pushover).grid(row=5, columnspan=2, pady=10)
        tk.Button(settings_grid_frame, text="Pushover 알림 테스트", command=self.test_pushover).grid(row=6, columnspan=2, pady=10)

        # 배속 슬라이더
        tk.Label(settings_grid_frame, text="배속:").grid(row=7, column=0, padx=5, pady=5)
        tk.Scale(settings_grid_frame, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.playback_speed).grid(row=7, column=1, padx=5, pady=5)

        # 설정 완료 버튼
        tk.Button(settings_grid_frame, text="설정 완료", command=self.hide_settings).grid(row=8, columnspan=2, pady=10)

        # 객체 검출 화면 구성
        self.stop_button = tk.Button(self.detect_frame, text="탐지 종료", command=self.stop_camera)
        self.stop_button.pack(side=tk.TOP, pady=5)

        self.canvas = tk.Canvas(self.detect_frame, width=800, height=600)
        self.canvas.pack()

    def show_settings(self):
        self.main_frame.pack_forget()
        self.adjust_window_size(600, 400)
        self.settings_frame.pack(fill=tk.BOTH, expand=True)

    def hide_settings(self):
        self.settings_frame.pack_forget()
        self.reset_window_size()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def test_pushover(self):
        if self.enable_pushover.get():
            user_key = self.pushover_user_key.get()
            api_token = self.pushover_api_key.get()
            message = "Pushover 알림 테스트 메시지입니다."
            threading.Thread(target=send_pushover_notification, args=(user_key, api_token, message)).start()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.adjust_window_size(800, 600)
            self.show_detect_frame()
            self.process_image(file_path)

    def open_video_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            self.start_video(file_path)

    def start_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.running = True
        self.adjust_window_size(800, 600)
        self.show_detect_frame()
        self.update_frame()

    def start_camera(self):
        self.cap = cv2.VideoCapture(2)
        self.running = True
        self.adjust_window_size(800, 600)
        self.show_detect_frame()
        self.update_frame()

    def show_detect_frame(self):
        self.main_frame.pack_forget()
        self.detect_frame.pack(fill=tk.BOTH, expand=True)

    def hide_detect_frame(self):
        self.detect_frame.pack_forget()
        self.reset_window_size()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def adjust_window_size(self, width, height):
        self.root.geometry(f"{width}x{height}")

    def reset_window_size(self):
        self.root.geometry("300x300")

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (800, 600))  # Resize frame to 800x600
                results_head = model_head(frame)
                results_posture = model_posture(frame)
                results_around = model_around(frame)

                centers_head = draw_boxes(frame, results_head, class_names_head, (255, 0, 0))
                centers_posture = draw_boxes(frame, results_posture, class_names_posture, (0, 255, 0))
                centers_around = draw_boxes(frame, results_around, class_names_around, (0, 0, 255))

                # 위험물질과 아기 거리 계산
                distances = []
                for center_p, _ in centers_posture:
                    for center_a, label in centers_around:
                        if label in ["scissors", "Knife"]:  # "scissors"와 "knife"에 대해서만 거리 계산
                            cv2.line(frame, center_p, center_a, (255, 255, 0), 2, lineType=cv2.LINE_AA)
                            distance = np.linalg.norm(np.array(center_p) - np.array(center_a))
                            distances.append(distance)
                            mid_point = ((center_p[0] + center_a[0]) // 2, (center_p[1] + center_a[1]) // 2)
                            distance_text = f"{distance:.2f}"
                            cv2.putText(frame, distance_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                self.check_alert_condition(centers_head, centers_posture, distances, results_posture)
                self.display_image(frame)

            delay = int(1000 / (30 * float(self.playback_speed.get())))  # Adjust delay based on playback speed
            self.root.after(delay, self.update_frame)

    def check_alert_condition(self, centers_head, centers_posture, distances, results_posture):
        current_time = time.time()

        if len(centers_posture) > 0:
            self.last_posture_detected_time = current_time

            # No head detection alert logic
            if len(centers_head) == 0:
                if self.no_head_start_time is None:
                    self.no_head_start_time = current_time
                elif current_time - self.no_head_start_time >= self.param_nohead_sec.get():
                    if not self.no_head_alert_triggered:
                        self.no_head_alert_triggered = True
                        message = f"No head detected for {self.param_nohead_sec.get()} seconds!"
                        threading.Thread(target=self.show_alert, args=(message,)).start()
                        if self.enable_pushover.get():
                            threading.Thread(target=send_pushover_notification, args=(self.pushover_user_key.get(), self.pushover_api_key.get(), message)).start()
                        self.root.after(5000, self.show_second_alert)
            else:
                self.no_head_start_time = None
                self.no_head_alert_triggered = False

            # Prone position alert logic
            if "baby-lying-on-stomach" in [class_names_posture[int(cls)] for cls in results_posture[0].boxes.cls]:
                if self.last_prone_time is None or current_time - self.last_prone_time <= 0.1:
                    self.last_prone_time = current_time
                elif current_time - self.last_prone_time >= self.param_prone_sec.get():
                    if not self.prone_alert_triggered:
                        self.prone_alert_triggered = True
                        message = f"Prone position detected for {self.param_prone_sec.get()} seconds!"
                        threading.Thread(target=self.show_alert, args=(message,)).start()
                        if self.enable_pushover.get():
                            threading.Thread(target=send_pushover_notification, args=(self.pushover_user_key.get(), self.pushover_api_key.get(), message)).start()
                        self.root.after(5000, self.show_second_alert)
            elif "baby-lying-on-back" in [class_names_posture[int(cls)] for cls in results_posture[0].boxes.cls]:
                # Track time when baby is lying on back
                if self.last_back_time is None:
                    self.last_back_time = current_time
                elif current_time - self.last_back_time >= 1:
                    # Reset prone alert if baby is lying on back for more than 1 second
                    self.last_prone_time = None
                    self.prone_alert_triggered = False
            else:
                self.last_back_time = None
                if self.last_prone_time is not None and current_time - self.last_prone_time <= 0.1:
                    # Keep last_prone_time as it is
                    pass
                else:
                    self.last_prone_time = None
                self.prone_alert_triggered = False

            # Dangerous object alert logic
            if distances:
                self.last_danger_time = current_time
                for distance in distances:
                    if float(distance) < float(self.param_danger_dist.get()):
                        if not self.dangerous_object_alert_triggered:
                            self.dangerous_object_alert_triggered = True
                            message = f"Dangerous object within {self.param_danger_dist.get()} meters!"
                            threading.Thread(target=self.show_alert, args=(message,)).start()
                            if self.enable_pushover.get():
                                threading.Thread(target=send_pushover_notification, args=(self.pushover_user_key.get(), self.pushover_api_key.get(), message)).start()
                            self.root.after(5000, self.show_second_alert)
                        break  # Only alert once for the first dangerous object detected
            else:
                if self.last_danger_time and current_time - self.last_danger_time > 1:
                    self.dangerous_object_alert_triggered = False
        else:
            if self.last_posture_detected_time is not None and current_time - self.last_posture_detected_time > 1:
                self.no_head_start_time = None
                self.no_head_alert_triggered = False
                self.prone_alert_triggered = False
                self.dangerous_object_alert_triggered = False

    def show_alert(self, message):
        alert_window = tk.Toplevel(self.root)
        alert_window.title("Warning")
        tk.Label(alert_window, text=message, padx=20, pady=20).pack()
        alert_window.after(2000, alert_window.destroy)  # 2초 후 창 닫기

    def show_second_alert(self):
        message = "Second Alert!"
        threading.Thread(target=self.show_alert, args=(message,)).start()
        if self.enable_pushover.get():
            threading.Thread(target=send_pushover_notification, args=(self.pushover_user_key.get(), self.pushover_api_key.get(), message)).start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.hide_detect_frame()

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (800, 600))  # Resize image to 800x600

        # Perform detection
        results_head = model_head(img)
        results_posture = model_posture(img)
        results_around = model_around(img)

        # Draw results on the image
        draw_boxes(img, results_head, class_names_head, (255, 0, 0))  # Red for head model
        centers_posture = draw_boxes(img, results_posture, class_names_posture, (0, 255, 0))  # Green for posture model
        centers_around = draw_boxes(img, results_around, class_names_around, (0, 0, 255))  # Blue for around model

        for center_p, _ in centers_posture:
            for center_a, label in centers_around:
                if label in ["scissors", "Knife"]:  # "scissors"와 "knife"에 대해서만 거리 계산
                    cv2.line(img, center_p, center_a, (255, 255, 0), 2, lineType=cv2.LINE_AA)
                    distance = np.linalg.norm(np.array(center_p) - np.array(center_a))
                    mid_point = ((center_p[0] + center_a[0]) // 2, (center_p[1] + center_a[1]) // 2)
                    distance_text = f"{distance:.2f}"
                    cv2.putText(img, distance_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        self.display_image(img)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

# Function to draw bounding boxes and labels on the image
def draw_boxes(img, results, class_names, color):
    centers = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = class_names[int(box.cls[0])]
            if label == "Person":
                continue
            confidence = box.conf[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append(((center_x, center_y), label))
            cv2.circle(img, (center_x, center_y), 5, color, -1)
    return centers

if __name__ == "__main__":
    root = tk.Tk()
    app = BabyMonitorApp(root)
    root.mainloop()
