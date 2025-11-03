import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from PIL import Image, ImageTk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def main():
    # --- Load YOLOv8 models ---
    fire_model = YOLO(r"C:\Users\Amol\OneDrive\Desktop\AD\runs\detect\train\weights\best.pt")  # Fire detection model
    accident_model = YOLO(r"C:\Users\Amol\OneDrive\Desktop\AD\ACCdataset\runs\detect\train2\weights\best.pt")  # Accident detection model

    # --- Evaluate model accuracy before launching GUI ---
    print("\n🔍 Evaluating Fire Detection Model...")
    fire_results = fire_model.val(data=r"C:\Users\Amol\OneDrive\Desktop\AD\firedataset\data.yaml", workers=0)
    print(f"🔥 Fire Model Accuracy:")
    print(f"  Precision: {fire_results.box.mp:.4f}")
    print(f"  Recall: {fire_results.box.mr:.4f}")
    print(f"  mAP50: {fire_results.box.map50:.4f}")
    print(f"  mAP50-95: {fire_results.box.map:.4f}")

    print("\n🔍 Evaluating Accident Detection Model...")
    accident_results = accident_model.val(data=r"C:\Users\Amol\OneDrive\Desktop\AD\ACCdataset\data.yaml", workers=0)
    print(f"🚗 Accident Model Accuracy:")
    print(f"  Precision: {accident_results.box.mp:.4f}")
    print(f"  Recall: {accident_results.box.mr:.4f}")
    print(f"  mAP50: {accident_results.box.map50:.4f}")
    print(f"  mAP50-95: {accident_results.box.map:.4f}")

    # --- Email configuration ---
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    SENDER_EMAIL = "amolsure88@gmail.com"
    SENDER_PASSWORD = "writ jqwm dlvv oqps"  # <--- Replace with your Gmail App Password (no spaces)
    FIRE_BRIGADE_EMAIL = "gammaforge22@gmail.com"
    HOSPITAL_EMAIL = "rameshteli56057@gmail.com"
    POLICE_EMAIL = "amolsutar2772@gmail.com"

    cap = None
    fire_detected = False
    accident_detected = False

    # --- Fire Alert ---
    def send_fire_alert():
        nonlocal fire_detected
        if fire_detected:
            return
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = FIRE_BRIGADE_EMAIL
            msg['Subject'] = "Fire Detected Alert!"
            body = "Alert! A fire has been detected in the monitored video feed. Immediate action is required."
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, FIRE_BRIGADE_EMAIL, msg.as_string())
            server.quit()
            fire_detected = True
            print("✅ Fire alert sent successfully!")
        except Exception as e:
            print(f"Error sending fire alert: {e}")

    # --- Accident Alert ---
    def send_accident_alert():
        nonlocal accident_detected
        if accident_detected:
            return
        try:
            subject = "Accident Detected Alert"
            body = "An accident has been detected. Please take immediate action."
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = SENDER_EMAIL
            msg["To"] = ", ".join([HOSPITAL_EMAIL, POLICE_EMAIL])

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, [HOSPITAL_EMAIL, POLICE_EMAIL], msg.as_string())
            server.quit()
            accident_detected = True
            print("✅ Accident alert sent successfully!")
        except Exception as e:
            print(f"Error sending accident alert: {e}")

    # --- Detection functions ---
    def detect_fire(frame):
        results = fire_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                if class_id == 0:
                    label = f"Fire {conf:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    send_fire_alert()
        return frame

    def detect_accident(frame):
        CONFIDENCE_THRESHOLD = 0.16
        results = accident_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                if class_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                    label = f"Accident {conf:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    send_accident_alert()
        return frame

    # --- Preprocess Frame ---
    def preprocess_frame(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Video Playback ---
    def play_video():
        nonlocal cap
        ret, frame = cap.read()
        if ret:
            processed_frame = preprocess_frame(frame)
            with ThreadPoolExecutor() as executor:
                fire_future = executor.submit(detect_fire, frame)
                accident_future = executor.submit(detect_accident, frame)
                frame = fire_future.result()
                frame = accident_future.result()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
        lbl_video.after(10, play_video)

    def start_video():
        nonlocal cap
        video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Unable to open video file.")
            else:
                play_video()

    def stop_video():
        nonlocal cap
        if cap:
            cap.release()
            cap = None
            lbl_video.config(image='')

    # --- GUI ---
    root = tk.Tk()
    root.title("Fire and Accident Detection")
    root.geometry("800x600")

    bg_image = Image.open(r"C:\Users\Amol\OneDrive\Desktop\DogVision\white-cloud-blue-sky-sea_74190-4488.jpg")
    bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
    bg_image = ImageTk.PhotoImage(bg_image)

    bg_label = tk.Label(root, image=bg_image)
    bg_label.place(relwidth=1, relheight=1)

    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), padding=10)

    video_frame = ttk.Frame(root)
    video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    global lbl_video
    lbl_video = ttk.Label(video_frame)
    lbl_video.pack(fill=tk.BOTH, expand=True)

    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)

    btn_start = ttk.Button(button_frame, text="Start Video", command=start_video)
    btn_start.pack(side=tk.LEFT, padx=5, pady=5)

    btn_stop = ttk.Button(button_frame, text="Stop Video", command=stop_video)
    btn_stop.pack(side=tk.RIGHT, padx=5, pady=5)

    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
