import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Google API Key
GOOGLE_API_KEY = "AIzaSyDhlw797sBH6BzqwC_8dK78kODSSg7lBtg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  

class PackageDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="best.pt"):
        """Initialize package detection processor for conveyor belt monitoring."""
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error loading YOLO model: {e}")

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: Could not open video file.")

        self.processed_track_ids = set()
        self.current_date = time.strftime("%Y-%m-%d")
        self.output_filename = f"package_data_{self.current_date}.txt"
        self.cropped_images_folder = "cropped_packages"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

        # Detection line parameters
        self.cx1 = 416  # X-position of the vertical line
        self.offset = 6  # Offset for detection tolerance

        # Initialize report file
        if not os.path.exists(self.output_filename):
            with open(self.output_filename, "w", encoding="utf-8") as file:
                file.write("Timestamp | Track ID | Package Type | Front Open | Damage Condition\n")
                file.write("-" * 80 + "\n")

    def encode_image_to_base64(self, image):
        _, img_buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(img_buffer).decode("utf-8")

    def analyze_image_with_gemini(self, image_path):
        """Analyze the package image using Gemini AI."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
               content=[
        {"type": "text", "text": 
         "Analyze the given image of a package and extract the following details with high accuracy:\n\n"
         "### **Box Open or Closed Detection:**\n"
         "- Focus ONLY on the **front flap of the box**.\n"
         "- If the front flap is **fully lifted, bent, or open**, exposing the inside, return 'Yes'.\n"
         "- If the front flap is **completely closed and properly sealed**, return 'No'.\n"
         "- If the box is **sealed with tape** and no flaps are lifted, it must be considered 'No'.\n"
         "- If the box has **a broken or partially lifted flap**, still return 'Yes'.\n"
         "- Ignore other sides of the box (top, bottom, back, etc.).\n\n"
         
         "### **Damage Condition:**\n"
         "- If there are **visible tears, dents, holes, or crushed areas**, return 'Yes'.\n"
         "- If the box appears intact, return 'No'.\n\n"

         "**Return results strictly in table format, with no additional text:**\n"
         "| Package Type (box) | Box Front Flap Open (Yes/No) | Damage Condition (Yes/No) |\n"
         "|--------------------|----------------------------|--------------------------|\n"
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
)

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invoking Gemini model: {e}")
            return "Error processing image."

    def process_crop_image(self, image, track_id):
        """Save and analyze cropped package images."""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = os.path.join(self.cropped_images_folder, f"package_{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        if extracted_data:
            with open(self.output_filename, "a", encoding="utf-8") as file:
                for row in extracted_data:
                    if "--------------" in row or not row.strip():
                        continue
                    values = [col.strip() for col in row.split("|")[1:-1]]
                    if len(values) == 3:
                        package_type, Box_flip_Open , damage_status = values
                        file.write(f"{timestamp} | Track ID: {track_id} | {package_type} | {Box_flip_Open} | {damage_status}\n")

            print(f"✅ Data saved for track ID {track_id}.")

    def crop_and_process(self, frame, box, track_id):
        """Crop and process detected packages."""
        if track_id in self.processed_track_ids:
            print(f"Track ID {track_id} already processed. Skipping.")
            return  

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        self.processed_track_ids.add(track_id)
        threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        """Process each video frame to detect and analyze packages."""
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2  # Calculate center X of the object

                # Object is detected only when its center crosses the line at x=416 (with offset)
                if self.cx1 - self.offset < cx < self.cx1 + self.offset:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    self.crop_and_process(frame, box, track_id)

        # Draw the vertical detection line
        cv2.line(frame, (416, 2), (416, 599), (0, 255, 0), 2)
        return frame
    
    @staticmethod
    def mouse_callback(event, x, y, flags, param):
        """Display mouse position for debugging."""
        if event == cv2.EVENT_MOUSEMOVE:
            print(f"Mouse Position: ({x}, {y})")

    def start_processing(self):
        """Start video processing."""
        cv2.namedWindow("Package Detection")
        cv2.setMouseCallback("Package Detection", self.mouse_callback)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = self.process_video_frame(frame)
            cv2.imshow("Package Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Data saved to {self.output_filename}")

if __name__ == "__main__":
    video_file = "co.mp4"  # Replace with your video file
    processor = PackageDetectionProcessor(video_file)
    processor.start_processing()
