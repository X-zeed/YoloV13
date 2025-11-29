import cv2
import numpy as np
from ultralytics import YOLO
import time

class YOLOv13Detector:
    def __init__(self, model_path='yolo11n.pt'):
        """
        Initialize YOLOv13 detector
        model_path: path to YOLO model weights
        ใช้ yolo11n.pt (YOLO11 nano) เป็นโมเดลล่าสุดที่รวดเร็วและมีประสิทธิภาพ
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
        
        # กำหนดสีสำหรับแต่ละ class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
        
    def detect_objects(self, frame, conf_threshold=0.5):
        """
        Detect objects in frame
        """
        # ทำการ detect
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        # วาด bounding boxes และ labels
        annotated_frame = results[0].plot()
        
        # ดึงข้อมูลการ detect
        detections = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence,
                'class': self.model.names[int(class_id)]
            })
        
        return annotated_frame, detections
    
    def run_webcam(self, camera_id=0, conf_threshold=0.5):
        """
        Run real-time object detection on webcam
        camera_id: ID ของกล้อง (0 = กล้องหลัก)
        conf_threshold: threshold ความมั่นใจในการตรวจจับ (0-1)
        """
        # เปิดกล้อง
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: ไม่สามารถเปิดกล้องได้")
            return
        
        # ตั้งค่าความละเอียด
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("กำลังเริ่มต้นการตรวจจับวัตถุ...")
        print("กด 'q' เพื่อออกจากโปรแกรม")
        print("กด '+' เพื่อเพิ่ม confidence threshold")
        print("กด '-' เพื่อลด confidence threshold")
        
        fps_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: ไม่สามารถอ่านภาพจากกล้องได้")
                break
            
            # ตรวจจับวัตถุ
            annotated_frame, detections = self.detect_objects(frame, conf_threshold)
            
            # คำนวณ FPS
            current_time = time.time()
            fps = 1 / (current_time - fps_time)
            fps_time = current_time
            
            # แสดงข้อมูล FPS และ threshold
            info_text = f'FPS: {fps:.1f} | Threshold: {conf_threshold:.2f} | Objects: {len(detections)}'
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # แสดงผล
            cv2.imshow('YOLOv13 Object Detection', annotated_frame)
            
            # จัดการ keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
            elif key == ord('-'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
        
        # ปิดทุกอย่าง
        cap.release()
        cv2.destroyAllWindows()
        print("ปิดโปรแกรมเรียบร้อย")

def main():
    """
    Main function
    """
    # สร้าง detector
    # คุณสามารถเปลี่ยนโมเดลได้ เช่น:
    # - yolo11n.pt (nano - เร็วที่สุด)
    # - yolo11s.pt (small)
    # - yolo11m.pt (medium)
    # - yolo11l.pt (large)
    # - yolo11x.pt (extra large - แม่นที่สุด)
    
    detector = YOLOv13Detector(model_path='yolo11n.pt')
    
    # เริ่มการตรวจจับแบบ real-time
    detector.run_webcam(camera_id=0, conf_threshold=0.5)

if __name__ == "__main__":
    main()