import cv2
from pathlib import Path
from simple_face import FaceAI

example_dir = Path(__file__).resolve().parent
known_image = example_dir / "zaim.png"

ai = FaceAI()

attendance = []
names = None

if not ai.add_person("zaim", str(known_image)):
    raise RuntimeError("Registration failed: test.jpg must contain a clear face.")


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read frame.")
            break

        faces = ai.recognizer_wrap.detect_faces(frame)

        
        if faces is not None and len(faces) > 0:
            status_text = f"Welcome my King {names}"
            text_color = (0, 255, 0)  
            
            
            for face in faces:
                name, score = ai._process_and_draw_face(frame, face)
                
                if name not in attendance:
                    if name == "Unkown":
                        continue
                    else:
                        names = name
                        attendance.append(name)
                        print(f"{name} have attend the class to day")
                else:
                    continue        
                
        else:
            status_text = "Toa kichwa hapo "
            text_color = (0, 0, 255)  

        
        cv2.putText(
            frame, 
            status_text, 
            (30, 50),                  
            cv2.FONT_HERSHEY_SIMPLEX,  
            0.8,                       
            text_color,                
            2,                         
            cv2.LINE_AA                
        )


        cv2.imshow("Custom OpenCV Face Recognition", frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()