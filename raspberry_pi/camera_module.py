from picamera2 import Picamera2
import requests
import json
import threading
import base64
import time
import cv2
from audio_output import DFPlayer
from collections import deque
import gc

df = DFPlayer()
df.set_volume(20)

with open('config.json') as config_file:
    config = json.load(config_file)

SERVER_URL = config['server_url']
RESOLUTION = config['camera_settings']['resolution']
FRAME_RATE = config['camera_settings']['frame_rate']

width, height = map(int, RESOLUTION.split('x'))
frame_counter = 0


AUDIO_DICT = {
    "bad": 1,
    "beautiful": 2,
    "goodbye": 3,
    "happy": 4,
    "hello": 5,
    "hungry": 6,
    "love": 7,
    "sorry": 8,
    "thank you": 9,
}

running = True
last_played_word = ""
last_play_time = 0
play_lock = threading.Lock()
result_buffer = deque(maxlen=30)  
result_lock = threading.Lock()


def play_audio_immediately(text, confidence):
    global last_play_time, last_played_word
    with play_lock:
        now = time.time()
        text = text.lower().strip()
        if confidence >= 0.7:
            if text != last_played_word or now - last_play_time >= 2:
                try:
                    if text.isdigit():
                        num = int(text)
                        if 1 <= num <= 9:
                            df.play_track(file_number=num)
                            print(f"Play number: {num} ({confidence:.2f})")
                    elif text in AUDIO_DICT:
                        df.play_track(file_number=AUDIO_DICT[text])
                        print(f"Play word: {text} ({confidence:.2f})")
                    last_played_word = text
                    last_play_time = now
                except Exception as e:
                    print(f"Audio error: {e}")

def process_results():
    while running:
        with result_lock:
            if len(result_buffer) == 30:  
                best_result = max(result_buffer, key=lambda x: x[1])
                text, confidence = best_result
                print(f"[BEST] {text} ({confidence:.2f})")
                play_audio_immediately(text, confidence)
                result_buffer.clear()
        time.sleep(0.1)

def send_image_to_server(image_data, frame_id):
    try:
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        payload = {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "frame_id": frame_id
        }

        response = requests.post(SERVER_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            recognized_text = result.get('recognized_text', '')
            confidence = result.get('confidence', 0)
            print(f"Frame {frame_id}: {recognized_text} ({confidence:.2f})")
            
            if recognized_text.strip():
                with result_lock:
                    result_buffer.append((recognized_text, confidence))
        else:
            print(f"Server error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        del image_data
        gc.collect()
        
def capture_and_send_image():
    global running, frame_counter, last_stream_time
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (width, height)}))
    picam2.start()
    time.sleep(1)

    frame_delay = 1.0 / FRAME_RATE

    try:
        while running:
            frame = picam2.capture_array()
            frame_resized = cv2.resize(frame, (224, 224))
            _, buffer = cv2.imencode('.jpg', frame_resized)
            image_data = buffer.tobytes()

            frame_counter += 1
            threading.Thread(target=send_image_to_server, args=(image_data, frame_counter)).start()

            

            time.sleep(frame_delay)
            
            del frame
            del frame_resized
            del buffer
            gc.collect()
    finally:
        running = False  
        picam2.stop()
        print("Camera released.")

if __name__ == "__main__":
    result_thread = threading.Thread(target=process_results, daemon=True)
    result_thread.start()
    
    try:
        capture_and_send_image()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        running = False  
        time.sleep(0.2) 
        print("Program stopped.")
