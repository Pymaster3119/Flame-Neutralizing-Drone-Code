import RPi.GPIO as GPIO
import time
import threading
from picamera2 import Picamera2
import numpy as np
import onnxruntime as ort
import send_emails

#ONNX setup
model_name = "/home/gset/Documents/onnx_exported_model.onnx"
session = ort.InferenceSession(model_name)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#PIN CONFIG
buzzer1 = 18
buzzer2 = 13
leds = [17,27,22,10,9]
L_1 = 6
L_2 = 11
Flame_sensor = 16

#Set up RPi pins & camera
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer1, GPIO.OUT)
GPIO.setup(buzzer2, GPIO.OUT)
GPIO.setup(L_1, GPIO.OUT)
GPIO.setup(L_2 , GPIO.OUT)
for i in leds:
        GPIO.setup(i, GPIO.OUT)
picam2 = Picamera2()
config = picam2.create_still_configuration(main = {"size":(256,256)})
picam2.configure(config)
picam2.start()
GPIO.setup(Flame_sensor, GPIO.IN)

def grab_image():
        picam2.capture_file("/home/gset/Desktop/img.jpg")
        return picam2.capture_array()

def run_cnn(img_data):
        img_data = img_data.astype(np.float32) / 256
        img_data = np.transpose(img_data, (2, 0, 1)) 
        img_data = np.expand_dims(img_data, axis=0)
        outputs = session.run([output_name], {input_name: img_data})
        outputs = 1/(1+np.exp(-outputs[0]))
        print(outputs)
        return outputs > 0.5    
        
def LEDBlink():
        while True:
                if alarmStarted:
                        for i in leds:
                                GPIO.output(i, GPIO.HIGH)
                        time.sleep(1)
                        for i in leds:
                                GPIO.output(i, GPIO.LOW)
                        time.sleep(1)
        
def run_buzzers():
        while True:
                if alarmStarted:
                        pwm.ChangeDutyCycle(30)
                        time.sleep(1)
                        pwm.ChangeDutyCycle(100)
                        time.sleep(1)
                else:
                        return
        
def stop_alarm():
        alarmStarted = False
        pwm.stop()

def start_alarm():
        global start_time
        start_time = time.time()
        global alarmStarted
        if not alarmStarted:
                alarmStarted = True
                alarmthread = threading.Thread(target = run_buzzers)
                alarmthread.start()
                ledthread = threading.Thread(target = LEDBlink)
                ledthread.start()
                picam2.capture_file("/home/gset/Desktop/img.jpg")
                #send_emails.SendEmail(send_emails.URLGenerator.CreateIMGBBURL("/home/gset/Desktop/img.jpg", "d7f80c64d>
                start_linact()
        print(time.time() - start_time)
def start_linact():
        global alarmStarted
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(6, GPIO.LOW)
        time.sleep(10)
        GPIO.output(11, GPIO.LOW)
        GPIO.output(6, GPIO.HIGH)
                
pwm = None
alarmStarted = False
alarmthread = None
ledthread = None
start_time = 0
predictions = []
percentframesrequired = 0.6
predperiod = 10
try:
        start = time.time()
        num_iter = 0
        pwm = GPIO.PWM(buzzer1, 100)
        pwm.start(0)
        while True:
                start_time = time.time()
                print('-' * 67)
                img = grab_image()
                end = time.time()
                num_iter += 1
                pred = run_cnn(img)[0][0].item()
                pred_fire = GPIO.input(Flame_sensor) == GPIO.HIGH
                predictions.append(pred)
                pred_total = False
                if len(predictions)==11:
                        del predictions[0]
                        ncorr = 0
                        for predi in predictions:
                                ncorr += 1 if predi else 0
                        if ncorr/10 >= percentframesrequired:
                                pred_total = True
                                
                print(f'CNN Prediction: {pred}')
                print(f'Fire prediction: {pred_fire}')
                print(f'Past predictions: {predictions}')
                print(f'Prediction total: {pred_total}')
                if pred_total:# and pred_fire:
                        print("Here")
                        start_alarm()       
except KeyboardInterrupt:
        print('DONE')
finally:
        pwm.stop()
        GPIO.output(buzzer1, GPIO.LOW)
        for i in leds:
                GPIO.output(i, GPIO.LOW)
        GPIO.cleanup()
        picam2.stop()