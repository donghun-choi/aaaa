import pandas as pd
from pynput import keyboard
from datetime import datetime
import time
import os
from mss import mss
from PIL import Image

# WASD 키 상태를 저장할 딕셔너리
key_status = {'w': False, 'a': False, 's': False, 'd': False}

# 키보드 리스너 설정
def on_press(key):
    global key_status
    try:
        k = key.char  # 단일 문자 키
    except:
        k = key.name  # 특수 키
    if k in key_status:
        key_status[k] = True

def on_release(key):
    global key_status
    try:
        k = key.char  # 단일 문자 키
    except:
        k = key.name  # 특수 키
    if k in key_status:
        key_status[k] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# frames 폴더 생성
if not os.path.exists('./frames'):
    os.makedirs('./frames')

# output.csv 파일 준비
with open('output.csv', 'w') as f:
    f.write('frame_count,W,A,S,D\n')

# 초당 60번 캡쳐
frame_count = 0
next_frame_time = time.perf_counter()
sct = mss()
while True:
    # 스크린 캡쳐
    img = sct.grab(sct.monitors[0])
    Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX').save(f'./frames/{frame_count}.jpg')

    # CSV에 키 상태 저장
    with open('output.csv', 'a') as f:
        f.write(f'{frame_count},{int(key_status["w"])},{int(key_status["a"])},{int(key_status["s"])},{int(key_status["d"])}\n')

    frame_count += 1
    next_frame_time += 1/60  # 60Hz
    while time.perf_counter() < next_frame_time:
        pass

