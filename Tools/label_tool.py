import cv2
import sys
import os
import pandas as pd

### KEY ###
KEY_LastFrame = 'z'
KEY_NextFrame = 'x'
KEY_ClearFrame = 'c'
KEY_Event1 = '1'
KEY_Event2 = '2'
KEY_Event3 = '3'
KEY_Zoom = 'a'
KEY_Save = 's'
KEY_Quit = 'q'
KEY_Esc = 27
KEY_Up = 'i'
KEY_Left = 'j'
KEY_Down = 'k'
KEY_Right = 'l'

### Video ###
VIDEO_NAME = sys.argv[1]

### CSV ###
CSV_NAME = os.path.splitext(VIDEO_NAME)[0] + '_ball.csv'
COLUMNS = ['Frame', 'Visibility', 'X', 'Y', 'Z', 'Event', 'Timestamp']

### Referenced CSV to keep timestamp ###
REFERED_CSV_NAME = os.path.splitext(VIDEO_NAME)[0] + '.csv'
refered_time_flag = False
if os.path.exists(REFERED_CSV_NAME):
    refered_df = pd.read_csv(REFERED_CSV_NAME)
    refered_time_flag = True
else:
    print("[WARNING] No Referenced timestamp csv file")



### Other Variables ###
ZOOM_SCALE = 2.5


### Label Color ###
# Event: Color
LABEL_COLOR = {0: (0, 0, 255),
               1: (0, 255, 255),
               2: (0, 255, 0),
               3: (255, 0, 0)}
LABEL_SIZE = 4
ZOOM_LABEL_SIZE = 2
TEXT_SIZE = 0.8
TEXT_COLOR = (0, 255, 255)

def empty_row(frame_idx : int, timestamp : float):
    return {'Frame': frame_idx, 'Visibility': 0, 'X': 0.0, 'Y': 0.0, 'Z': 0.0, 'Event': 0, 'Timestamp': timestamp}

def dict_save_to_dataframe(df):
    pd_df = pd.DataFrame.from_dict(df, orient='index', columns=COLUMNS)
    pd_df.to_csv(CSV_NAME, encoding = 'utf-8',index = False)

def on_Mouse(event, x, y, flags, param):
    global img, scale_X, scale_Y
    if event == cv2.EVENT_LBUTTONDOWN:
        if scale_flag:
            x = float(x + scale_X - half_W)
            y = float(y + scale_Y - half_H)
        df[current_frame_idx]['Visibility'] = 1
        df[current_frame_idx]['X'] = x
        df[current_frame_idx]['Y'] = y

    elif event == cv2.EVENT_RBUTTONDOWN:
        if scale_flag:
            x = float(x + scale_X - half_W)
            y = float(y + scale_Y - half_H)
        df[current_frame_idx]['Visibility'] = 1
        df[current_frame_idx]['X'] = x
        df[current_frame_idx]['Y'] = y
        df[current_frame_idx]['Event'] = 1

def update_frame(cap, current_frame_idx, current_row):
    global scale_flag, scale_X, scale_Y
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    scale_X = None
    scale_Y = None
    if scale_flag:
        scale_flag = False
        for i in range(current_frame_idx-1, -1, -1):
            if df[i]['Visibility']:
                scale_X = int(df[i]['X'])
                scale_Y = int(df[i]['Y'])
                scale_flag = True
                break

        if scale_flag:
            scale_X = min(width - half_W, max(scale_X, half_W))
            scale_Y = min(height - half_H, max(scale_Y, half_H))
            frame = frame[scale_Y - half_H : scale_Y + half_H,
                          scale_X - half_W : scale_X + half_H]

    # if Ball, Draw Color on label
    if current_row['Visibility']:
        if scale_X and scale_Y:
            frame = cv2.circle(frame, (int(current_row['X'] - scale_X + half_W), int(current_row['Y'] - scale_Y + half_H)), ZOOM_LABEL_SIZE, LABEL_COLOR[current_row['Event']], -1)
        else:
            frame = cv2.circle(frame, (int(current_row['X']), int(current_row['Y'])), LABEL_SIZE, LABEL_COLOR[current_row['Event']], -1)
    frame = cv2.putText(frame, f"Frame: {current_row['Frame']}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)
    frame = cv2.putText(frame, f"Event: {current_row['Event']}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, 2)
    return frame

if __name__ == '__main__':
    print(f"-------------------------\n"
          f"| Last Frame:          {KEY_LastFrame}\n"
          f"| Next Frame:          {KEY_NextFrame}\n"
          f"| Clear Label & Event: {KEY_ClearFrame}\n"
          f"| Event 1 (Hit):       {KEY_Event1}\n"
          f"| Event 2 (Land):      {KEY_Event2}\n"
          f"| Event 3 (Serve):     {KEY_Event3}\n"
          f"| Zoom:                {KEY_Zoom}\n"
          f"| Save:                {KEY_Save}\n"
          f"| Save & Quit:         {KEY_Quit}\n"
          f"| No Save & Quit:      Esc\n"
          F"| Move 1 pixel:        {KEY_Up},{KEY_Left},{KEY_Down},{KEY_Right}\n"
          f"-------------------------\n")

    # Read Video
    cap = cv2.VideoCapture(VIDEO_NAME)
    total_frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)



    if os.path.isfile(CSV_NAME): # Exist csv file
        df = pd.read_csv(CSV_NAME)
        assert all(df.columns == COLUMNS), f"{CSV_NAME} column format is not {COLUMNS} !"
        df = df.to_dict('index')
        if len(df) == 0:
            if refered_time_flag:
                df[0] = empty_row(frame_idx=0, timestamp=refered_df['Timestamp'].iloc[0])
            else:
                df[0] = empty_row(frame_idx=0, timestamp=0.0)
    else: # New File
        df = {}
        if refered_time_flag:
            df[0] = empty_row(frame_idx=0, timestamp=refered_df['Timestamp'].iloc[0])
        else:
            df[0] = empty_row(frame_idx=0, timestamp=0.0)
    current_frame_idx = len(df) - 1 # Jump to last frame idx
    print(f"Start from Frame: {current_frame_idx}")


    half_W = int(width/ZOOM_SCALE//2)
    half_H = int(height/ZOOM_SCALE//2)

    print (f"Total Frame: {total_frame_idx}")
    print (f"Width: {width}")
    print (f"Height: {height}")
    print (f"Fps: {fps}")

    scale_flag = False
    scale_X = None
    scale_Y = None

    img = update_frame(cap, current_frame_idx, df[current_frame_idx])
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_Mouse, img)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # insert a initial row
        if current_frame_idx >= len(df): # new row
            if refered_time_flag:
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=refered_df['Timestamp'].iloc[current_frame_idx])
            else:
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=current_frame_idx/fps)
        # display the image and wait for a keypress

        img = update_frame(cap, current_frame_idx, df[current_frame_idx])
        if img is None:
            print(f"Frame {current_frame_idx} is broken, save and quit. Please report it.")
            dict_save_to_dataframe(df)
            break

        cv2.imshow('image', img)
        key = cv2.waitKeyEx(1)
        if key & 0xFFFF == ord(KEY_NextFrame):
            if current_frame_idx < total_frame_idx-1:
                current_frame_idx += 1
            if current_frame_idx >= len(df):
                if refered_time_flag:
                    df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=refered_df['Timestamp'].iloc[current_frame_idx])
                else:
                    df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=current_frame_idx/fps)
            print(f"Current Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_LastFrame):
            if current_frame_idx > 0:
                current_frame_idx -= 1
            print(f"Current Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event1):
            df[current_frame_idx]['Event'] = 1
            print(f"{KEY_Event1} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event2):
            df[current_frame_idx]['Event'] = 2
            print(f"{KEY_Event2} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Event3):
            df[current_frame_idx]['Event'] = 3
            print(f"{KEY_Event3} Pressed At Frame: {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Save):
            dict_save_to_dataframe(df)
            print(f"Save At Frame {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Quit):
            dict_save_to_dataframe(df)
            print(f"Save At Frame {current_frame_idx}")
            break

        elif key & 0xFFFF == ord(KEY_ClearFrame): # clear label of this frame
            if refered_time_flag:
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=refered_df['Timestamp'].iloc[current_frame_idx])
            else:
                df[current_frame_idx] = empty_row(frame_idx=current_frame_idx, timestamp=current_frame_idx/fps)
            print(f"Clear Frame {current_frame_idx}")

        elif key & 0xFFFF == ord(KEY_Up): # press keyboard direction up
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['Y'] -= 1

        elif key & 0xFFFF == ord(KEY_Down): # press keyboard direction down
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['Y'] += 1

        elif key & 0xFFFF == ord(KEY_Left): # press keyboard direction left
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['X'] -= 1

        elif key & 0xFFFF == ord(KEY_Right): # press keyboard direction right
            if df[current_frame_idx]['Visibility']:
                df[current_frame_idx]['X'] += 1

        elif key & 0xFFFF == ord(KEY_Zoom):
            scale_flag = not scale_flag

        elif key & 0xFFFF == KEY_Esc:
            break

