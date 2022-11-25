import cv2
import os
import csv
import argparse
import sys
import pandas as pd
import shutil

### KEY ###
N = 20 # Step of key V,B
Key_LastFrame     = ['z','Z']
Key_NextFrame     = ['x','X']
Key_ClearStartEnd = ['c','C']
Key_LastNFrame    = ['v','V']
Key_NextNFrame    = ['b','B']
Key_Start         = ['s','S']
Key_End           = ['e','E']
Key_Quit          = ['q','Q'] # Quit
Key_Esc           = 27        # Quit
Key_Yes           = ['y','Y']

def toInt(df):
    df.Frame = df.Frame.astype('int64')
    df.Visibility = df.Visibility.astype('int64')
    df.Event = df.Event.astype('int64')
    return df

videos = []

videofile = sys.argv[1]

for f in sorted(os.listdir(os.path.dirname(videofile))):
    if f.endswith(".avi") or f.endswith(".mp4"):
        videos.append(os.path.join(os.path.dirname(videofile), f))

video = cv2.VideoCapture(videofile)
total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(video.get(cv2.CAP_PROP_FPS))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print(f"Total frame: {total_frame}, FPS: {fps}, Width: {output_width}, Height: {output_height}")

# total_frame = 2000 (if video too large, avoid READ Error, BAD CODE)

currentFrame = 0
print(f"Current frame: {currentFrame}")

images = [None] * total_frame

i = 0
while(True):
    success, image = video.read()
    if not success or i >= total_frame:
        break
    images[i] = image
    i = i+1

image=images[currentFrame]
cv2.namedWindow("image")

start = None
end = None

output_idx = 1

while(True):
    cv2.imshow("image", cv2.resize(image,(output_width*2//3,output_height*2//3)))
    Key = cv2.waitKey(1) & 0xFF

    if Key in [ord(x) for x in Key_NextFrame]:     #jump next frame
        if currentFrame < total_frame-1:
            if images[currentFrame+1] is not None:
                image = images[currentFrame+1]
                currentFrame += 1
                print(f'Current frame: {currentFrame}')
            else:
                print(f'Frame {currentFrame+1} is broken')
        else:
            print('This is the last frame')
    elif Key in [ord(x) for x in Key_NextNFrame]:     #jump next N frame
        if currentFrame < total_frame-N:
            if images[currentFrame+N] is not None:
                image = images[currentFrame+N]
                currentFrame += N
                print(f'Current frame: {currentFrame}')
            else:
                print(f'Frame {currentFrame+N} is broken')
        else:
            print(f'This is the last {N} frame')
    elif Key in [ord(x) for x in Key_LastFrame]:     #jump last frame
        if currentFrame == 0:
            print('This is the first frame')
        else:
            currentFrame -= 1
            print(f'Current frame: {currentFrame}')
            image = images[currentFrame]
    elif Key in [ord(x) for x in Key_LastNFrame]:     #jump last N frame
        if currentFrame < N:
            print(f'This is the first {N} frame')
        else:
            currentFrame -= N
            print(f'Current frame: {currentFrame}')
            image = images[currentFrame]
    elif Key in [ord(x) for x in Key_Start]:
        if start is None:
            start = currentFrame
            print(f"START AT Frame {currentFrame}")
        else:
            print(f"YOU DIDNT PRESS {','.join(Key_End)} AFTER last {','.join(Key_Start)} !")
            sys.exit(1)
    elif Key in [ord(x) for x in Key_End]:
        end = currentFrame
        if start is None:
            print(f"YOU FORGET TO PRESS {','.join(Key_Start)} !")
            sys.exit(1)
        elif start > end:
            print("THE FRAME OF START > END !")
            sys.exit(1)
        else:
            # First Video
            start_timestamp = None
            end_timestamp = None
            if os.path.exists(os.path.splitext(videofile)[0] + '.csv'):
                df = pd.read_csv(os.path.splitext(videofile)[0] + '.csv')
                start_timestamp = float(df['Timestamp'].iloc[start])
                end_timestamp = float(df['Timestamp'].iloc[end])

            ### Enter file saved path
            #path = input(f'Please Enter Saved Path:')
            path = os.path.join("./tmp/", os.path.basename(os.path.dirname(videofile))+f"_{output_idx}")
            try:
                os.makedirs(path, exist_ok=False)
            except:
                print(f"{path} isn't a path!")
                sys.exit(1)

            # Cut Each Video (Including itself)
            for v in videos:
                if input(f'====== Cut {v} ? (y/Y if yes) ======') in Key_Yes:
                    print(f"YES")
                    csv_file = os.path.splitext(v)[0] + '.csv'

                    ### Reference csv timestamp to cut videos
                    use_csv = False
                    if os.path.exists(csv_file) and start_timestamp and end_timestamp:
                        print(f'Detect {os.path.basename(csv_file)}, cut with timestamp of it!')
                        use_csv = True

                    ### Save csv
                    if use_csv:
                        df = pd.read_csv(csv_file)
                        start_idx = df['Timestamp'].sub(start_timestamp).abs().idxmin()
                        end_idx = df['Timestamp'].sub(end_timestamp).abs().idxmin()
                        _output_csv = os.path.join(path, os.path.basename(csv_file))
                        if os.path.exists(_output_csv):
                            print(f"{_output_csv} already exists!")
                            sys.exit(1)

                        df.iloc[start_idx:end_idx+1,:].to_csv(_output_csv, encoding = 'utf-8', index = False)
                    else:
                        start_idx = start
                        end_idx = end

                    ### Save Video
                    tmp = cv2.VideoCapture(v)
                    _width = int(tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                    _height = int(tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    _fps = float(video.get(cv2.CAP_PROP_FPS))
                    _output_video_path = os.path.join(path, os.path.basename(v))
                    if os.path.exists(_output_video_path):
                        print(f"{_output_video_path} already exists!")
                        sys.exit(1)
                    output_video = cv2.VideoWriter(_output_video_path, fourcc, _fps, (_width, _height))
                    jj = 0
                    while(True):
                        success, image_tmp = tmp.read()
                        if not success:
                            break
                        if jj >= start_idx and jj <= end_idx:
                            output_video.write(image_tmp)
                        jj += 1
                    output_video.release()
                    print(f"====== SAVE {v}, FRAME FROM {start_idx} TO {end_idx} into {_output_video_path}, LENGTH: {end_idx-start_idx+1} ======\n")

                    # Reset start/end
                    start = None
                    end = None

                else:
                    print(f"NO")

            ### Save config
            shutil.copyfile(os.path.join(os.path.dirname(videofile), 'config'), os.path.join(path, 'config'))
            for cfg in sorted(os.listdir(os.path.dirname(videofile))):
                if cfg.endswith(".cfg"):
                    shutil.copyfile(os.path.join(os.path.dirname(videofile), cfg), os.path.join(path, cfg))

            print("DONE")



            output_idx += 1

    elif Key in [ord(x) for x in Key_ClearStartEnd]: # Clear Start and End
        print("Clear Start and End")
        start = None
        end = None
    elif Key in [ord(x) for x in Key_Quit] or Key == Key_Esc:
        if start or end:
            print(f"YOU FORGET TO PRESS {','.join(Key_End)} !")
        break


video.release()
