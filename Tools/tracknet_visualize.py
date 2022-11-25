import cv2
import os
import csv
import queue
import argparse


parser = argparse.ArgumentParser(description = 'Video & TrackNet CSV Visualization Tool')
parser.add_argument('--video', type=str, required=True, help = 'video file name')
parser.add_argument('--csv', type=str, required=True, help = 'csv(should contain Frame,Visibility,X,Y)')

args = parser.parse_args()

videofile_basename = os.path.splitext(os.path.basename(args.video))[0] # 去除副檔名
# 目的資料夾
directory = os.path.join('./output/',videofile_basename)

# 如果沒有該資料夾則建立
if not os.path.exists(directory):
    os.makedirs(directory)

# 讀取CSV檔案
frame = {}
visibility = {} # 有無辨識到球
x = {} # 球的x座標
y = {} # 球的y座標
with open(args.csv, newline='') as csvfile:
    rows = csv.DictReader(csvfile) # 將第一列當作欄位的名稱，將往後的每一列轉成Dictionary

    # 讀取每一列(row)
    for i, row in enumerate(rows):
        frame[i] = int(row['Frame'])
        visibility[i] = int(row['Visibility']) # 將欄位 的值Visibility 加到 visibility
        x[i] = (float(row['X'])) # 將欄位 X 的值加到 x
        y[i] = (float(row['Y'])) # 將欄位 Y 的值加到 y

# 讀取範例影片
video = cv2.VideoCapture(args.video)

# 記錄目前處理到第幾幀(從0開始)
currentFrame = 0

# 影片的FPS (影片每秒有多少張相片)
fps = int(video.get(cv2.CAP_PROP_FPS))

# 影片的寬
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

# 影片的高
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# cv2的影片編碼方式
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 欲輸出之影片名稱以及相關設定
output_video_path = os.path.join(directory,videofile_basename+'_visualize.mp4')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width,output_height))

while(True):
    # 從影片讀取一張相片 (Frame)
    success, image = video.read()

    # 如果讀取不到Frame則離開 (代表影片讀取終了)
    if not success: 
        break
    if currentFrame not in visibility or visibility[currentFrame] == 0:
        pass
    else:
        draw_x = int(x[currentFrame]) # 欲畫之羽球中心座標x (以圖片左上角為原點向右)
        draw_y = int(y[currentFrame]) # 欲畫之羽球中心座標y (以圖片左上角為原點向下)
        # 在image上(x,y)的位置畫上大小為4，顏色(B,G,R)為紅色(0,0,255)，線條寬度為-1(-1代表實心)的圓
        cv2.circle(image,(draw_x, draw_y), 8, (0,0,255), -1)
        cv2.putText(image, f"{frame[currentFrame]}", (draw_x, draw_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 欲輸出之圖片檔案名稱 (frame0001.jpg, frame0002.jpg, frame0003.jpg ......)
    filename = os.path.join(directory, 'frame{:0>4d}.jpg'.format(currentFrame))

    # 將相片寫入圖片檔案中
    cv2.imwrite(filename, image)

    print('將 Frame {} 輸出至檔案 {}'.format(currentFrame, filename))

    output_video.write(image)

    # 將Frame Index值加一
    currentFrame = currentFrame + 1

# 釋放影片資源
video.release()
output_video.release()
