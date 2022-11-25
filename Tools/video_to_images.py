import cv2
import os
import sys

path = sys.argv[1]

# 讀取範例影片
video = cv2.VideoCapture(path)

# 目的資料夾
directory = f'./images/{os.path.basename(os.path.splitext(path)[0])}'

# 如果沒有該資料夾則建立
if not os.path.exists(directory):
    os.makedirs(directory)
else:
    print(f"Folder {directory} already exists. Failed.")
    sys.exit(1)


# 記錄目前處理到第幾幀(從0開始)
currentFrame = 0

while True:
	# 從影片讀取一張相片 (Frame)
    success,image = video.read()

    # 如果讀取不到Frame則離開 (代表影片讀取終了)
    if not success:
        break

    # 欲輸出之圖片檔案名稱 (frame0001.jpg, frame0002.jpg, frame0003.jpg ......)
    filename = os.path.join(directory, 'frame{:0>4d}.jpg'.format(currentFrame))

    # 將相片寫入圖片檔案中
    cv2.imwrite(filename, image)

    print('將 Frame {} 輸出至檔案 {}'.format(currentFrame, filename))

    # 將Frame Index值加一
    currentFrame = currentFrame + 1

# 釋放影片資源
video.release()
