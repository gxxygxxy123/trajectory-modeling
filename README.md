# 羽球軌跡建模與預測

## Resource
Paper： https://etd.lib.nctu.edu.tw/cgi-bin/gs32/tugsweb.cgi?o=dnctucdr&s=id=%22GT309551034%22.&searchmode=basic  
Dataset: https://drive.google.com/drive/folders/1KaPYvxVkpDieZUxfACg5bTdc8senjDDC?usp=sharing   

## Introduction
The capability of shuttlecock trajectory prediction is a turnkey technique to develop an interactive badminton bot. In this work, several shuttlecock trajectory modeling methods are proposed for trajectory prediction, including query-based, formula-based, and RNN-based. There are four main tasks in the development of the model. First, a shuttlecock trajectory dataset is collected and the data is pre-processed(smooth) appropriately. Second, various modeling methods are developed including query-based model, formula-based model, and RNN-based model for trajectory prediction. Third, for RNN-based model training, we utilize formula-based model to initialize a RNN-based model. Last, performance evaluation are provided, and the pros and cons of these approaches are considered. To avoid huge dataset collection for deep learning training, the formula-based model is utilized to generate training dataset to initialize deep learning model. We also estimate the drag coefficient of shuttlecock from real data.
In conclusion, Query models indicate the regularity in the shuttlecock trajectory. They show similar results with other models. However, it requires data for building a database and the cost time increases as the query database expanded. Formula-based models perform a good result in 100ms and 200ms, but it is hard to estimate the pitch angle, velocity and the drag coefficient in 50ms and lead to unfavorable results. Transformer performs well but may be inappropriate in the real-time shuttlecock trajectory prediction problem because the computation of time is large and the output length is uncertain. The maximum decoder output needs to be set to a large value especially in high FPS and it takes more time to predict because it unrolls each point in testing time. BLSTM performs well in 50ms and has similar results with formula-based models in 100ms in same drag coefficients and the computation of time is small enough to deal with the real cases.

## System Environment
- Ubuntu 20.04
- NVIDIA GeForce RTX 3080 Ti
    * CUDA 11.4,  Driver Version: 470.141.03  
    (cuda, cudnn and tensorflow, tutorial: https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)
- Python 3.8.10
    * pandas
    * numpy
    * scipy
    * sklearn
    * skspatial
    * shapely
    * matplotlib
    * seaborn
    * Pytorch 1.10.0+cu113
    * Opencv 4.2.0
    * MQTT
    * PyQt5 (UI)
    * PyOpenGL (UI)
- Environment installation
    ```
    $ pip3 install pandas
    $ pip3 install numpy
    $ pip3 install scipy
    $ pip3 install scikit-learn
    $ pip3 install scikit-spatial
    $ pip3 install shapely
    $ pip3 install matplotlib
    $ pip3 install seaborn
    $ pip3 install opencv-python
    $ pip3 install paho-mqtt
    $ pip3 install PyQt5
    $ pip3 install pyqt5-tools
    $ pip3 install PyOpenGL
    ```

## Codes File Structure
```
📦trajectory-modeling
│   README.md  
│
├───📂RNN
│   │   trainBLSTM.py
│   │   testBLSTM.py
│   │   trainTF.py
│   │   testTF.py
│   │   query1.py
│   │   query2.py
│   │   testPhysic.py
│   │   ...
│   │
│   ├───📂transformer (for model layers)
│   ├───📂weight
│   │
│   └───📂utils
│       │   velocity.py
│       │   error_function.py
│       │   ...
│   
├───📂Tools (some required PyQt5/PyOpenGL)
│   │   label_tool.py
│   │   extrinsic_tool.py
│   │   ...
│
│
└───📂an_task
    │   smooth.py
    │   plot_dataset.py
    │   ...
```

### RNN📂
4種羽球軌跡預測模型
1. Aerodynamics Model (http://www.shenlj.cn/en/IACSS2019_SHEN.pdf)
2. Query-based Models 1 & 2
3. BLSTM
4. Transformer (https://github.com/FGiuliari/Trajectory-Transformer)

#### utils📂
- error_function.py (計算羽球軌跡預測之誤差Spatial Error, Spatial-temporal Error)  
<img src="https://i.imgur.com/JN1VyLT.png" width=300 height=100>
<img src="https://i.imgur.com/edqevvk.png" width=300 height=100>  

- param.py (隨機初始化模型參數)  
- predict.py (羽球軌跡預測API -- BLSTM、Transfomer)  
    * mean: Data的平均
    * std: Data的標準差
    * out_time: 欲預測的最大時間
    * fps: Data(軌跡)的FPS
    * touch_ground_stop: 預測到落地即停止 (default: True)  

- velocity.py (計算一段羽球軌跡之初速度Vxy) 
    * speed_t: 欲計算之速度於時間t
    * ang_t: 從speed_t開始往後x秒的點使用linear regression計算其角度
    * V_2point: 使用前兩點計算速度
    * tangent_xt: 使用x-t圖之斜率計算速度  
    <img src="https://i.imgur.com/YrZGZaH.png" width=300 height=200>  
    * vx_t_poly: 原柏宇之方法  

#### weight📂
BLSTM、Transformer pre-trained weights.  
檔案名稱: {Model}\_{FPS}\_{t}\_{alpha}  
例如:BLSTM_120_12_0.242為使用alpha=0.242訓練之120FPS，輸入為0.1秒(12個點)之weight.  
模型之超參數會記錄在weight中，在torch.load(WEIGHT)時可讀取其dict.

#### transformer📂
Transformer之layers.

#### 其餘檔案
- blsm.py (定義BLSTM之layers)  
- dataloader.py
    * 資料集之dataloader
        * smooth_2d: 2D平滑(polynomial regression)  
        * poly: smooth_2d平滑之係數
        * smooth_2d_x_accel: 對x值進行調整(平滑)
    * 物理模型產生軌跡之dataloader
        * in_max_time: 軌跡input時間(給網路訓練用)
        * out_max_time: 軌跡output時間(給網路訓練用)
        * cut_under_ground: 碰到地面及停止
        * noise_t: 對時間t加上noise
        * noise_xy: 對x,y加上noise
        * dxyt: 給網路訓練用，訓練資料為點與點之間的向量
        * network_in_dim: 給網路訓練用，2為(X,Y)，3為(X,Y,t)
        * drop_mode: 0為完整資料，1為不完整資料但連續，2為隨機drop點 (Deprecate)
        * fps_range: 軌跡input產生的FPS範圍，預設120.0~120.0
        * elevation_range: 軌跡產生的仰角範圍
        * speed_range: 軌跡產生的速度範圍(km/hr)
        * output_fps_range: = 軌跡output的FPS範圍，預設120.0~120.0
        * starting_point: 軌跡的起始X,Y,Z
        * model: 訓練model時，BLSTM與TF有不同input。model = {'BLSTM','TF'}
        * alpha: 空氣阻力加速度(default: 0.2151959552)
        * g: 重力(default: 9.81)
    * 讀取蒐集的Dataset
    ```
    d = RNNDataSet(dataset_path=資料集路徑, fps=資料集FPS)
    d.whole_2d() --- 2D軌跡 (預設為已平滑，若smooth_2d=True)
    d.whole_3d() --- 3D軌跡 (預設為未平滑)
    d.whole_3d2d() --- 3D軌跡 (已投影至平面上但仍維持3D座標，已平滑，若smooth_2d=True)
    ```

- individual_TF.py (整個Transformer的架構)  
- physic_model.py (空氣動力模型的API)  
    提供了兩種方式  
    * physics_predict3d: 輸入兩個點，以兩個點的速度及仰角使用空氣動力模型預測。
    * physics_predict3d_v2: 輸入起始點及速度，使用空氣動力模型預測。(因速度方向可能震動或有誤差，可使用utils.velocity計算穩定的方向，詳情可參考論文)  
- query1.py (Query-based Model 1)  
- query2.py (Query-based Model 2)  
- testBLSTM.py (測試BLSTM)
    ```
    python3 testBLSTM.py -t 0.1 -w WEIGHT --folder DATASET_FOLDER --fps DATASET_FPS [--draw_predict] [--no_show]
    ```
    * t為軌跡輸入時間。例如：0.1, 0.2 ...  
    * w為預訓練模型  
    * folder 為測試資料集  
    * fps 為資料集FPS.
    * draw_predict 為將預測的軌跡畫出來。實線為輸入，虛線為GT，透明線為PD。(會跑一小段時間)  
    <img src="https://i.imgur.com/4w4L4TE.png" width=200 height=150>  
    * no_show 為不plt.show()。  

- testPhysic.py (測試空氣阻力模型)
    ```
    python3 testPhysic.py -t 0.1 --folder DATASET_FOLDER --fps DATASET_FPS  [--ang_time ANG_TIME] (--V_2point | --tangent_xt) [--draw_predict] [--no_show] [--alpha ALPHA]
    ```
    * ang_time 為使用輸入資料之最後?秒來計算角度。  
    * V_2point/tangent_xt 擇一。可參考util.velocity。  

- testTF.py (測試BLSTM)
    ```
    python3 testBLSTM.py -t 0.1 -w WEIGHT --folder DATASET_FOLDER --fps DATASET_FPS [--draw_predict] [--no_show]
    ```
- threeDprojectTo2D.py (smoothing、投影等常用函式 (有些用不到的可刪除，過舊，可能有BUG))
    * FitVerticalPlaneTo2D 將羽球軌跡投影到垂直地面之最小平面，並smoothing。(可參考論文)  

- trainBLSTM.py (訓練BLSTM。超參數與引數可參考其程式碼)
    ```
    trainBLSTM.py [-t TIME] [-e EPOCH] [--hidden_size HIDDEN_SIZE] [--hidden_layer HIDDEN_LAYER] [--physics_data PHYSICS_DATA] [--batch_size BATCH_SIZE] [--lr LR] [--save_epoch SAVE_EPOCH]
                        [--fig_pth FIG_PTH] [--wgt_pth WGT_PTH] [--noisexy] [--alpha ALPHA]
    ```
    * t為軌跡輸入時間。例如：0.1, 0.2 ...  
    * e 為訓練EPOCH數  
    * save_epoch 為每多少個epoch儲存weight
    * fig_pth 為inference預測之輸出圖片路徑(default: ./figure/BLSTM/)
    * wgt_pth 為weight儲存路徑(default: ./weight/BLSTM/)
    * noisexy 為訓練資料增加3cm左右的noise(default: False)
    * alpha 空氣動力模型參數

- trainTF.py (訓練Transformer。超參數與引數可參考其程式碼)  
    ```
    trainTF.py [-t TIME] [-e EPOCH] [--physics_data PHYSICS_DATA] [--batch_size BATCH_SIZE] [--save_epoch SAVE_EPOCH] [--fig_pth FIG_PTH] [--wgt_pth WGT_PTH] [--noisexy] [--alpha ALPHA]
                    [--emb_size EMB_SIZE] [--heads HEADS] [--layers LAYERS] [--d_ff D_FF] [--dropout DROPOUT] [--factor FACTOR] [--warmup WARMUP]
    ```

- alpha.py (找出資料集裡面每一小段軌跡時間為T之初速度、角度及物理產生之最相近軌跡的alpha)  
    ```
    python3 alpha.py --folder DATASET_FOLDER --fps DATASET_FPS -t T
    ```
    <img src="https://i.imgur.com/VIHI6pk.png" width=300 height=200>  

#### 我自己論文的小檔案
- draw_time_after.py (配合test時儲存的pkl繪製該圖片)  
    <img src="https://i.imgur.com/TKWNrwV.png" width=200 height=150>  


### an_task📂
家安的論文部分程式

- 3d_camera.py (我們攝影系統與VICON系統的比較)  
    <img src="https://i.imgur.com/ZSILzBr.png" width=300 height=180>  
    ```
    python3 3d_camera.py
    ```
- plot_dataset.py (軌跡資料集Overview)  
    <img src="https://i.imgur.com/MgJ4Va5.png" width=400 height=250>  
    ```
    python3 plot_dataset.py --folder FOLDER --fps FPS [--ang_time ANG_TIME] [--dt DT] [--no_smooth] [--each] [--heatmap] [--no_show] (--V_2point | --tangent_xt)
    ```
    *  --folder FOLDER      資料集路徑
    *  --fps FPS            資料集FPS
    *  --ang_time ANG_TIME  以軌跡前ANG_TIME秒計算仰角 (default: 0.05)
    *  --dt DT              將軌跡切成每DT為一段 (default: 0.1)
    *  --no_smooth          未經平滑的Dataset
    *  --each               顯示每一筆軌跡
    *  --heatmap            顯示每一小段的軌跡heatmap
    *  --no_show            No plt.show()
    *  --V_2point           使用前兩點計算速度
    *  --tangent_xt         使用x-t圖之斜率計算速度  

- remove_point.py (軌跡去頭之成效 (Deprecate))  

- smooth.py (軌跡平滑的效果)  
    <img src="https://i.imgur.com/GUi3dGw.png" width=200 height=120>  
    ```
    python3 smooth.py --folder FOLDER [--fps FPS] --by BY [--each]
    ```
    *  folder FOLDER      資料集路徑
    *  fps FPS            資料集FPS
    *  by {'point','trajectory'} 以點或一條軌跡為單位計算平滑前後誤差
    *  each               顯示每條軌跡平滑的效果

- validate_physic_model.py (驗證空氣阻力模型和資料集)  
    <img src="https://i.imgur.com/VQZK4Lv.png" width=300 height=220>  
    ```
    python3 validate_physic_model.py --folder FOLDER --fps FPS [--ang_time ANG_TIME] [--offset_t OFFSET_T] [--smooth] [--poly POLY] (--V_2point | --tangent_xt)
    ```
    *  folder FOLDER      資料集路徑
    *  fps FPS            資料集FPS
    *  ang_time ANG_TIME  以軌跡前ANG_TIME秒計算仰角 (default: 0.05)
    *  offset_t OFFSET_T  計算速度仰角時，忽略前OFFSET_T的軌跡(default: 0)
    *  smooth             將軌跡平滑 (通常要加!!)
    *  poly               平滑polynomial regression係數
    *  V_2point           使用前兩點計算速度
    *  tangent_xt         使用x-t圖之斜率計算速度 


### Tools📂
常用工具

- cut_video.py (將蒐集回來的羽球資料剪輯)
    ```
    python3 cut_video.py [VIDEO(.mp4/.avi)]
    ```
    大小寫不限
    * Z: 上一張圖片
    * X: 下一張圖片
    * C: 清除start, end
    * V: 上N張圖片(預設20，程式碼裡面開頭可調)
    * B: 下N張圖片(預設20，程式碼裡面開頭可調)
    * S: 於此張圖片紀錄start
    * E: 於此張圖片紀錄end，並剪取start~end的這一段影片，若同目錄底下有VIDEO.csv則一併剪取，預設儲存於./tmp/
    * Q: 離開程式
    * ESC: 離開程式
    * Y: 互動時輸入(yes)

- do_3d.py (對資料集裡的軌跡檔案重新計算一次3D)  
    ```
    python3 do_3d.py --folder [DATASET FOLDER]
    ```

- extrinsic_tool.py (計算影片外部參數(預設為讀取coachAI/replay/裡的所有影片))
    ```
    python3 extrinsic_tool.py
    ``` 
    <img src="https://i.imgur.com/kv755ft.png" width=320 height=180><img src="https://i.imgur.com/yYqXqxS.png" width=100 height=180>  

- fake_ts.py (將FOLDER中影片之ts重新依照FPS從0以等差數列計算(若相機ts有誤))  
- label_tool.py (標記程式(可參考：https://hackmd.io/@gxxygxxy123/Hy2GDTlqY))  
    ```
    python3 label_tool.py [VIDEO(.mp4/.avi)]
    ``` 
- remove_pre_post_point.py (移除軌跡前後的點 (當初軌跡去頭去彈地用的，非必要別用))  
- tracknet_visualize.py (將2D csv標記檔繪製於video上)  
    ```
    python3 tracknet_visualize.py --video [VIDEO(.mp4/.avi)] --csv [CSV]
    ``` 
- trajectories_opengl.py (於顯示OpenGL顯示軌跡(DATASET))  
    ```
    python3 trajectories_opengl.py --folder [DATASET FOLDER]
    ``` 
    <img src="https://i.imgur.com/rwLc08R.png" width=320 height=180>  

- video_to_images.py (將影片依幀輸出Frames)  
    ```
    python3 video_to_images.py [VIDEO(.mp4/.avi)]
    ``` 

## Others

### Image Source 錄影範例 (deprecate)
```
$ python3 Reader/Image_Source/main.py --nodename CameraReaderL
$ mosquitto_pub -h localhost -t cam_control -m "{\"recording\": true, \"path\": \".\/replay\/test\/\"}"
$ mosquitto_pub -h localhost -t cam_control -m "{\"recording\": false}"
```
- [ ] Camera Synchronization. Cameras start recording at same time precisely.
- [ ] Camera capture time, the PTS timestamp is wrong. (https://www.theimagingsource.com/documentation/tiscamera/timestamps.html)
