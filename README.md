# hansung_capstone_design
Drone Detecting program with ssd-mobilenet

선행 작업 
1. https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10
2. https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

２-２ DEEP LEARNING – SSD MOBILE NET을 이용한 드론 이미지 학습
２-２-１ tensorflow git hub에서 TensorFlow Object Detection API 저장소 다운
 https://github.com/tensorflow/models 
C 드라이브 tensorflow1폴더생성후 압축풀기 C:\tensorflow1\models 
* 다운로드 버튼 옆에 branch에서 태그 1.14로 꼭 바꿔서 다운(tensorflow version)

２-２-２ 양자화 된 SSD-MobileNet 모델 다운 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
아래쪽 보면 ssd_mobilenet_v2_quantized_coco 다운 
TensorFlow Lite는 Faster-RCNN과 같은 RCNN 모델을 지원하지 않는다.
object_detection폴더에 "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03" 폴더 확인

２-２-３ 트레이닝 및 tfrecord 파일생성 준비하기
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
다운 후 1번에서 만들 폴더안에 C:\tensorflow1\models\research\object_detection에 압축 풀기.
 

２-２-４ 학습준비하기
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
cmd에서 python path 설정
cd C:\tensorflow1\models\research 후 아래 명령어 입력

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

\object_detection\protos 폴더에 프로토파일 생성
C:\tensorflow1\models\research> python setup.py build
C:\tensorflow1\models\research> python setup.py install

build하고 install

２-２-５학습준비하기(2)
C:\tensorflow1\models\research\object_detection\images 폴더에
사진 분배 test폴더에 20퍼센트 train폴더에 80퍼센트
예) 라벨링한 이미지가 100장이면 train에 80장(xml포함하면160) test에 20장(xml포함하면 40)

cmd창에
C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
train_labels.csv 이랑 test_labels.csv가 이미지 폴더에 생성
xml.etree.ElementTree.ParseError: no element found: line 1, column 0 오류 발생시 파일에 문제
object_detection폴더에있는 generate_tfrecord.py파일 열어서 클래스 수정하기
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'nine':
        return 1
    elif row_label == 'ten':
        return 2
    elif row_label == 'jack':
        return 3
    elif row_label == 'queen':
        return 4
    elif row_label == 'king':
        return 5
    elif row_label == 'ace':
        return 6
    else:
        None
위 내용을 아래로 변경하여 저장
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Drone':
        return 1
else:
        None

cmd창에
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
실행하면 학습 레코드파일이랑 테스트 레코드파일 생성

받은 ssd_mobilenet_v2_quantized_coco 파일중에 
graph.pbtxt,   labelmap.pbtxt,   pipeline.config, 
ssd_mobilenet_v2_quantized_300x300_coco.config 4개를
object_detection\training폴더안에 복사 labelmap.pbtxt파일을 열어서

item {
  id: 1
  name: 'Drone'
}
로 변경 레코드파일 만들기 전에 generate_tfrecord.py에서 id값이 같아야 한다.
Training 폴더에 있는ssd_mobilenet_v2_quantized_300x300_coco.config파일 열어서 
9번째 줄에 num_classes : 1로 수정
141번째 줄에 batch_size: 24을 6으로 수정(메모리 에러 줄어듦)
156번째 줄 수정
fine_tune_checkpoint to: 
"C:/tensorflow1/models/research/object_detection/ ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt"
175번째 줄 수정 
input_path to: "C:/tensorflow1/models/research/object_detection/train.record"
177번째 줄 수정
label_map_path to: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
181번째 줄 수정
\images\test 폴더안에 이미지 숫자로 변경 폴더안에 ex)이미지 20장있으면 20
191번째 줄 수정
label_map_path to: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

２-２-６학습하기
Python path 지정 *cmd창 종료 후 재시작시 다시 지정 
set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
object_detection폴더위치에서 
python train.py --logtostderr –train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_quantized_300x300_coco.config실행
 
학습시작.
loss율 1.4~2.1정도로 계속 유지할 때까지 진행

２-２-７ 동결그래프 pb화하기
cmd 재시작시 python path를 재지정 해주어야한다.
mkdir TFLite_model  폴더만들기
set CONFIG_FILE=C:\\tensorflow1\models\research\object_detection\training\ssd_mobilenet_v2_quantized_300x300_coco.config
set CHECKPOINT_PATH=C:\\tensorflow1\models\research\object_detection\training\model.ckpt-XXXX
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model 

#학습한 수만큼 XXXX 이름 수정할 것
설정이 완료되었으면 다음 명령을 실행
 

python export_tflite_ssd_graph.py --pipeline_config_path=%CONFIG_FILE% --trained_checkpoint_prefix=%CHECKPOINT_PATH% --output_directory=%OUTPUT_DIR% --add_postprocessing_op=true
실행 된 후 \ object_detection \ TFLite_model 폴더에 tflite_graph.pb 및 tflite_graph.pbtxt라는 두 개의 새 파일이 있어야함

 

２-３ [2 – 1] 과정에서 만들어진 pb파일을 라즈베리파이에서 이용할 수 있는 tflite파일로 변환
[가이드 git hub : https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#part-1---how-to-train-convert-and-run-custom-tensorflow-lite-object-detection-models-on-windows-10]
step2부터 진행
２-３-１Install MSYS2
https://www.msys2.org/
pacman -Syu

pacman -Su
pacman -S patch unzip

 
２-３-２ Install Visual C++ Build Tools 2015(Visual Studio communite 2017 사용)
Visual studio 지난 버전 : https://visualstudio.microsoft.com/ko/vs/older-downloads/?rr=https%3A%2F%2Fwww.google.com%2F
２-３-３ Update Anaconda and create tensorflow-build environment
conda update -n base -c defaults conda
conda update --all

conda create -n tensorflow-build pip python=3.6
conda activate tensorflow-build

python -m pip install --upgrade pip

conda install -c anaconda git

２-３-４ Download Bazel and Python package dependencies
pip install six numpy wheel
pip install keras_applications==1.0.6 --no-deps
pip install keras_preprocessing==1.0.5 --no-deps

conda install -c conda-forge bazel=0.21.0
(visual studio communite 2017d에서는 0.24.1)

２-３-５ Download TensorFlow source and configure build
mkdir C:\tensorflow-build
cd C:\tensorflow-build

git clone https://github.com/tensorflow/tensorflow.git 
cd tensorflow 

git checkout r1.13
(visual studio communite 2017에서는 r1.14)
python ./configure.py

You have bazel 0.21.0- (@non-git) installed. 

Please specify the location of python. [Default is C:\ProgramData\Anaconda3\envs\tensorflow-build\python.exe]: 
  
Found possible Python library paths: 

  C:\ProgramData\Anaconda3\envs\tensorflow-build\lib\site-packages 

Please input the desired Python library path to use.  Default is [C:\ProgramData\Anaconda3\envs\tensorflow-build\lib\site-packages] 

Do you wish to build TensorFlow with XLA JIT support? [y/N]: N 
No XLA JIT support will be enabled for TensorFlow. 

Do you wish to build TensorFlow with ROCm support? [y/N]: N 
No ROCm support will be enabled for TensorFlow. 
  
Do you wish to build TensorFlow with CUDA support? [y/N]: N 
No CUDA support will be enabled for TensorFlow. 
대문자 알파벳 입력

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 

bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg 

activate tensorflow-build
cd C:\tensorflow-build
set OUTPUT_DIR=C:\\tensorflow1\models\research\object_detection\TFLite_model
pb파일 위치를 지정
 
bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=%OUTPUT_DIR%/tflite_graph.pb --output_file=%OUTPUT_DIR%/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops 

  

  
２-４ GOOGLE CORAL 사용을 위한 edgetpu.tflite 생성
* Edge TPU 컴파일러는 최신 데비안 기반 Linux 시스템에서 실행할 수 있음 RASPBERRY PI에서 불가능
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt-get update

sudo apt-get install edgetpu-compiler

 
 
