## FOTS_TF(端到端的文本识别-nba记分牌识别)
#这是计算机视觉大公司“商汤科技”发布的一篇OCR领域的代码，具体使用方法如下
### 1. custom训练数据
最终数据需要的形式是每个图片对应一个txt包含每一个bbox的（xyxyxyxy,gt）这样的label数据，
因此第一步首先把标注数据的csv转成一个一个的txt。
eg：即把nba_train_1023.csv转为training_gt_1080p_v1106（数据预处理）
``` python
python get_custom_sbb.py
``` 
### 2. train（模型训练）
``` python
#!/bin/sh
python /FOTS_TF/main_train.py \
--batch_size_per_gpu=16 \
--num_readers=6 \
--gpu_list='0' \
--restore=False \
--checkpoint_path='checkpoints/bs16_1080p_v1106_aughsv/' \
--pretrained_model_path='models/model.ckpt-733268' \
--training_data_dir='training_img_1080p_v1106' \
--training_gt_data_dir='training_gt_1080p_v1106'
``` 
其中，checkpoint_path为要保存的模型的路径；pretrained_model_path为加载icdar的预训练模型路径，请在以下百度网盘链接中下载。

### 3. infer（模型推理）
``` python
#!/bin/sh
python main_test_bktree.py \
--test_data_path='samples' \
--checkpoint_path='checkpoints/bs16_540p_v1106_aughsv/' \
--output_dir='outputs/outputs_bs16_540p_v1106_aughsv_2016' 
``` 

### 4. eval（模型评估）
在这一部分，我们用了后处理逻辑，然后用来评估test集的准确率，大家不需要这一部分，可以忽略。
``` python
#!/bin/sh
python /data/ceph_11015/ssd/anhan/nba/FOTS_TF/main_test_bktree_eval_v2.py \
--just_infer=False \
--check_teamname=False \
--test_data_path='/data/ceph_11015/ssd/templezhang/scoreboard/EAST/data/check_res_15161718_test_null.csv' \
--checkpoint_path='/data/ceph_11015/ssd/anhan/nba/FOTS_TF/checkpoints/bs16_540p_v1106_aughsv/' \
--output_dir='/data/ceph_11015/ssd/anhan/nba/FOTS_TF/outputs/outputs_bs16_540p_v1106_aughsv_eval' 
``` 
--checkpoint_path为模型的最终位置

pretrained-model地址：

链接：https://pan.baidu.com/s/1xyfZM_EPxFLOGEv9p3E9Vg 
提取码：pv8r
将下载好的model的放入文件夹的根目录,也可以修改config文件

nba_train_1023.csv训练数据地址：

链接：https://pan.baidu.com/s/13Jrc2gsnza4-gjnUlDcKeg 
提取码：4lf9
将下载好训练数据.csv的放入文件夹的根目录，也可以修改config文件地址