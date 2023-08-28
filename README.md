# Efficient Video Portrait Reenactment via Grid-based Codebook
by [Kaisiyuan Wang](https://unibruce.github.io/), [Hang Zhou](https://hangz-nju-cuhk.github.io/), [Qianyi Wu](https://wuqianyi.top/), [Jiaxiang Tang](https://me.kiui.moe/), Zhiliang Xu, Borong Liang, Tianshu Hu, Errui Ding, Jingtuo Liu, [Ziwei Liu](https://liuziwei7.github.io/), [Jingdong Wang](https://jingdongwang2017.github.io/).
## Introduction
This repository is a PyTorch implementation of our Siggraph 2023 paper [Efficient Video Portrait Reenactment via Grid-based Codebook](https://wywu.github.io/projects/VPGC_website).

![image](https://github.com/uniBruce/Mead/blob/master/Figures/mead.png)

## Installation 
This repository is based on Pytorch, so please follow the official instructions [here](https://pytorch.org/). The code is tested under pytorch1.7 and Python 3.6 on Ubuntu 16.04.  

## Usage
### Training set & Testing set Split
Please refer to the Section 6 "Speech Corpus of Mead" in the supplementary material. The speech corpora are basically divided into 3 parts, (i.e., common, generic, and emotion-related). For each intensity level, we directly use the last 10 sentences of neutral category and the last 6 sentences of the other seven emotion categories as the testing set. Note that all the sentences in the testing set come from the "emotion-related" part. Meanwhile if you are trying to manipulate the emotion category, you can use all the 40 sentences of neutral category as the input samples.
### Training
1. Download the dataset from [here](https://wywu.github.io/projects/MEAD/MEAD.html). We package the audio-visual data of each actor in a single folder named after "MXXX" or "WXXX", where "M" and "W" indicate actor and actress, respectively.
2. As Mead requires different modules to achieve different functions, thus we seperate the training for Mead into three stages. In each stage, the corresponding configuration (.yaml file) should be set up accordingly, and used as below:
#### Stage 1: Audio-to-Landmarks Module
```
cd Audio2Landmark
python train.py --config config.yaml
```
#### Stage 2: Neutral-to-Emotion Transformer
```
cd Neutral2Emotion
python train.py --config config.yaml
```
#### Stage 3: Refinement Network
```
cd Refinement
python train.py --config config.yaml
```
### Testing
1. First, download the [pretrained models](https://drive.google.com/drive/folders/1NgW0pqKU-jawqSi-RXiebUcI1_qj6wxM?usp=sharing) and put them in models folder.
2. Second, download the [demo audio data](https://drive.google.com/file/d/1G0sclW7AHqofyQAZFf6DqH4sTYSR85S9/view?usp=sharing).
3. Run the following command to generate a talking sequence with a specific emotion
```
cd Refinement
python demo.py --config config_demo.yaml
```
You can try different emotions by replacing the number with other integers from 0~7.
- 0:angry
- 1:disgust
- 2:contempt
- 3:fear
- 4:happy
- 5:sad
- 6:surprised
- 7:neutral

In addition, you can also try compound emotion by setting up two different emotions at the same time.

![image](https://github.com/uniBruce/Mead/blob/master/Figures/compound_emotion.png)

3. The results are stored in outputs folder. 

## Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{kaisiyuan2020mead,
 author = {Wang, Kaisiyuan and Wu, Qianyi and Song, Linsen and Yang, Zhuoqian and Wu, Wayne and Qian, Chen and He, Ran and Qiao, Yu and Loy, Chen Change},
 title = {MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation},
 booktitle = {ECCV},
 month = Augest,
 year = {2020}
} 
```
