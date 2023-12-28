## Requirements

```shell
conda create -y --name flowformer python=3.8 && conda activate flowformer
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install matplotlib tensorboard scipy
pip install yacs loguru einops timm==0.4.12 imageio opencv-python
```


## Test

```shell
python evaluate_FlowFormer_inr.py --dataset sintel_inr --model logs/sintel_inr/record/sintel_inr.pth --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_ratio_1_steps_20000_dec/ --flow_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_flows/ --occlu_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_occlusions/
```

## Train

```shell
python -u train_FlowFormer_inr.py --name sintel_inr --stage sintel_inr --validation sintel_inr --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_ratio_1_steps_20000_dec/ --flow_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_resized_flows/
```

----

train & test with non-inr data (resize)

```shell
python -u train_FlowFormer_inr.py --name sintel_inr --stage sintel_inr --validation sintel_inr --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_resized_images/ --flow_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_train_resized_flows/

python evaluate_FlowFormer_inr.py --dataset sintel_inr --model checkpoints/final.pth --image_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_images/ --flow_root //ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_flows/ --occlu_root /ssd/0/yrz/Dataset/Sintel_INR/Sintel_custom_test_resized_occlusions/
```

