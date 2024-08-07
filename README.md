This is the repo for the code of our NeurIPS submission 12474.

### An example script to run the code:

`
CUDA_VISIBLE_DEVICES=0 python main_base.py  --dataset mnistandusps --ratio 0 --alpha 4 --seed 40 --lossmode lc1`


### Some important files and folders:
- data_loader.py: Data preprossing and loading
- main_base.py: The main implementation of our method
- /baseline: implementation of baselines

### Requirements
```
python==3.8
aif360==0.6.1
h5py==2.10.0
torch==1.12.1
torchvision==0.13.1
numpy==1.20.1
pandas==1.2.4
scikit-learn==1.3.2
scikit-image==0.18.1
pillow==8.2.0
```
