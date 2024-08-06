This is the repo for the code of our NeurIPS submission 12474.

### An example script to run the code:

`
CUDA_VISIBLE_DEVICES=0 python main_base.py  --dataset mnistandusps --ratio 0 --alpha 4 --seed 40 --lossmode lc1`


### Some key files:
- data_loader.py: Data preprossing and loading
- main_base.py: The main implementation of our method