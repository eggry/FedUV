# FedUV
## Run
```
python main.py --dataset_root /data2/datasets/ --verbose --cuda --device 1 --normalize --comment withnormalize127 

python main.py --dataset_root /data2/datasets/ --verbose --cuda --device 0 --comment wonormalize127

python main.py --dataset_root /data2/datasets/ --verbose --cuda --device 2 --normalize --code_length 511 --message_length 67 --d_min 175 --comment withnormalize511

python main.py --dataset_root /data2/datasets/ --verbose --cuda --device 1 --code_length 511 --message_length 67 --d_min 175 --comment wonormalize511
```

## Result 
The model overfits. The metrics are shown in [this Tensorboard](https://wandb.ai/eggry/FedUV/runs/44pg523e/tensorboard).
