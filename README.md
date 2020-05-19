# MU-GAN: Facial Attribute Editing based on Multi-attention Mechanism
Official implementation of MU-GAN

# Requirements
*Python 3

*PyTorch 0.4.0

# Useage
Train:
```
sudo CUDA_VISIBLE_DEVICES=x python3 train.py --img_size 128 --experiment_name 128_shortcut1_inject1_none --gpu --num_workers x --batch_size x
```

Test:
```
sudo CUDA_VISIBLE_DEVICES=x python3 test_x_x.py --experiment_name 128_shortcut1_inject1_none --test_int 1.0 --gpu --load_epoch x
```

# Todo
Currently, the different variants of the code are in a mess. We will reorganize the code when we return to school.

# Acknowledgements
This code refers to the projects:[AttGAN-PyTorch](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/)
Thanks for their excellent work！
