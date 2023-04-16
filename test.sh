# mini-imagenet
python test.py --dataset mini_imagenet --data_path /home/jiangweihao/data/mini-imagenet --n_shot 1 \
--pth_path /home/jiangweihao/code/svrg_fsl/save/mini_imagenet_9_5_resnet12_512_svrg_freq10/best_model.pth

# tiered_imagenet
python test.py --dataset tiered_imagenet --data_path /home/jiangweihao/data/tiered_imagenet --n_shot 5 \
--pth_path

# FC100
python test.py --dataset FC100 --data_path /home/jiangweihao/data/FC100 --n_shot 1 \
--pth_path 

# CUB
python test.py --dataset cub --data_path /home/jiangweihao/data/CUB_200_2011 --n_shot 5 \
--pth_path 