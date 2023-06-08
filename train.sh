# # mini-imagenet
# python train_2.py --dataset mini_imagenet --data_path /home/jiangweihao/data/mini-imagenet --num_classes 64 --gpu 4 --batch_size 128 --lr 0.1

# # tiered_imagenet
# python train_2.py --dataset tiered_imagenet --data_path /home/jiangweihao/data/tiered_imagenet --num_classes 351 --gpu 4 --batch_size 256 --lr 0.1

# # FC100
# python train_2.py --dataset FC100 --data_path /home/jiangweihao/data/FC100 --num_classes 60 --gpu 6 --batch_size 128 --lr 0.1

# # cub
# python train_2.py --dataset cub --data_path /home/jiangweihao/data/CUB_200_2011 --num_classes 100 --gpu 5 --batch_size 128 --lr 0.1


python train_mae_image_decoder.py --dataset tiered_imagenet --epochs 90 

python train_mae_image_decoder.py --dataset fs --epochs 90  

python train_mae_image_decoder.py --dataset fc100 --epochs 90 