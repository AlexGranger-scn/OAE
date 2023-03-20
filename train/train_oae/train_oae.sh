dataset=cnc
result_path=training_ckpt/$dataset/oae
train_data_split=embedding/data_split/$dataset/train.csv
val_data_split=embedding/data_split/$dataset/val.csv
data_npy_path=embedding/$dataset/train/$dataset\_resnet_$dataset.npy
batchSize=1024
niter=400
lr=0.01
checkpoint=2
ngpu=0



for dimension in 256
    do

mkdir -p $result_path/$dimension/result
mkdir -p $result_path/$dimension/ckpt

python3 train_oae/main.py \
        --data_npy_path $data_npy_path \
        --train_data_split $train_data_split \
        --val_data_split $val_data_split \
        --outf $result_path/$dimension/ckpt \
        --log_path $result_path/$dimension/result \
        --batchSize $batchSize \
        --dimension $dimension \
        --niter $niter \
        --lr $lr \
        --checkpoint $checkpoint \
        --ngpu $ngpu \

    done