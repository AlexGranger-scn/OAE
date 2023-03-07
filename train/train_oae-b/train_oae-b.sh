dataset=cnc
result_path=training_ckpt/$dataset/oae-b
train_data_split=embedding/data_split/$dataset/train.csv
val_data_split=embedding/data_split/$dataset/val.csv
data_npy_path=embedding/$dataset/train/$dataset\_resnet_$dataset.npy
batchSize=1024
niter=400
lr=0.01
checkpoint=2
ngpu=0
# 


for binary_bits in 256
    do

mkdir -p $result_path/$binary_bits/result
mkdir -p $result_path/$binary_bits/ckpt

python3 train/train_oae-b/main.py \
        --checkpoint $checkpoint \
        --ngpu $ngpu \
        --data_npy_path $data_npy_path \
        --train_data_split $train_data_split \
        --val_data_split $val_data_split \
        --batchSize $batchSize \
        --binary_bits $binary_bits \
        --niter $niter \
        --lr $lr \
        --outf $result_path/$binary_bits/ckpt \
        --log_path $result_path/$binary_bits/result \

    done