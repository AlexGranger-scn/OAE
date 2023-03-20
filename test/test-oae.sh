data_set=cnc
enroll_path=/home/wangjiaying/embedding/$data_set/test/enroll_resnet_$data_set.npy
test_path=/home/wangjiaying/embedding/$data_set/test/test_resnet_$data_set.npy
enroll_matrix=/home/wangjiaying/embedding/$data_set/test/generate/enroll.npy
test_matrix=/home/wangjiaying/embedding/$data_set/test/generate/test.npy
ngpu=0
binary=256

for ((i=0; i<=0;i=i+8))
    do 
    cut_dimension=$i
    pth_path=/home/wangjiaying/MAE/training_ckpt/vox/MAE_S/1Layer-no-bias/398.pth
    test_result_path=training_ckpt/$dataset/oae-b/$binary_bits/$cut_dimension/result

    echo "binary: $binary"
    echo "data_set: $data_set"
    echo "pth_path: $pth_path"

    mkdir -p $test_result_path
    mkdir -p $test_result_path/score

python3  test/test_oae.py\
    --ngpu $ngpu \
    --score_csv $test_result_path/score/scores.csv \
    --path_top $test_result_path/score/topNs \
    --output_sr_path $test_result_path/score/score-top10-metas \
    --enroll_path $enroll_path\
    --cut_dimension $cut_dimension\
    --test_path $test_path \
    --enroll_matrix $enroll_matrix\
    --test_matrix $test_matrix \
    --log_path $test_result_path \
    --binary $binary \
    --data_set $data_set \
    --pth_path $pth_path 
done