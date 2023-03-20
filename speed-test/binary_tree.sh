data_set=vox
enroll_path=embedding/$data_set/test/enroll_resnet_$data_set.npy
test_path=embedding/$data_set/test/test_resnet_$data_set.npy
enroll_matrix=embedding/$data_set/test/generate/enroll.npy
test_matrix=embedding/$data_set/test/generate/test.npy
dir_path=speed-test/test_result
ngpu=0
binary=256
mean_depth=0
stage=1
for cut_dimension in 32
    do 
    pth_path_binary=speed-test/maes-$data_set-binary-256.pth
    pth_path_dense=speed-test/mae-$data_set-dense-256.pth
    result_path=$dir_path/$data_set/$cut_dimension
    #test_result_path=MAE/training_ckpt/$data_set/$mae_mode/$cfg_mode/$binary/result

    echo "binary: $binary"
    echo "data_set: $data_set"
    echo "pth_path: $pth_path_binary"

    mkdir -p $result_path

python3  speed-test/tree.py\
    --ngpu $ngpu \
    --enroll_path $enroll_path\
    --cut_dimension $cut_dimension\
    --test_path $test_path \
    --enroll_matrix $enroll_matrix\
    --mean_depth $mean_depth\
    --output_sr_path $result_path/top10-meta\
    --csv_path $result_path/score.csv\
    --path_top $result_path/top10\
    --test_matrix $test_matrix \
    --fig_path $test_result_path/tree.png \
    --binary $binary \
    --dataset $data_set\
    --pth_path_binary $pth_path_binary\
    --pth_path_dense $pth_path_dense\
    --stage $stage
done