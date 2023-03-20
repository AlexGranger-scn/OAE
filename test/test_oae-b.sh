data_set=cnc
enroll_path=embedding/$data_set/test/enroll_resnet_$data_set.npy
test_path=embedding/$data_set/test/test_resnet_$data_set.npy
enroll_matrix=embedding/$data_set/test/generate/enroll.npy
test_matrix=embedding/$data_set/test/generate/test.npy
ngpu=0
type=test
binary=2000

for cut_dimension in 128
    do  

    pth_path=training_ckpt/$dataset/oae-b/$binary_bits/ckpt/398.pth
    test_result_path=training_ckpt/$dataset/oae-b/$binary_bits/$cut_dimension/result

    echo "binary: $binary"
    echo "data_set: $data_set"
    echo "pth_path: $pth_path"

    mkdir -p $test_result_path
    mkdir -p $test_result_path/score

python3  test/test_oae-b.py\
    --ngpu $ngpu \
    --score_csv $test_result_path/score/scores.csv \
    --path_top $test_result_path/score/topNs \
    --output_sr_path $test_result_path/score/score-top10-metas \
    --enroll_path $enroll_path\
    --test_path $test_path \
    --enroll_matrix $enroll_matrix\
    --test_matrix $test_matrix \
    --log_path $test_result_path \
    --binary $binary \
    --cut_dimension $cut_dimension\
    --data_set $data_set\_$type \
    --pth_path $pth_path 
done