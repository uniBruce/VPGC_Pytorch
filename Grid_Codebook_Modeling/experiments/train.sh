# Currently, we do not consider 1)f8 and 2)translation equivalance setting.

jobname=$1
export http_proxy='http://agent.baidu.com:8188'
export https_proxy='http://agent.baidu.com:8188'
# export NCCL_IB_DISABLE=1


# if [[ $jobname == 'lrw_base' ]]; then
#     export MASTER_PORT=29701
#     cmd="python main.py \
#             --base configs/"$jobname"_pc.yaml \
#             -t True --gpus 0,1 "
#     echo $cmd
#     $cmd
# fi


if [[ $jobname == 'lrw_base_8192' ]]; then
    export MASTER_PORT=29711
    resume_path=RESUME/logs/base_8192/
    cmd="python main.py \
            --base configs/"$jobname"_pc.yaml \
            -t True --gpus 0, \
            --resume ${resume_path} "
    echo $cmd
    $cmd
fi


if [[ $jobname == 'lrw_base_8192_local' ]]; then
    export MASTER_PORT=29702
    resume_path=RESUME/logs/base_8192/
    cmd="python main.py \
            --base configs/lrw_base_8192.yaml \
            -t True --gpus 0, \
            --resume ${resume_path} "
    echo $cmd
    $cmd
fi


if [[ $jobname == 'student_base_8192_local' ]]; then
    export MASTER_PORT=29702
    cmd="python main.py \
            --base configs/student_base_8192.yaml \
            -t True --gpus 0,"
    echo $cmd
    $cmd
fi



if [[ $jobname == 'student_base_8192_pc' ]]; then
    export MASTER_PORT=29702
    cmd="python main.py \
            --base configs/student_base_8192_pc.yaml \
            -t True --gpus 0,"
    echo $cmd
    $cmd
fi



if [[ $jobname == 'student_base_8192_random_mask_local' ]]; then
    resume_path=logs/2022-02-12T13-23-55_student_base_8192_pc
    export MASTER_PORT=29791
    cmd="/opt/_internal/cpython-3.7.0/bin/python main.py \
            --name $jobname \
            --base configs/student_base_8192_random_mask.yaml \
            -t True --gpus 0, \
            --resume ${resume_path} 
        "
    echo $cmd  
    $cmd
fi


if [[ $jobname == 'vqpp_double_maskz' ]]; then
    resume_path=/root/taming-transformer/PRETRAIN/student_pretrain/epoch=3-step=269999.ckpt
    export MASTER_PORT=49994
    cmd="python main.py \
            --name $jobname \
            --base configs/vqpp_double.yaml \
            -t True --gpus 0, \
        "
    echo $cmd  
    $cmd
fi


if [[ $jobname == 'vqref' ]]; then
    # resume_path=/root/taming-transformer/logs/2022-02-22T13-44-28_vqpp_double/checkpoints/epoch=3-step=149999.ckpt
    export MASTER_PORT=49991
    cmd="python main.py \
            --name $jobname \
            --base configs/vqref.yaml \
            -t True --gpus 0, \
        "
    echo $cmd  
    $cmd
fi


if [[ $jobname == 'student_base_8192_random_mask_shift_local' ]]; then
    resume_path=logs/2022-02-12T13-23-55_student_base_8192_pc
    export MASTER_PORT=29792
    cmd="python main.py \
            --name $jobname \
            --base configs/student_base_8192_random_mask_shift.yaml \
            -t True --gpus 0, "
    echo $cmd
    $cmd
fi


if [[ $jobname == 'student_base_8192_random_mask_shift_pc' ]]; then
    resume_path=logs/2022-02-12T13-23-55_student_base_8192_pc
    export MASTER_PORT=29792
    cmd="python main.py \
            --base configs/student_base_8192_random_mask_shift_pc.yaml \
            -t True --gpus 0,
            --resume ${resume_path} "
    echo $cmd
    $cmd
fi


if [[ $jobname == 'student_base_8192_random_mask_pc' ]]; then
    resume_path=logs/2022-02-12T13-23-55_student_base_8192_pc
    export MASTER_PORT=29754
    cmd="python main.py \
            --base configs/student_base_8192_random_mask_pc.yaml \
            -t True --gpus 1,
            --resume ${resume_path} "
    echo $cmd
    $cmd
fi


# if [[ $jobname == 'lrw_base_1024' ]]; then
#     export MASTER_PORT=29746
#     resume_path=RESUME/logs/base_1024/
#     cmd="python main.py \
#             --base configs/"$jobname"_pc.yaml \
#             -t True --gpus 1, \
#             --resume ${resume_path} "
#     echo $cmd
#     $cmd
# fi


# if [[ $jobname == 'lrw_base_1024_local' ]]; then
#     export MASTER_PORT=29709
#     resume_path=RESUME/logs/base_1024/
#     cmd="python main.py \
#             --base configs/lrw_base_1024.yaml \
#             -t True --gpus 0,1 \
#             --resume ${resume_path} "
#     echo $cmd
#     $cmd
# fi



# if [[ $jobname == 'lrw_f8_8192' ]]; then
#     export MASTER_PORT=29702
#     cmd="python main.py \
#             --base configs/lrw_f8_8192_pc.yaml \
#             -t True --gpus 0,1 "
#     echo $cmd
#     $cmd
# fi

# if [[ $jobname == 'lrw_f8_8192_debug' ]]; then
#     export MASTER_PORT=29702
#     cmd="python main.py \
#             --base configs/lrw_f8_8192_debug.yaml \
#             -t True --gpus 0,1 "
#     echo $cmd
#     $cmd
# fi



# if [[ $jobname == 'lrw_f8_8192_gumbel_debug' ]]; then
#     export MASTER_PORT=29702
#     cmd="python main.py \
#             --base configs/lrw_f8_8192_gumbel_debug.yaml \
#             -t True --gpus 0,1 "
#   
#     echo $cmd
#     $cmd
# fi


# if [[ $jobname == 'lrw_base_8192_te' ]]; then
#     export MASTER_PORT=29703
#     cmd="python main.py \
#             --base configs/"$jobname"_pc.yaml \
#             -t True --gpus 2, "
#     echo $cmd
#     $cmd
# fi


# if [[ $jobname == 'lrw_te' ]]; then
#     export MASTER_PORT=29700
#     cmd="python main.py \
#             --base configs/"$jobname"_pc.yaml \
#             -t True --gpus 0 "
#     echo $cmd
#     $cmd
# fi
