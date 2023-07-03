!git clone https://github.com/NVIDIA/FasterTransformer.git
%cd FasterTransformer
!git checkout 6b3fd4392831f972d48127e881a048567dd92811

!apt update
!DEBIAN_FRONTEND=noninteractive apt install -y cmake xz-utils zstd libz-dev


!mkdir -p build
%cd build 

!git submodule init && git submodule update
!cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON ..
!make -j32

print("List of FasterTransformer examples:")
!ls ./bin

%cd /dli/task

!tar -axf ./weights/gpt-j/ft/step_383500_slim.tar.zstd -C ./model_repository/

!ls ./model_repository/step_383500/

!wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
!wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models

%cd FasterTransformer/build


!python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
                                --output-dir ../../models/j6b_ckpt \
                                --ckpt-dir ../../model_repository/step_383500/ \
                                --n-inference-gpus 1
  
  
  
!ls ../../models/j6b_ckpt/

!CUDA_VISIBLE_DEVICES=1  ./bin/gpt_gemm 1 1 128 16 256 16384 50256 1 1

!CUDA_VISIBLE_DEVICES=1 ./bin/gptj_example

!python3 ../examples/pytorch/gpt/utils/gpt_token_converter.py \
                       --vocab_file=../../models/gpt2-vocab.json  \
                       --bpe_file=../../models/gpt2-merges.txt

!python3 ../examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
                                --output-dir ../../models/j6b_ckpt \
                                --ckpt-dir ../../model_repository/step_383500/ \
                                --n-inference-gpus 2

!ls ../../models/j6b_ckpt/

!CUDA_VISIBLE_DEVICES=1,2 mpirun -n 2 --allow-run-as-root ./bin/gptj_example

!CUDA_VISIBLE_DEVICES=1  ./bin/gpt_gemm 1 1 128 16 256 16384 50256 1 2

!CUDA_VISIBLE_DEVICES=1,2 mpirun -n 2 --allow-run-as-root ./bin/gptj_example

!python3 ../examples/pytorch/gpt/utils/gpt_token_converter.py \
                               --vocab_file=../../models/gpt2-vocab.json  \
                               --bpe_file=../../models/gpt2-merges.txt

