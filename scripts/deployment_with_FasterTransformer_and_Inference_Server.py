

!git clone https://github.com/triton-inference-server/fastertransformer_backend
%cd fastertransformer_backend
!git checkout 571a1fce438409087f5d3889237541828cc24ba5
%cd ../


!mkdir triton-model-store
!cp -r ./fastertransformer_backend/all_models/gptj triton-model-store/
!ls triton-model-store/gptj/
!CUDA_VISIBLE_DEVICES=3  ./FasterTransformer/build/bin/gpt_gemm 1 1 128 16 256 16384 50256 1 2


import os
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

os.system("CUDA_VISIBLE_DEVICES=2,3 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=./triton-model-store/gptj/ &")


URL = "localhost:8000"
MODEl_GPTJ_FASTERTRANSFORMER = "ensemble" 
OUTPUT_LEN = 128
BATCH_SIZE = 1
BEAM_WIDTH = 1
TOP_K = 1
TOP_P = 0.0
start_id = 220
end_id = 50256

client = httpclient.InferenceServerClient("localhost:8000",concurrency=1,verbose=False)


def prepare_tensor(name, input):
    tensor = httpclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    tensor.set_data_from_numpy(input)
    return tensor


def prepare_inputs(input0):
    bad_words_list = np.array([[""]], dtype=object)
    stop_words_list = np.array([[""]], dtype=object)
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
    runtime_top_k = (TOP_K * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    runtime_top_p = TOP_P * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
    random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.int32)
    is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
    beam_width = (BEAM_WIDTH * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
    start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
    end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        
    inputs = [
        prepare_tensor("INPUT_0", input0_data),
        prepare_tensor("INPUT_1", output0_len),
        prepare_tensor("INPUT_2", bad_words_list),
        prepare_tensor("INPUT_3", stop_words_list),
        prepare_tensor("runtime_top_k", runtime_top_k),
        prepare_tensor("runtime_top_p", runtime_top_p),
        prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
        prepare_tensor("temperature", temperature),
        prepare_tensor("len_penalty", len_penalty),
        prepare_tensor("repetition_penalty", repetition_penalty),
        prepare_tensor("random_seed", random_seed),
        prepare_tensor("is_return_log_probs", is_return_log_probs),
        prepare_tensor("beam_width", beam_width),
        prepare_tensor("start_id", start_ids),
        prepare_tensor("end_id", end_ids)]
  
    return inputs

print("Write an input prompt for the model and execute the cell:")

input_user = input()
input0 = [[input_user],]
inputs = prepare_inputs(input0)


result = client.infer(MODEl_GPTJ_FASTERTRANSFORMER, inputs)
output0 = result.as_numpy("OUTPUT_0")
print(output0)


'''
Now you know how to run large models in production. Triton Inference Server allows you to access your model from any device in the world.

To conclude let us review the end-to-end pipeline summary again:

'''
