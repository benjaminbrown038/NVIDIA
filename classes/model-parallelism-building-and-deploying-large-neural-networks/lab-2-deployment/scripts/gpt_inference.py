import torch
from transformers import AutoTokenizer, AutoModelForCasualLM

model = AutoModelForCasualLM.from_pretrained("/weights/gpt-j/hf/")
tokenizer = AutoTokenizer.from_pretrained("./weights/gpt-j/hf/")
tokenizer = AutoTokenizer.from_pretrained("./weights/gpt-j/hf/")

assert torch.cuda.is_available()
device = torch.device("cuda:0")
model.half().to(device)
model = model.eval()


with torch.no_grad():
    output = model.generate(input_ids=None, max_length=128, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decoding the generated text
for sentence in output:
    sentence = sentence.tolist()
    text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
    print(text)


input_ids = tokenizer.encode("English: I do not speak French. French: Je ne parle pas français." \
                             "English: See you later! French: À tout à l'heure!" \
                             "English: Where is a good restaurant? French: Où est un bon restaurant?" \
                             "English: What rooms do you have available? French:", return_tensors="pt").cuda(0)

output = model.generate(input_ids=input_ids, max_length=82, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
sentence = output[0].tolist()
text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
print(text)

output = model.generate(input_ids=input_ids, max_length=80, num_return_sequences=1, num_beams=5, temperature=0.7, repetition_penalty=3.0, pad_token_id=tokenizer.eos_token_id)
sentence = output[0].tolist()
text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
print(text)

input_ids = tokenizer.encode("Create an SQL request to find all users that live in Califorian and have more than 1000 credits.", return_tensors="pt").cuda(0)
output = model.generate(input_ids=input_ids, max_length=82, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
output = model.generate(input_ids=input_ids, max_length=80, num_return_sequences=1, num_beams=5, temperature=0.7, repetition_penalty=3.0, pad_token_id=tokenizer.eos_token_id)
sentence = output[0].tolist()
text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
print(text)

# Generate the sentence.
import time

execution_time = 0
num_iterations = 10
with torch.no_grad():
    for _ in range(num_iterations):
        start = time.time()
        output = model.generate(input_ids=None, max_length=128, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=50256)
        end = time.time()
        execution_time += end - start

print("Average inference time of 128 tokens is:",
     1000 * (execution_time/float(num_iterations)), "ms")


'''

128 tokens can be generated in 6.3 seconds. 
Let us move to the next notebook and test an optimized inference pipeline.

'''










