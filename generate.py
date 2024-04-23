import fla
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'ridger/rwkv_r_340M_15B'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda()
input_prompt = "Hello World!"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])