import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或者 "true"
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
name = '/vol2/matmulfreellm/hgrn_bit_1.3B_100B_realbit'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda().half()
input_prompt = "I am "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=32,  do_sample=True, top_p=0.4, temperature=0.6)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])