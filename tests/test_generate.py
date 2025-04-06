from transformers import AutoTokenizer

def test_generate_script():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with a valid model name
    input_prompt = "Test prompt"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    assert input_ids is not None
