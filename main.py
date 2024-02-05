from transformers import AutoTokenizer, AutoModelForCausalLM


# Download and load the model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Generate text
#prompt = "give me information about fishes "#+input("venta:")+" "
prompt = "Write a poem about either love, nature, or dreams."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids,do_sample=True, top_p=0.95, temperature=0.7)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
