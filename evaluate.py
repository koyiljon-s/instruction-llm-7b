from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

# load model
model_path = "mistral-uzbek-lora"  # yoki boshqa o'z model yo'lingiz
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

# give instruction
instruction = "Qahva do'koni ochish uchun marketing strategiyasini tavsiya qiling."


inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, max_new_tokens=300, streamer=streamer)