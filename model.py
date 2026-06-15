from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Enable 4-bit and Offload to CPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Moves part of the model to CPU
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto",  
    quantization_config=quantization_config
)


def generate_answer(question, context):
    """Generates an answer using the most relevant PDF content."""
    input_text = f"السؤال: {question}\nالمحتوى: {context[:1024]}\nالإجابة:"  # Trim to 1024 tokens
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    output = model.generate(**inputs, max_new_tokens=256)  # Limit token generation
    return tokenizer.decode(output[0], skip_special_tokens=True)
