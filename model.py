from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4-bit on GPU only — Falcon-7B in nf4 (~4 GB) fits comfortably on a T4 (16 GB),
# so we don't need CPU/disk offload. The previous `llm_int8_enable_fp32_cpu_offload=True`
# combined with `device_map="auto"` was leaving some weights on the meta device,
# which crashed inference with: "Tensor on device meta is not on the expected device cuda:0".
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},  # pin everything to GPU 0, no offload
    quantization_config=quantization_config,
)


def generate_answer(question, context):
    """Generates an answer using the most relevant PDF content."""
    input_text = f"السؤال: {question}\nالمحتوى: {context[:1024]}\nالإجابة:"  # Trim to 1024 tokens
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")

    output = model.generate(**inputs, max_new_tokens=256)  # Limit token generation
    return tokenizer.decode(output[0], skip_special_tokens=True)
