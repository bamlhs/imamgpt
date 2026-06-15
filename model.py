from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Qwen2.5-7B-Instruct: strong multilingual (including Arabic), proper chat template,
# ungated on the HF Hub. nf4-quantized weights are ~4 GB, comparable to Falcon-7B,
# so VRAM headroom on a T4 stays the same as before.
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=quantization_config,
)


SYSTEM_PROMPT = (
    "أنت مساعد إسلامي سُنّي. اعتمد في إجابتك فقط على المحتوى المرجعي المُعطى لك "
    "من كتب أهل السنة (الرحيق المختوم، كتاب التوحيد، الأربعون النووية، "
    "رياض الصالحين، فقه السنة). أجب بالعربية الفصحى، بإيجاز ووضوح، ودون اختلاق. "
    "إن لم يتضمن النص المرجعي ما يكفي للإجابة، فقُل صراحةً: "
    "(المعلومات المتوفرة في المراجع لا تكفي للإجابة على هذا السؤال)."
)


def generate_answer(question, context):
    """Generates an answer grounded in the retrieved Sunni reference content."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"السؤال: {question}\n\n"
                f"المحتوى المرجعي:\n{context}\n\n"
                "استخرج إجابتك من المحتوى أعلاه فقط."
            ),
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to("cuda")
    input_len = inputs.input_ids.shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens — drop the prompt echo.
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
