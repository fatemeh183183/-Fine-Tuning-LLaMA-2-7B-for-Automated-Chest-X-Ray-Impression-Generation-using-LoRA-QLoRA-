import streamlit as st
import torch
from transformers.models.auto import AutoTokenizer
from transformers.models.auto import AutoModelForCausalLM

from peft import PeftModel

from huggingface_hub import login
import os

# Authenticate to access gated LLaMA-2 model
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("⚠️ Warning: No HF_TOKEN found — gated models may fail to load")


st.title("🩻 Radiology Impression Generator (Fine-tuned LLaMA-2)")

# ---- Load model ----
@st.cache_resource
def load_model():
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Base LLaMA-2 model
    ADAPTER_PATH = "fateme-sh/radiology-llama-merged"  #  LoRA fine-tuned model

    # Load the base model
    BASE_MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Apply your fine-tuned adapter
    MODEL = PeftModel.from_pretrained(BASE_MODEL, ADAPTER_PATH)

    # Load tokenizer from adapter repo (it has your special tokens)
    TOKENIZER = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
    TOKENIZER.pad_token = TOKENIZER.eos_token

    return MODEL, TOKENIZER

model, tokenizer = load_model()

# ---- Inference function ----
def generate_impression(prompt, max_new_tokens=150, temperature=0.7):
    formatted_prompt = f"[INST] {prompt.strip()} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
  # Decode and clean result
         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = result.replace("[INST]", "").replace("[/INST]", "").strip()
        return result

# ---- Streamlit interface ----
user_input = st.text_area("📝 Enter radiology findings:")
if st.button("🧠 Generate Impression"):
    with st.spinner("Generating report..."):
        impression = generate_impression(user_input)
    st.subheader("🧠 Impression:")
    st.write(impression)