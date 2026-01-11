https://colab.research.google.com/drive/1IMCBA-VrIDWf-7IJsr9IPjzPA1yqUuCR
🩺 Fine-Tuning LLaMA-2 (7B) for Automated Chest X-Ray Impression Generation

Using LoRA / QLoRA

📌 Overview

This project presents a large language model fine-tuned to automatically generate the IMPRESSION section of chest X-ray radiology reports from FINDINGS text.

The model is based on Meta’s LLaMA-2 (7B) and fine-tuned using LoRA / QLoRA, enabling efficient training on limited GPU resources.
The goal is to support radiologists, clinicians, and researchers by reducing reporting time while maintaining clinical consistency.

🧠 Model & Methodology

Base model: LLaMA-2-7B

Fine-tuning method: LoRA / QLoRA

Task: Findings → Impression generation

Domain: Chest X-ray radiology reports

Training format: Instruction-style supervised fine-tuning

🗂 Repository Structure
.
├── radiology fine tune/
│   ├── Final_Tune_radiology1_llama2_finetuning.ipynb
│   ├── train.jsonl
│   ├── validate.jsonl
│   ├── test.jsonl
│
├── Radiology_llama_merged/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── app.py                # Streamlit inference app
├── requirements.txt
├── README.md

▶️ Google Colab (Training Notebook)

You can view and run the full fine-tuning pipeline in Google Colab here:

👉 Colab Notebook:
https://colab.research.google.com/drive/1IMCBA-VrIDWf-7IJsr9IPjzPA1yqUuCR

(The notebook may not render correctly on GitHub due to widget metadata. Please open it directly in Colab.)

🧪 Dataset

The dataset consists of paired FINDINGS → IMPRESSION radiology text samples and is stored in JSONL format:

train.jsonl

validate.jsonl

test.jsonl

Each entry contains structured clinical text used for supervised fine-tuning.

🖥 Streamlit Application

A Streamlit app is provided to test the fine-tuned model interactively.

To run locally:

pip install -r requirements.txt
streamlit run app.py


The app allows users to paste radiology FINDINGS and receive a concise IMPRESSION.

⚠️ Notes

Model weights are not included due to size and licensing constraints.

The Jupyter notebook may show “Invalid Notebook” on GitHub preview — this is a known GitHub limitation and does not affect functionality.

Open the notebook in Google Colab or Jupyter Lab for full access.

📜 License & Acknowledgements

LLaMA-2 is subject to Meta’s license terms.

This project is intended for research and educational purposes only.
