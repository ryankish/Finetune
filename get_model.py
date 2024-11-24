from transformers import AutoTokenizer, AutoModelForCausalLM

proj_dir = "/data2/ryan/finetune"

# Load tokenizer and model into the custom directory
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")