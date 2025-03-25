import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel

# Load model and tokenizer
model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True
)

# Initialize model for inference
model = FastLanguageModel.for_inference(model)

def chat(user_input):
    messages = [{"role": "user", "content": user_input}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    input_ids = inputs.to("cuda") if torch.cuda.is_available() else inputs
    
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=80,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract assistant's response after the last assistant token
    if "assistant" in decoded_output.lower():
        response = decoded_output.split("assistant", 1)[-1].strip()
        response = response.split("user", 1)[0].strip()
    else:
        response = decoded_output
    
    return response

# Define Gradio interface
demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
    outputs=gr.Textbox(label="Assistant Response"),
    title="AI Assistant",
    description="Chat with an AI assistant."
)

demo.launch()