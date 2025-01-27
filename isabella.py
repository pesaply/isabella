# Step 1: Import necessary libraries
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Step 2: Load a multilingual model (e.g., mT5 for Swahili)
model_name = "google/mt5-small"  # You can also use "facebook/xlm-roberta-base" for other tasks
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: Define the AI assistant function
def isabella_assistant(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a response using the model
    outputs = model.generate(
        inputs["input_ids"],
        max_length=200,  # Adjust the response length
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Step 4: Run the AI assistant
print("Isabella AI Assistant: Habari! Mimi ni Isabella, msaidizi wako wa AI. Ninaweza kukusaidia vipi leo?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Isabella AI Assistant: Kwaheri! Kuwa na siku njema!")
        break
    response = isabella_assistant(user_input)
    print(f"Isabella AI Assistant: {response}")
