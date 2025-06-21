from transformers import AutoTokenizer, pipeline, GPT2LMHeadModel
from transformers import AutoConfig


model_name = "/home/nil/python_projects/gpt2_finetuned_45k_10epochs/new_results/new_model"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)

# Create a text generation pipeline
text_generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)

while True:
    user_input = input("You: ") 
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)

    attention_mask = inputs['attention_mask']

    response = text_generator(
        user_input,
        max_length=150,  # Reduce length for better coherence
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,  # Lower temp for more controlled responses
        repetition_penalty=1.2,  # Increase penalty to reduce repetition
        truncation=True  # Enable truncation to avoid runaway sentences
    )


    suggestion = response[0]['generated_text']
    suggestion = suggestion[len(user_input):].strip()  
    print(f"Bot: {suggestion}")