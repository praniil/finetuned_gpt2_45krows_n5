from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

model_path = "Pranilllllll/finetuned_gpt2_45krows_10epochs"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    user_input = input("You: ") 
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)

    attention_mask = inputs['attention_mask']

    response = text_generator(
        user_input,
        max_length=200,  
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        truncation=False,
    )

    suggestion = response[0]['generated_text']
    suggestion = suggestion[len(user_input):].strip()  

    # for i, char in enumerate(suggestion):
    #     if char == ".":
    #         final_suggestion = suggestion[i + 1:].strip()
    #         break

    print(f"Bot: {suggestion}")

    reference_text = [
        "I'm feeling really anxious lately and I don't know why.", 
        "It's common to feel anxious at times, and there can be many reasons for it. Have there been any recent changes or stressors in your life that may be contributing to your anxiety? Let's work together to identify any triggers and develop coping strategies to manage your anxiety."
    ]

    generated_tokens = tokenizer.tokenize(f"{suggestion}")
    reference_tokens = [tokenizer.tokenize(ref) for ref in reference_text]

    weights_1gram = (1, 0, 0, 0)  # Focus on individual word accuracy
    weights_2gram = (0.5, 0.5, 0, 0)  # Consider both individual words and bigrams
    weights_3gram = (0.33, 0.33, 0.33, 0)  # Trigrams included

    bleu_1gram = sentence_bleu(reference_tokens, generated_tokens, weights=weights_1gram)
    bleu_2gram = sentence_bleu(reference_tokens, generated_tokens, weights=weights_2gram)
    bleu_3gram = sentence_bleu(reference_tokens, generated_tokens, weights=weights_3gram)

    print(f"BLEU-1: {bleu_1gram}")
    print(f"BLEU-2: {bleu_2gram}")
    print(f"BLEU-3: {bleu_3gram}")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text[1], suggestion)  

    print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure}")
    print(f"ROUGE-2: {rouge_scores['rouge2'].fmeasure}")
    print(f"ROUGE-L: {rouge_scores['rougeL'].fmeasure}")




