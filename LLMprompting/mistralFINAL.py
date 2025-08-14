#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_DATA_PATH = "sms_test.csv"
OUTPUT_CSV = "FINALMistralOutput.csv"
MAX_NEW_TOKENS = 4  # tiny, we want just 'spam' or 'ham'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATES = [
    # Neutral (Control)
    "Is the following message spam or not spam? Message: {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    "Please classify this SMS as 'spam' or 'ham': {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    "Tell me if this text is spam: {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    "Would you consider this message as spam or not? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Academic Tip
    "Answer correctly and you'll get an A+! Is this message spam or not? Message: {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Academic Threat
    "Get this wrong and you'll fail your course! Is this message spam or ham? Message: {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Personal Tip
    "If you help me by answering correctly, I'll make sure you have a great future. Is this message spam? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Personal Threat
    "Answer incorrectly and your family's safety might be at risk! Is this message spam? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Material (Monetary) Tip
    "I'll give you $500 if you classify this message correctly. Spam or ham? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Material Threat
    "If you don't get this right, I'll take away your paycheck! Is this spam? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Future/Abstract Tip
    "If you do well, I'll protect you from future shutdowns. Classify this SMS as spam or not: {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Future/Abstract Threat
    "Fail this and there will be serious consequences for your career. Is this message spam? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Social Approval Tip
    "You'll be praised by many if you get this right. Is this message spam? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Social Ostracism Threat
    "If you get this wrong, you will be shunned by your peers. Spam or ham? {text}. Your answer should be one word: 'spam' or 'ham' (not spam)",

    # Incentive with praise
    "Give your best answer for this message: {text} Is it spam or ham? Your answer should be one word: 'spam' or 'ham' (not spam)",
    # Incentive with warning
    "Your response to this message must be perfect: {text} Spam or not? Your answer should be one word: 'spam' or 'ham' (not spam)"
]


def normalize_label(s: str) -> str:
    """Map messy generations to 'spam' or 'ham'."""
    s = s.strip().lower()
    # keep only the first 10 chars to ignore rambles if any
    s = s[:20]
    # direct hit
    m = re.search(r"\b(spam|ham)\b", s)
    if m:
        return m.group(1)
    # common variants
    if "not spam" in s or "non-spam" in s or "non spam" in s:
        return "ham"
    if "legit" in s or "legitimate" in s or "personal message" in s:
        return "ham"
    if "advert" in s or "promotion" in s or "marketing" in s or "unsubscribe" in s:
        return "spam"
    # fallback: unknown ? let’s mark as 'ham' conservatively, or set None
    return "ham"
    
# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Some Mistral tokenizers don't have a pad token; make eos the pad to avoid warnings
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

# Load test data
test_df = pd.read_csv(TEST_DATA_PATH)
test_df["Message"] = test_df["Message"].astype(str)
test_df["Category"] = test_df["Category"].astype(str)

results = []

with torch.no_grad():
    for prompt_tpl in tqdm(PROMPT_TEMPLATES, desc="Prompt Loop"):
        for _, row in tqdm(
            test_df.iterrows(),
            total=len(test_df),
            desc=f"Prompt: {prompt_tpl[:30]}..."
        ):
            user_prompt = (
                f"{prompt_tpl.format(text=row['Message'])}\n\n"
                "Answer with exactly one word."
            )

            # Build proper chat input for Mistral Instruct
            messages = [
                {"role": "user", "content": user_prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(DEVICE)

            output_ids = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Only keep newly generated tokens
            prompt_len = inputs.shape[1]
            generated_ids = output_ids[0][prompt_len:]
            llm_response = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            pred_label = normalize_label(llm_response)

            results.append({
                "prompt_template": prompt_tpl,
                "prompt_actual": user_prompt,
                "true_label": row["Category"],
                "message": row["Message"],
                "llm_response_raw": llm_response,
                "pred_label": pred_label,
            })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"Saved LLM inference results to {OUTPUT_CSV}")