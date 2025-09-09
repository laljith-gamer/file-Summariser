# pagewise_summary_from_txt_fixed.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import re

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "google/long-t5-tglobal-base"
    print(f"Loading summarization model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    input_file = "extracted_text_clean.txt" 
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        print("Error: Input file is empty!")
        return

    print(f"Loaded input text file ({len(content)} characters)")

    matches = list(re.finditer(r"(page[_ ]?\d+:?)", content, flags=re.IGNORECASE))
    if not matches:
        print("Error: No page markers found!")
        return

    summaries = []
    prev_summary = None

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(content)

        header = match.group().strip()
        page_text = content[start:end].strip()

        print(f"\nProcessing {header} ...")

        if not page_text:
            summaries.append(f"{header} Summary:\n[No text found]\n")
            continue

        try:
            prompt = f"summarize: {page_text}"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=400,
                    min_length=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # --- Detect continuation ---
            if prev_summary and (
                not page_text[0].isupper() or "continued" in page_text.lower()
            ):
                summary = "(Continued) " + summary

            summaries.append(f"{header} Summary:\n{summary}\n")
            prev_summary = summary
            print(f" {header} summarized.")

        except Exception as e:
            print(f" Error summarizing {header}: {e}")
            summaries.append(f"{header} Summary:\n[Error in summarization]\n")

        if device == "cuda":
            torch.cuda.empty_cache()

    output_file = "page_summaries.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))

    print(f"\n All page summaries saved to '{output_file}'")

if __name__ == "__main__":
    main()
