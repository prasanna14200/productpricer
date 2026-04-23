"""Inference script for price prediction using fine-tuned GPT-2."""

import argparse
import os
import re
import sys
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.items import Item


def load_model(model_path: str = "models/fine_tuned"):
    """Load the fine-tuned model and tokenizer.

    Args:
        model_path: Path to the fine-tuned model

    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model: {e}")
        print("Falling back to base GPT-2 model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e2:
            print(f"Error: Could not load any model: {e2}")
            return None, None


def extract_price_from_text(text: str) -> Optional[float]:
    """Extract price from generated text.

    Args:
        text: Generated text containing price information

    Returns:
        Extracted price as float, or None if not found
    """
    # Look for price patterns like $12.99, $5, 12.99, etc.
    price_patterns = [
        r'\$[\d,]+\.?\d*',  # $12.99, $5, $1,234.56
        r'[\d,]+\.?\d*\s*dollars?',  # 12.99 dollars
        r'price[:\s]*[\$]?[\d,]+\.?\d*',  # price: $12.99
    ]

    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Extract the first numeric price found
            match = matches[0]
            # Remove $ and extract number
            numeric_part = re.search(r'[\d,]+\.?\d*', match.replace('$', ''))
            if numeric_part:
                price_str = numeric_part.group().replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue

    return None


def predict_price(description: str, model_path: str = "models/fine_tuned") -> str:
    """Predict price for a product description.

    Args:
        description: Product description
        model_path: Path to the model

    Returns:
        Prediction result as string
    """
    # Load model
    model, tokenizer = load_model(model_path)
    if not model or not tokenizer:
        return "Error: Could not load model"

    # Create item and prompt
    item = Item(description, "Unknown")
    prompt = item.make_prompt()

    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract price from generated text
        price = extract_price_from_text(generated_text)

        if price is not None:
            return f"Predicted price: ${price:.2f}"
        else:
            # Fallback: try to extract any reasonable price
            return f"Generated response: {generated_text[len(prompt):].strip()}"

    except Exception as e:
        return f"Error during prediction: {str(e)}"


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Predict product price from description")
    parser.add_argument("--description", required=True,
                       help="Product description for price prediction")
    parser.add_argument("--model-path", default="models/fine_tuned",
                       help="Path to the fine-tuned model")

    args = parser.parse_args()

    result = predict_price(args.description, args.model_path)
    print(result)


if __name__ == "__main__":
    main()