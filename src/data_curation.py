"""Data curation script for preparing training data."""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from src.items import Item
from src.loaders import ItemLoader


def load_amazon_reviews(category: str = "All_Beauty", max_samples: int = 10000) -> List[Dict[str, Any]]:
    """Load Amazon reviews dataset.

    Args:
        category: Product category to load
        max_samples: Maximum number of samples to load

    Returns:
        List of review data
    """
    try:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}",
                              split="full", trust_remote_code=True)
        data = []

        for i, item in enumerate(dataset):
            if i >= max_samples:
                break

            # Extract relevant information
            review_data = {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "rating": item.get("rating", 0),
                "price": item.get("price", None),
                "category": category
            }
            data.append(review_data)

        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def filter_valid_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter data to include only items with valid prices and titles.

    Args:
        data: Raw review data

    Returns:
        Filtered data with valid items
    """
    valid_data = []

    for item in data:
        title = item.get("title", "").strip()
        price = item.get("price")

        # Skip items without title or price
        if not title or price is None:
            continue

        # Convert price to float if it's a string
        if isinstance(price, str):
            # Extract numeric price
            import re
            price_match = re.search(r'[\d,]+\.?\d*', price.replace('$', ''))
            if price_match:
                price = float(price_match.group().replace(',', ''))
                item["price"] = price
                valid_data.append(item)

    return valid_data


def create_training_data(items: List[Item], output_dir: str = "data/processed"):
    """Create training and validation datasets.

    Args:
        items: List of Item objects
        output_dir: Output directory for processed data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter items with prices
    priced_items = [item for item in items if item.price is not None]

    if not priced_items:
        print("Warning: No items with prices found!")
        return

    # Split into train/validation
    split_idx = int(0.8 * len(priced_items))
    train_items = priced_items[:split_idx]
    val_items = priced_items[split_idx:]

    # Save to JSONL format
    ItemLoader.save_to_jsonl(train_items, os.path.join(output_dir, "fine_tune_train.jsonl"))
    ItemLoader.save_to_jsonl(val_items, os.path.join(output_dir, "fine_tune_validation.jsonl"))

    print(f"Created training data: {len(train_items)} train, {len(val_items)} validation samples")


def main():
    """Main data curation function."""
    parser = argparse.ArgumentParser(description="Prepare training data for product pricer")
    parser.add_argument("--category", default="All_Beauty",
                       help="Amazon product category (default: All_Beauty)")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum number of samples to load (default: 10000)")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Output directory for processed data")

    args = parser.parse_args()

    print(f"Loading {args.category} dataset with max {args.max_samples} samples...")

    # Load data
    data = load_amazon_reviews(args.category, args.max_samples)
    print(f"Loaded {len(data)} raw samples")

    # Filter valid items
    valid_data = filter_valid_items(data)
    print(f"Found {len(valid_data)} valid items with prices")

    # Convert to Item objects
    items = ItemLoader.load_in_parallel(valid_data)
    print(f"Processed {len(items)} items")

    # Create training data
    create_training_data(items, args.output_dir)

    print("Data curation completed!")


if __name__ == "__main__":
    main()