"""Data loading utilities for parallel processing."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from .items import Item


class ItemLoader:
    """Handles loading and processing of item data."""

    @staticmethod
    def from_chunk(chunk: List[Dict[str, Any]]) -> List[Item]:
        """Convert a chunk of data into Item objects.

        Args:
            chunk: List of dictionaries containing item data

        Returns:
            List of Item objects
        """
        items = []
        for item_data in chunk:
            try:
                # Extract relevant fields from the data
                name = item_data.get('title', item_data.get('name', 'Unknown Product'))
                category = item_data.get('category', 'Unknown')
                price = item_data.get('price')

                # Convert price to float if it's a string
                if isinstance(price, str):
                    # Extract numeric price from string
                    price_match = re.search(r'[\d,]+\.?\d*', price.replace('$', ''))
                    if price_match:
                        price = float(price_match.group().replace(',', ''))

                item = Item(name=name, category=category, price=price)
                items.append(item)
            except Exception as e:
                print(f"Warning: Failed to parse item: {e}")
                continue

        return items

    @staticmethod
    def load_in_parallel(data: List[Dict[str, Any]], max_workers: int = 4) -> List[Item]:
        """Load items in parallel for better performance.

        Args:
            data: Full dataset as list of dictionaries
            max_workers: Maximum number of worker threads

        Returns:
            List of all Item objects
        """
        if not data:
            return []

        # Split data into chunks
        chunk_size = max(1, len(data) // max_workers)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ItemLoader.from_chunk, chunk) for chunk in chunks]
            results = []
            for future in futures:
                results.extend(future.result())

        return results

    @staticmethod
    def save_to_jsonl(items: List[Item], filepath: str):
        """Save items to JSONL format for training.

        Args:
            items: List of Item objects
            filepath: Path to save the file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in items:
                if item.price is not None:
                    data = {
                        "text": f"{item.make_prompt()} ${item.price:.2f}",
                        "price": item.price
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

    @staticmethod
    def load_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file.

        Args:
            filepath: Path to the JSONL file

        Returns:
            List of training examples
        """
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found")
        return data