"""Item data class for product price prediction."""

import re
from typing import Optional
from transformers import AutoTokenizer


class Item:
    """Represents a product item with price prediction capabilities."""

    def __init__(self, name: str, category: str, price: Optional[float] = None):
        """Initialize an Item.

        Args:
            name: Product name/description
            category: Product category
            price: Optional price for training data
        """
        self.name = name
        self.category = category
        self.price = price

    @classmethod
    def parse(cls, text: str) -> 'Item':
        """Parse item from text format.

        Args:
            text: Text containing item information

        Returns:
            Item instance
        """
        # Extract category and name from text
        lines = text.strip().split('\n')
        category = lines[0] if lines else "Unknown"
        name = ' '.join(lines[1:]) if len(lines) > 1 else category

        return cls(name=name, category=category)

    def make_prompt(self) -> str:
        """Create a prompt for price prediction.

        Returns:
            Formatted prompt string
        """
        return f"Product: {self.name}\nCategory: {self.category}\nPrice:"

    def test_prompt(self) -> str:
        """Create a test prompt for validation.

        Returns:
            Test prompt string
        """
        return f"How much does this cost: {self.name}?"

    def __str__(self) -> str:
        """String representation of the item."""
        price_str = f" (${self.price:.2f})" if self.price else ""
        return f"{self.category}: {self.name}{price_str}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Item(name='{self.name}', category='{self.category}', price={self.price})"