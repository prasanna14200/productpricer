"""Unit tests for the Product Pricer application."""

import unittest
from src.items import Item


class TestItem(unittest.TestCase):
    """Test cases for the Item class."""

    def test_item_creation(self):
        """Test creating an Item instance."""
        item = Item("Test Product", "Electronics", 29.99)

        self.assertEqual(item.name, "Test Product")
        self.assertEqual(item.category, "Electronics")
        self.assertEqual(item.price, 29.99)

    def test_item_creation_no_price(self):
        """Test creating an Item without price."""
        item = Item("Test Product", "Electronics")

        self.assertEqual(item.name, "Test Product")
        self.assertEqual(item.category, "Electronics")
        self.assertIsNone(item.price)

    def test_make_prompt(self):
        """Test creating a prompt for price prediction."""
        item = Item("Red Coffee Maker", "Kitchen", 49.99)
        prompt = item.make_prompt()

        expected = "Product: Red Coffee Maker\nCategory: Kitchen\nPrice:"
        self.assertEqual(prompt, expected)

    def test_test_prompt(self):
        """Test creating a test prompt."""
        item = Item("Red Coffee Maker", "Kitchen")
        prompt = item.test_prompt()

        expected = "How much does this cost: Red Coffee Maker?"
        self.assertEqual(prompt, expected)

    def test_parse_item(self):
        """Test parsing an item from text."""
        text = "Kitchen\nRed Coffee Maker"
        item = Item.parse(text)

        self.assertEqual(item.category, "Kitchen")
        self.assertEqual(item.name, "Red Coffee Maker")

    def test_str_representation(self):
        """Test string representation of Item."""
        item = Item("Test Product", "Electronics", 29.99)
        str_repr = str(item)

        self.assertIn("Test Product", str_repr)
        self.assertIn("Electronics", str_repr)
        self.assertIn("$29.99", str_repr)

    def test_str_representation_no_price(self):
        """Test string representation without price."""
        item = Item("Test Product", "Electronics")
        str_repr = str(item)

        self.assertIn("Test Product", str_repr)
        self.assertIn("Electronics", str_repr)
        self.assertNotIn("$", str_repr)


if __name__ == '__main__':
    unittest.main()