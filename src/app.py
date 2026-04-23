"""Gradio web interface for product price prediction."""

import gradio as gr
from src.inference import predict_price


def create_interface():
    """Create the Gradio interface for price prediction.

    Returns:
        Gradio Interface object
    """
    def predict(description):
        """Prediction function for Gradio interface.

        Args:
            description: Product description input

        Returns:
            Prediction result
        """
        if not description.strip():
            return "Please enter a product description."

        return predict_price(description)

    # Create the interface
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Enter product description (e.g., 'A red coffee maker with timer')",
            label="Product Description"
        ),
        outputs=gr.Textbox(label="Predicted Price"),
        title="🛍️ Product Pricer",
        description="Predict product prices using AI! Enter a product description and get an estimated price.",
        examples=[
            ["A red coffee maker with programmable timer"],
            ["Wireless Bluetooth headphones with noise cancellation"],
            ["Stainless steel kitchen knife set"],
            ["LED desk lamp with adjustable brightness"],
            ["Yoga mat with carrying strap"]
        ],
        theme="default"
    )

    return iface


# Create the interface
iface = create_interface()

if __name__ == "__main__":
    # Launch the interface
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True for public sharing
    )