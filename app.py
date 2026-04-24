"""Root app launcher for Hugging Face Spaces."""

from src.app import iface

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
