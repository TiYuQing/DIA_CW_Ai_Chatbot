## Installation Requirements

Before running the code, ensure you have the following dependencies installed:

```bash
# Install PyTorch with CUDA 12.1 support
pip install torchvision==0.18.0 torchaudio==2.3.0 torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install transformer-related packages
pip install --upgrade transformers==4.41.0 accelerate==0.30.0 bitsandbytes==0.43.0

# Install ChromaDB for vector storage
pip install chromadb

# Optional: Create and activate virtual environment
python -m venv chatbot-env
source chatbot-env/bin/activate  # Linux/MacOS
# or
# chatbot-env\Scripts\activate  # Windows
