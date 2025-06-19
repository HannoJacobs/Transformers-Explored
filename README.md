# Transformers Explored

A comprehensive educational implementation of Transformer architectures from scratch, featuring both decoder-only (GPT-style) and encoder-decoder (T5-style) models for language modeling and machine translation tasks.

## 🎯 Project Overview

This repository provides clean, educational implementations of Transformer models designed for learning and experimentation. The codebase includes:

- **Decoder-Only Transformer**: Autoregressive language model (similar to GPT) for text generation
- **Encoder-Decoder Transformer**: Sequence-to-sequence model for machine translation
- **Multi-Head Attention**: Custom implementation with detailed annotations
- **Complete Training Pipeline**: Data loading, training, evaluation, and inference

## 🏗️ Architecture

### Core Components

- **Multi-Head Attention (MHA)**: From-scratch implementation with educational annotations (`src/mha.py`)
- **Positional Encoding**: Sinusoidal position embeddings
- **Custom Transformer Layers**: Pre-norm architecture with residual connections
- **Tokenization**: Word-level tokenizers for different tasks

### Model Variants

1. **Decoder-Only Model** (`src/decoder_only.py`)
   - Autoregressive text generation
   - Causal self-attention masking
   - Trained on Shakespeare text

2. **Encoder-Decoder Model** (`src/encoder_decoder.py`)
   - Machine translation (English → Afrikaans)
   - Cross-attention between encoder and decoder
   - Teacher forcing during training

## 📊 Datasets

### Shakespeare Text Generation
- **Files**: `tiny_shakespeare_{nano|mini|full}.txt`
- **Task**: Next-token prediction
- **Sizes**: Nano (46KB), Mini (151KB), Full (1.1MB)

### English-Afrikaans Translation
- **Source**: Parallel sentence pairs
- **Size**: 526,648 sentence pairs (filtered)
- **Max Length**: 10 words per sentence
- **Variants**: 1K, 10K, 100K, and full datasets
- **Format**: CSV with 'src' and 'target' columns

## 🚀 Quick Start

### Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train decoder-only model (text generation)
python src/decoder_only.py

# Train encoder-decoder model (translation)
python src/encoder_decoder.py
```

### Running Inference

```bash
# Generate text with trained decoder-only model
python src/infer_decoder_only.py

# Translate with trained encoder-decoder model
python src/infer_encoder_decoder.py
```

### Testing

```bash
# Run all tests
./test.sh

# Or run pytest directly
pytest tests/ -v -rP
```

## 📁 Project Structure

```
Transformers-Explored/
├── src/                          # Core implementation
│   ├── mha.py                   # Multi-Head Attention implementation
│   ├── decoder_only.py          # GPT-style decoder-only model
│   ├── encoder_decoder.py       # T5-style encoder-decoder model
│   ├── infer_decoder_only.py    # Text generation inference
│   └── infer_encoder_decoder.py # Translation inference
├── tests/                        # Unit tests
│   ├── test_mha.py              # MHA tests
│   ├── test_decoder.py          # Decoder-only tests
│   └── test_encoder_decoder.py  # Encoder-decoder tests
├── Datasets/                     # Training data
│   ├── tiny_shakespeare_*.txt   # Shakespeare corpus
│   ├── eng_afr/                 # Translation datasets
│   └── synth_*.csv              # Synthetic datasets
├── tools/                        # Utilities
│   ├── df_viewer.py             # Dataset viewer
│   └── translation_dataset_cleaner.ipynb
└── requirements.txt             # Dependencies
```

## 🔧 Configuration

### Model Hyperparameters

Both models use similar configurations that can be adjusted:

```python
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
D_MODEL = 512           # Model dimension
NHEAD = 8              # Number of attention heads
NUM_LAYERS = 4         # Number of transformer layers
DIM_FEEDFORWARD = 512  # FFN hidden dimension
DROPOUT = 0.1          # Dropout rate
```

### Sequence Lengths

- **Input Max Sequence Length**: 10 tokens
- **Output Max Sequence Length**: 10 tokens
- **Generation Max Length**: 100 tokens

## 📈 Features

### Educational Focus
- **Extensive Comments**: Every component is thoroughly documented
- **Clear Architecture**: Modular design for easy understanding
- **Step-by-Step Implementation**: From basic attention to full transformer

### Production-Ready Elements
- **GPU Support**: Automatic CUDA detection and usage
- **Proper Masking**: Causal and padding masks implemented correctly
- **Checkpointing**: Model saving and loading functionality
- **Evaluation Metrics**: Perplexity and loss tracking

### Inference Capabilities
- **Text Generation**: Configurable temperature and top-k sampling
- **Machine Translation**: Beam search and greedy decoding
- **Interactive Usage**: Easy-to-use inference scripts

## 🧪 Testing

The project includes comprehensive tests:

- **Unit Tests**: Individual component testing (MHA, layers)
- **Integration Tests**: Full model forward/backward passes
- **Shape Tests**: Tensor dimension validation
- **Gradient Tests**: Backpropagation verification

## 📚 Dependencies

- **PyTorch**: 2.2.2 - Deep learning framework
- **NumPy**: 1.26.4 - Numerical computing
- **Pandas**: 2.2.3 - Data manipulation
- **Matplotlib**: 3.8.4 - Visualization
- **scikit-learn**: 1.4.2 - ML utilities
- **pytest**: 8.4.0 - Testing framework

## 🎓 Educational Value

This implementation prioritizes:

1. **Clarity over Performance**: Code is optimized for understanding
2. **Complete Implementation**: No black-box components
3. **Progressive Complexity**: From simple attention to full models
4. **Real Applications**: Working on actual NLP tasks
