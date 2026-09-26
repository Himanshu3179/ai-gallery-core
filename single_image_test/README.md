# Single Image Testing (Moondream2)

This tool allows you to test image processing and description generation on a single image using **Moondream2** (`vikhyatk/moondream2`).

## Features
- **Takes image path** via CLI argument or interactive prompt.
- **Extracts detailed metadata**: Dimensions, file size (bytes & human-readable), format, color mode, timestamps, and EXIF summary.
- **Tracks inference performance**: Measures exact generation time in seconds.
- **Outputs JSON to console** and maintains [`output.json`](./output.json).
- **Prepends new results**: If [`output.json`](./output.json) exists, newly generated results are automatically prepended to the **top** (newest first).

---

## How to Run

### Option 1: Pass image path as argument (Fast Default: ~2-3s)
```bash
python single_image_test/test_single.py "/path/to/your/image.jpg"
```

### Option 2: Custom token limit & prompt (optional)
```bash
# Generate a strict 1-sentence caption with max 35 tokens (~2s)
python single_image_test/test_single.py "/path/to/your/image.jpg" --max-tokens 35 --prompt "Give a 1-sentence caption."

# Or generate a detailed essay if desired (slower ~8-12s)
python single_image_test/test_single.py "/path/to/your/image.jpg" --max-tokens 200 --prompt "Describe this image in detail, focusing on setting, colors, and lighting."
```

### Option 3: Interactive input
```bash
python single_image_test/test_single.py
# It will prompt: Enter image path:
```

---

## 🔬 Multi-Model Comparison Tool (`compare_models.py`)

Benchmark **4 different vision-language models** side-by-side on the same image:

1. **BLIP-Base** (`Salesforce/blip-image-captioning-base`, ~220M)
2. **GIT-Base** (`microsoft/git-base`, ~176M)
3. **SmolVLM-500M** (`HuggingFaceTB/SmolVLM-500M-Instruct`, ~500M)
4. **Moondream2** (`vikhyatk/moondream2`, ~1.86B)

### Run Comparison:
```bash
python single_image_test/compare_models.py "/path/to/your/image.jpg"
```

* Displays a clean comparative table in the terminal with parameter sizes, load times, inference times, and generated captions.
* Automatically saves/prepends the comparison report to [`comparison_output.json`](./comparison_output.json).

---

## ⚡ Direct Dual-Encoder Benchmark (`test_dual_encoder.py`)

Benchmark **direct pixel-to-vector embedding** using CLIP (`openai/clip-vit-base-patch32`) for ultra-high throughput indexing (**10,000+ images in < 1 minute**):

### Run Dual-Encoder:
```bash
python single_image_test/test_dual_encoder.py "/path/to/your/image.jpg"
```

* Encodes image directly to a 512-dimensional vector in **~3.5 milliseconds**.
* Demonstrates live zero-shot semantic text search matching against test queries.
* Automatically saves reports to [`dual_encoder_output.json`](./dual_encoder_output.json).

---

## ⚡ Performance & Batch Optimization
For in-depth analysis of inference latency, timing breakdowns, and recommendations for batch processing (100–200+ photos), see the [Performance & Batch Optimization Guide](../docs/PERFORMANCE_AND_OPTIMIZATIONS.md).
