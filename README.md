# Audio-to-Image Generation Model

A sophisticated AI-powered system that transforms audio inputs into visual representations by leveraging state-of-the-art audio classification and image generation models.

## 🎯 Overview

This project implements an innovative audio-to-image generation pipeline that analyzes audio content and generates corresponding images. The system uses **MediaPipe's YAMNet** model for audio classification and **Playground v2.5** (based on Stable Diffusion XL) for high-quality image generation.

## 🏗️ Architecture

The system follows a five-stage pipeline architecture:

![Architecture Diagram](Architecture%20diagram.drawio.png)

### Pipeline Stages:

1. **Audio Input**: Accepts audio files in `.wav` or `.mp3` format
2. **Audio Classifier**: Processes audio using MediaPipe YAMNet neural network
3. **Classification Processing**: Extracts and ranks audio categories against timestamps
4. **Prompt Preparation**: Generates descriptive text prompts from top-ranked categories
5. **Image Generation**: Creates images using Playground diffusion model
6. **Output**: Returns high-quality 1024x1024 images

## ✨ Features

- **Multi-format Support**: Works with both WAV and MP3 audio files
- **Advanced Audio Classification**: Utilizes MediaPipe's YAMNet model for accurate audio content recognition
- **Temporal Analysis**: Classifies audio content across multiple timestamps
- **Intelligent Prompt Generation**: Automatically extracts top 2 most frequent audio categories
- **High-Quality Image Generation**: Produces 1024x1024 images using Playground v2.5 model
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with FP16 precision
- **Flexible Configuration**: Customizable inference steps and guidance scale

## 🔧 How It Works

### Complete Workflow:

#### 1. Audio Input Processing
- Load audio file in `.wav` or `.mp3` format
- Convert MP3 to WAV if necessary using the provided utility function
- Prepare audio data for classification

#### 2. Audio Classification with YAMNet
- MediaPipe's YAMNet neural network analyzes the audio
- Generates classifications for each timestamp in the audio
- Outputs category labels with confidence scores
- **Example output:**
  ```
  Timestamp 0: Dog (0.26)
  Timestamp 1: Domestic animals, pets (0.20)
  Timestamp 2: Domestic animals, pets (0.26)
  ```

#### 3. Category Extraction & Processing
- `get_top_categories()` function processes the classifier output
- Counts category occurrences across all timestamps
- Identifies and returns the **2 most frequent categories**

#### 4. Text Prompt Generation
- Creates a descriptive prompt: `"generate an image of [category1], [category2]"`
- Prepares the prompt for the image generation model

#### 5. Image Generation with Playground
- Playground v2.5 model (based on Stable Diffusion XL) generates image from the text prompt
- Uses 50 inference steps with a guidance scale of 3
- Outputs a 1024x1024 RGB image representing the audio content

## 📋 Requirements

### Python Packages:
```bash
mediapipe          # Audio classification with YAMNet
diffusers          # Image generation pipeline
torch              # Deep learning framework
pydub              # Audio format conversion
ffmpeg             # Audio processing backend
scipy              # WAV file handling
numpy              # Numerical operations
transformers       # Model utilities
peft               # Parameter-efficient fine-tuning
```

### Hardware Requirements:
- **GPU**: CUDA-capable GPU (recommended for optimal performance)
- **GPU Memory**: Minimum 8GB VRAM
- **RAM**: Minimum 16GB system RAM

### Models:
- **YAMNet**: Audio classifier (TFLite format, ~1MB)
- **Playground v2.5**: Image generation model (1024px aesthetic variant, FP16)

## 🚀 Installation

```bash
# Install MediaPipe for audio classification
pip install -q mediapipe

# Install audio processing libraries
pip install pydub ffmpeg

# Install model libraries
pip install datasets transformers peft

# Install/upgrade Diffusers for image generation
pip install --upgrade git+https://github.com/huggingface/diffusers.git

# Download YAMNet classifier model
wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
```

## 💻 Usage

### Complete Pipeline Example:

```python
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile
from pydub import AudioSegment
from diffusers import DiffusionPipeline
import torch
from collections import Counter

# Load the image generation model
pipe = DiffusionPipeline.from_pretrained(
    "playgroundai/playground-v2.5-1024px-aesthetic",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Step 1: Convert MP3 to WAV (if needed)
def convert_mp3_to_wav(input_mp3_path, output_wav_path):
    audio = AudioSegment.from_mp3(input_mp3_path)
    audio.export(output_wav_path, format="wav")
    print(f"Conversion successful: {output_wav_path}")

# Step 2: Classify audio
def audio_classifier(audio_input):
    base_options = python.BaseOptions(model_asset_path='classifier.tflite')
    options = audio.AudioClassifierOptions(
        base_options=base_options, max_results=4)
    
    with audio.AudioClassifier.create_from_options(options) as classifier:
        sample_rate, wav_data = wavfile.read(audio_input)
        audio_clip = containers.AudioData.create_from_array(
            wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
        classification_result_list = classifier.classify(audio_clip)
        
        results = ""
        for idx, timestamp in enumerate(range(len(classification_result_list))):
            classification_result = classification_result_list[idx]
            top_category = classification_result.classifications[0].categories[0]
            output = f'Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})'
            results += output + "\n"
    
    return results

# Step 3: Extract top categories
def get_top_categories(input_string):
    lines = input_string.strip().split("\n")
    categories = []
    
    for line in lines:
        try:
            category_start = line.find(":") + 2
            category_end = line.rfind("(") - 1
            category = line[category_start:category_end].strip()
            categories.append(category)
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
    
    category_counts = Counter(categories)
    top_categories = [category for category, count in category_counts.most_common(2)]
    
    return ", ".join(top_categories)

# Step 4: Generate image from audio
audio_file_name = "your_audio.wav"
audio_classifier_output = audio_classifier(audio_file_name)

prompt = f"generate an image of {get_top_categories(audio_classifier_output)}"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]

# Step 5: Display or save the image
image.show()
image.save("output_image.png")
```

### Quick Start:

```python
# For MP3 files
convert_mp3_to_wav("input.mp3", "output.wav")

# Classify and generate
audio_file = "output.wav"
classification = audio_classifier(audio_file)
prompt = f"generate an image of {get_top_categories(classification)}"
image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=3).images[0]
image.show()
```

## 🎛️ Configuration

### Audio Classifier Settings:
- **max_results**: Number of top classification results per timestamp (default: 4)
- **model_asset_path**: Path to YAMNet TFLite model
- **Input format**: 16kHz mono WAV audio

### Image Generation Settings:
- **num_inference_steps**: Number of denoising steps (default: 50, range: 20-100)
- **guidance_scale**: Controls prompt adherence (default: 3, range: 1-20)
- **torch_dtype**: Precision format (default: `torch.float16`)
- **variant**: Model variant (default: `"fp16"`)
- **Output resolution**: 1024x1024 pixels

## 🔬 Technical Details

### Audio Classification:
- **Model**: MediaPipe YAMNet (TensorFlow Lite)
- **Architecture**: MobileNet-based audio event classifier
- **Training Data**: AudioSet (YouTube-8M)
- **Classes**: 521 audio event categories
- **Input**: 16kHz mono WAV audio
- **Output**: Category labels with confidence scores per timestamp
- **Processing**: Temporal segmentation with overlapping windows

### Image Generation:
- **Model**: Playground v2.5
- **Base Architecture**: Stable Diffusion XL
- **Components**: U-Net with cross-attention layers
- **Text Encoder**: CLIP-based dual text encoders
- **VAE**: Variational autoencoder for image encoding/decoding
- **Precision**: FP16 for efficient inference
- **Resolution**: 1024x1024 pixels
- **Scheduler**: EDM (Euler Diffusion Model) scheduler

## 📁 Project Structure

```
Audio-to-image-Generation-Model/
├── README.md                          # Project documentation (this file)
├── Architecture diagram.drawio.png    # System architecture visualization
└── Complete_workflow.ipynb            # Complete audio-to-image pipeline implementation
```

## 🎓 Model Information

### YAMNet Audio Classifier:
- **Developer**: Google Research
- **Training Dataset**: AudioSet (over 2 million human-labeled 10-second sound clips)
- **Recognition**: 521 audio event classes
- **Size**: ~1MB (TFLite format)
- **Efficiency**: Lightweight model optimized for real-time inference
- **Framework**: TensorFlow Lite

### Playground v2.5:
- **Developer**: Playground AI
- **Base**: Stable Diffusion XL architecture
- **Optimization**: Aesthetic image generation at 1024px
- **Variant**: FP16 precision for reduced memory footprint
- **Size**: ~5.1GB (FP16 variant)
- **Framework**: Diffusers (Hugging Face)

## 🎨 Use Cases

- **Audio Visualization**: Create visual representations of music, soundscapes, or ambient audio
- **Content Creation**: Generate artwork inspired by audio recordings
- **Accessibility**: Provide visual descriptions of audio content
- **Educational Tools**: Help students understand audio-visual correlations
- **Creative Applications**: Experimental art projects combining audio and visual media

## ⚠️ Limitations

- Requires significant GPU memory (minimum 8GB VRAM)
- Processing time depends on audio length and GPU capability
- Image quality depends on the clarity and distinctness of audio categories
- Works best with audio containing identifiable sound events
- MP3 to WAV conversion may affect classification accuracy

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check issues page or submit a pull request.


## 🙏 Acknowledgments

- **MediaPipe Team** - For the YAMNet audio classification model
- **Playground AI** - For the Playground v2.5 image generation model
- **Hugging Face** - For the Diffusers library and model hosting
- **Google Research** - For AudioSet dataset and YAMNet architecture
- **Stability AI** - For the Stable Diffusion foundation

## 📞 Support

For questions or issues, please open an issue in the repository or refer to `Complete_workflow.ipynb` for detailed implementation.

---

**Note**: This project requires significant computational resources. GPU acceleration is highly recommended for practical usage. The first run will download model weights (~5GB), which may take time depending on your internet connection.