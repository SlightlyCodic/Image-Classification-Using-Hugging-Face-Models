# 🖼️ AI Image Classification Dashboard

A Streamlit-powered web app for classifying images using state-of-the-art AI models from Hugging Face. Instantly upload images, get predictions with confidence scores, and view detailed analytics—all in your browser.

## Features
- **Image Classification**: Classify single images using top models like ViT and ResNet-50.
- **Model Selection**: Choose between Google ViT (recommended) and Microsoft ResNet-50 (faster).
- **Prediction Analytics**: View confidence scores, top predictions, and download results as CSV.
- **History & Dashboard**: Track all classified images, review predictions, and analyze trends.
- **Interactive Visualizations**: Bar charts and histograms for prediction classes and confidence distribution.

## Demo
![Demo Screenshot](demo_screenshot.png)

## Getting Started
### Prerequisites
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SlightlyCodic/Image-Classification-Using-Hugging-Face-Models.git
   cd Image-Classification-Using-Hugging-Face-Models
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install manually:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn pillow torch transformers
   ```

### Run the App
```bash
streamlit run main.py
```

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Upload an image (JPG, JPEG, PNG).
- Select your preferred model and prediction options in the sidebar.
- Click "Load Model" and then "Classify Image".
- View predictions, analytics, and download results.

## File Structure
```
main.py            # Streamlit app source code
requirements.txt   # Python dependencies (recommended)
README.md          # Project documentation
```

## Models Used
- [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)

## Author
By Chirag Joshi

## License
This project is licensed under the MIT License.
