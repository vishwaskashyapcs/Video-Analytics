# Video Speech-to-Text Transcriber

A modern web application that extracts audio from uploaded videos and transcribes speech to text using Sarvam AI's powerful speech recognition API with automatic language detection and English translation.

## Features

✨ **Modern UI Design** - Beautiful, responsive interface with glassmorphism effects
🎥 **Video Upload** - Support for multiple video formats (MP4, AVI, MOV, MKV, WMV, FLV, WebM)
🎵 **Audio Extraction** - Automatic audio extraction from video files using FFmpeg
🌍 **Auto Language Detection** - Automatically detects the spoken language
🔤 **English Translation** - Translates speech to English for universal understanding
📝 **Interactive Results** - Copy to clipboard and download transcript features
🧠 **Sentiment Analysis** - AI-powered sentiment, emotions, reasons, and suggestions
📱 **Mobile Responsive** - Works seamlessly on desktop and mobile devices

## Prerequisites

Before running this application, make sure you have the following installed:

1. **Python 3.7+** - [Download Python](https://www.python.org/downloads/)
2. **FFmpeg** - Required for audio extraction from videos

### Installing FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract and add to your system PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Installation

1. **Clone or download this project**
   ```bash
   cd "/Users/krishsharma/Desktop/speech to text"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify FFmpeg installation**
   ```bash
   ffmpeg -version
   ```

## Configuration

The application is already configured with your Sarvam AI API key in the `.env` file. The API key is securely stored and loaded automatically.

## Running the Application

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to**
   ```
   http://localhost:8080
   ```

3. **Upload a video file and get instant transcription!**

### Sentiment Analysis

After transcription, the app automatically performs sentiment analysis using the Sarvam Chat Completions API and displays:

- Overall sentiment (positive, negative, neutral, or mixed) with confidence
- Detected emotions with intensities
- Reasons behind the classification
- Strengths and weaknesses in the feedback
- Actionable suggestions (especially for negative/mixed sentiment)

## Usage

1. **Upload Video**: Click the upload area or drag & drop a video file
2. **Supported Formats**: MP4, AVI, MOV, MKV, WMV, FLV, WebM (Max: 100MB)
3. **Processing**: The app will automatically extract audio and transcribe it
4. **Results**: View the transcribed text with options to copy or download

## API Integration

This application uses the **Sarvam AI Speech-to-Text Translation API** and the **Sarvam Chat Completions API** which provide:

- **Automatic Language Detection**: Detects the spoken language in your video
- **High-Quality Transcription**: Advanced AI-powered speech recognition
- **English Translation**: Translates non-English speech to English
- **Multiple Audio Formats**: Supports various audio formats extracted from videos
- **Sentiment & Emotions**: Post-transcription analysis with reasons and suggestions

## File Structure

```
speech to text/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API key)
├── README.md             # This file
├── uploads/              # Temporary file storage
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Upload page
│   └── result.html      # Results page
└── static/              # Static assets
    └── css/
        └── style.css    # Custom CSS styles
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Make sure FFmpeg is installed and available in your system PATH
   - Test with: `ffmpeg -version`

2. **File upload fails**
   - Check file size (max 100MB)
   - Ensure the file format is supported
   - Verify the uploads directory exists

3. **API errors**
   - Verify your Sarvam AI API key is correct
   - Check your internet connection
   - Ensure you have API credits remaining

4. **Audio extraction fails**
   - Try a different video format
   - Check if the video file is corrupted
   - Ensure FFmpeg is properly installed

### Error Messages

- **"FFmpeg not found"**: Install FFmpeg and add it to your PATH
- **"Invalid file type"**: Use a supported video format
- **"File too large"**: Reduce video file size to under 100MB
- **"API Error"**: Check your API key and internet connection

## Development

To contribute to this project or modify it:

1. **Development Mode**: The app runs in debug mode by default
2. **Code Structure**: Main logic is in `app.py`, templates in `templates/`
3. **Styling**: Custom CSS in `static/css/style.css`, using Tailwind CSS
4. **Dependencies**: Add new requirements to `requirements.txt`

## Security Notes

- API keys are stored in `.env` file (not committed to version control)
- Uploaded files are automatically deleted after processing
- File size limits prevent abuse
- Input validation prevents malicious uploads

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check the console for error messages
4. Ensure your Sarvam AI API key is valid

---

**Built with Flask and modern web technologies**