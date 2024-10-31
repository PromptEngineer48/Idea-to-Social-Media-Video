# Auto Video Generator 🎥

An AI-powered tool that automatically generates video content from news articles and search queries. This project combines web scraping, text summarization, voice generation, image generation, and text-to-speech to create engaging video content.

## 🌟 Features

- Web scraping of recent news articles
- AI-powered text summarization
- Automatic image generation based on content
- Text-to-speech narration
- Automatic video assembly with transitions and effects
- YouTube-ready title and description generation
- Easy-to-use Gradio web interface

## 🔧 Prerequisites

Before running this project, make sure you have the following API keys:
- EXA API key (for web scraping)
- Novita AI API key (for image generation)
- Deepgram API key (for text-to-speech)

## WorkFlow

![save](https://github.com/user-attachments/assets/fc36e165-5d57-4690-b9c1-2f6e1f3b87db)


## 🛠️ Installation

1. Create a virtual environment:
```bash
python -m venv AUTOVIDEOS
source AUTOVIDEOS/bin/activate  # On Windows: AUTOVIDEOS\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements_final.txt
```

3. Create a `.env` file in the project root with your API keys:
```env
EXA_API_KEY=your_exa_api_key
NOVITA_API_KEY=your_novita_api_key
BASE_URL=your_base_url(this can be found in novita api any model use file)
DG_API_KEY=your_deepgram_api_key
```

## 📚 Project Structure

```
AUTOVIDEOS/
├── audio/           # Generated audio files
├── images/          # Generated image files
├── output/          # Final video and text outputs
└── main.py         # Main application file
```

## 🚀 Usage

1. Run the application:
```bash
python main.py
```

2. Access the Gradio interface in your browser (typically at http://localhost:7860)

3. Enter a search query (e.g., "Latest News on Robotics")

4. Click "Generate" and wait for the process to complete

## 🎥 Video Generation Process

1. **Web Scraping**: Fetches recent articles related to your search query
2. **Text Processing**: 
   - Summarizes the articles into concise, engaging content
   - Generates prompts for image creation
3. **Media Generation**:
   - Creates images based on the content
   - Converts text to speech
4. **Video Assembly**:
   - Combines images with audio
   - Adds transitions and effects
   - Creates final video with subtitles
5. **Metadata Generation**:
   - Creates YouTube-ready title and description
   - Includes relevant hashtags

## ⚙️ Key Functions

- `web_scrapper()`: Fetches recent articles using the EXA API
- `generate_summary()`: Creates concise summaries using AI
- `generate_photo()`: Creates images using Novita AI
- `generate_voice()`: Converts text to speech using Deepgram
- `assemble_video()`: Combines all elements into final video
- `generate_youtube_description_and_title()`: Creates video metadata

## 🎨 Customization

You can customize various aspects of the video generation:
- Image dimensions (default: 1280x720)
- Audio amplification levels
- Video transitions and effects
- Text-to-speech voice settings
- Summary length and style

## ⚠️ Limitations

- Image generation requires stable internet connection
- Processing time varies based on content length
- API rate limits may apply
- Large videos may require significant processing power

## 📝 License

This project is open-source. Feel free to use, modify, and distribute as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 🙏 Acknowledgments

- Novita AI for image generation
- Deepgram for text-to-speech
- EXA for web scraping capabilities
- OpenAI for text processing
- Gradio for the user interface
