## Created a virtual environment called AUTOVIDEOS


import os
import gradio as gr
import requests
import time
from exa_py import Exa
from dotenv import load_dotenv
from openai import OpenAI
from typing import Union, List
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import moviepy.editor as mp
from PIL import Image
import random
import re

# Ensure you download the punkt tokenizer if you haven't done so
nltk.download("punkt")

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

# Load environment variables from a .env file
load_dotenv()

# Retrieve Novita API key, and base URL from environment variables
exa = Exa(os.getenv("EXA_API_KEY"))
novita_api_key = os.getenv("NOVITA_API_KEY")
base_url = os.getenv("BASE_URL")
dg_api_key = os.getenv("DG_API_KEY")
flux_api_key = os.getenv("FLUX_API_KEY")


# Initialize OpenAI client with base URL and Novita API key
client = OpenAI(
    base_url=base_url,
    api_key=novita_api_key,
)

# Set up necessary directories
os.makedirs("audio", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)

generate_summary_model = "meta-llama/llama-3.1-8b-instruct"
generate_photo_model = "meta-llama/llama-3.1-8b-instruct"


def web_scrapper(search_query):

    from datetime import datetime, timedelta

    one_week_ago = datetime.now() - timedelta(days=7)
    date_cutoff = one_week_ago.strftime("%Y-%m-%d")

    search_response = exa.search_and_contents(
        search_query, use_autoprompt=True, start_published_date=date_cutoff
    )

    urls = [result.url for result in search_response.results]
    print("\nURLs:")
    for url in urls:
        print(url)

    results = search_response.results

    result_items = "\n".join(result.text for result in results)

    print(f"{len(results)} items total, stored in result_items:")

    return result_items, urls


def generate_summary(article_text):
    # Generate a list containing individual sentences

    chat_completion_res = client.chat.completions.create(
        model=generate_summary_model,
        messages=[
            {
                "role": "system",
                "content": "Summarize the article text given by the user below in 4-5 sentences strictly, beginning directly with the main points. Do not include phrases like 'Here’s a summary of the article text' or 'In this article,' etc. Start directly with the summary content itself. Keep the content engaging and funny. Always end with a thank you note and ask for subscription for the YouTube channel.",
            },
            {
                "role": "user",
                "content": f"{article_text}",
            },
        ],
        max_tokens=1048,
    )

    results = chat_completion_res.choices[0].message.content

    # Find the first colon and take everything after it
    first_colon_index = results.find(":")

    if first_colon_index != -1:
        summary_content = results[
            first_colon_index + 1 :
        ].strip()  # Take content after the first colon
    else:
        summary_content = results  # In case there's no colon, use the full result

    sentences = sent_tokenize(summary_content)
    sentences = [sentence for sentence in sentences if sentence]

    return sentences


def generate_photo_prompt(sentence) -> str | None:
    # Generate a list containing individual sentences

    chat_completion_res = client.chat.completions.create(
        model=generate_photo_model,
        messages=[
            {
                "role": "system",
                "content": """
                        Generate a vivid and well-crafted prompt for photo generation based on the user's input, focusing on imaginative and precise descriptions. Use rich, evocative language to bring out the atmosphere, details, and aesthetic style that align with the user’s requirements. No need for any explanations. Start giving the answer directly. Keep your answer within 2 sentences.
                """,
            },
            {
                "role": "user",
                "content": f"User's Sentence: \n{sentence}",
            },
        ],
        max_tokens=1048,
    )

    results = chat_completion_res.choices[0].message.content
    return results


def generate_voice(sentence, serial_number) -> None:
    try:
        filename = f"audio/audio{serial_number}.wav"
        SPEAK_OPTIONS = {"text": sentence}

        # STEP 1: Create a Deepgram client using the API key from environment variables
        deepgram = DeepgramClient(api_key=dg_api_key)

        # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
        options = SpeakOptions(
            model="aura-zeus-en", encoding="linear16", container="wav"
        )

        # STEP 3: Call the save method on the speak property
        response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")


def amplify_audio(serial_number, gain_dB=5) -> None:
    """Increase the volume of an audio file."""

    filename = f"audio/audio{serial_number}.wav"
    audio = AudioSegment.from_wav(filename)
    louder_audio = audio + gain_dB
    louder_audio.export(filename, format="wav")
    print(f"Volume increased by {gain_dB} db")


def generate_photo(photo_description: str, serial_number: int) -> None:
    """Generate a photo based on the provided description."""

    from novita_client import (
        NovitaClient,
        Txt2ImgRequest,
        Samplers,
        ModelType,
        save_image,
    )

    photo_client = NovitaClient(novita_api_key)

    req = Txt2ImgRequest(
        model_name="sdxlUnstableDiffusers_v11_216694.safetensors",
        prompt=photo_description,
        negative_prompt="",
        width=1280,
        height=720,
        sampler_name="Euler a",
        cfg_scale=7,
        steps=28,
        batch_size=1,
        n_iter=1,
        seed=0,
    )

    save_image(
        photo_client.sync_txt2img(req).data.imgs_bytes[0],
        f"images/photo{serial_number}.png",
    )


# def generate_photo_sd3(
#     photo_description: str, serial_number: int, max_retries: int = 10, delay: int = 5
# ) -> None:
#     """Generate a photo based on the provided description and save it with a serial number in the filename.

#     Args:
#         photo_description (str): Description of the photo to generate.
#         serial_number (int): Serial number for saving the photo.
#         max_retries (int): Maximum number of retries for checking task status.
#         delay (int): Delay in seconds between retries.
#     """

#     # Initial request to start image generation
#     url = "https://api.novita.ai/v3/async/txt2img"
#     headers = {
#         "Authorization": f'Bearer {os.getenv("NOVITA_API_KEY")}',
#         "Content-Type": "application/json",
#     }
#     data = {
#         "extra": {"response_image_type": "png"},
#         "request": {
#             "model_name": "sd3_base_medium.safetensors",
#             "prompt": photo_description,
#             "width": 1080,
#             "height": 720,
#             "image_num": 1,
#             "steps": 28,
#             "seed": -1,
#             "clip_skip": 1,
#             "guidance_scale": 4.5,
#             "sampler_name": "FlowMatchEuler",
#         },
#     }

#     # Step 1: Submit the generation request
#     response = requests.post(url, headers=headers, json=data)

#     if response.status_code == 200:
#         response_data = response.json()
#         task_id = response_data.get("task_id")
#         if not task_id:
#             print("Task ID not found in response.")
#             return

#         # Step 2: Polling to check if the image is ready
#         result_url = f"https://api.novita.ai/v3/async/task-result?task_id={task_id}"

#         for attempt in range(max_retries):
#             print(f"Try# {attempt}")
#             result_response = requests.get(result_url, headers=headers)

#             if result_response.status_code == 200:
#                 result_data = result_response.json()
#                 task_status = result_data.get("task", {}).get("status")

#                 # Check if task is completed
#                 if task_status == "TASK_STATUS_SUCCEED":
#                     images = result_data.get("images", [])
#                     if images:
#                         for idx, image_info in enumerate(images):
#                             image_url = image_info.get("image_url")
#                             if image_url:
#                                 # Download the image from the URL
#                                 img_response = requests.get(image_url)
#                                 if img_response.status_code == 200:
#                                     with open(
#                                         f"images/photo{serial_number}.png", "wb"
#                                     ) as img_file:
#                                         img_file.write(img_response.content)
#                                     print(
#                                         f"Photo saved as photo_{serial_number}_{idx}.jpeg"
#                                     )
#                                 else:
#                                     print(f"Failed to download image from {image_url}")
#                         return  # Exit once images are saved
#                     else:
#                         print("No images found in the completed response.")
#                         return
#                 elif task_status == "TASK_STATUS_FAILED":
#                     print("Image generation task failed.")
#                     return

#             else:
#                 print(
#                     "Error retrieving task result:",
#                     result_response.status_code,
#                     result_response.text,
#                 )
#                 return

#             # Wait before retrying
#             print(
#                 f"Attempt {attempt + 1}/{max_retries}: Waiting for {delay} seconds before retrying..."
#             )
#             time.sleep(delay)

#         print("Image generation timed out.")
#     else:
#         print("Error:", response.status_code, response.text)


def save_sentences_to_file(sentences: List[str], filename: str) -> None:
    with open(filename, "w") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def generate_youtube_description_and_title(script):
    # Generate a list containing individual sentences

    chat_completion_res = client.chat.completions.create(
        model=generate_summary_model,
        messages=[
            {
                "role": "system",
                "content": "Generate a catchy youtube title and an excellent long description for my youtube video based on the script given below. Also, include lots of hashtags. Remember you are creating a youtube video title and description.",
            },
            {
                "role": "user",
                "content": f"{script}",
            },
        ],
        max_tokens=1048,
    )

    results = chat_completion_res.choices[0].message.content

    return results


import os
from novita_client import NovitaClient
from novita_client.utils import base64_to_image


def img2video() -> None:
    from novita_client import (
        NovitaClient,
        Txt2ImgRequest,
        Samplers,
        ModelType,
        save_image,
    )

    photo_client = NovitaClient(novita_api_key)

    res = photo_client.txt2video(
        model_name="darkSushiMixMix_225D_64380.safetensors",
        width=1024,
        height=576,
        guidance_scale=7.5,
        steps=20,
        seed=-1,
        # prompts=[
        #     {"prompt": "A girl, baby, portrait, 5 years old", "frames": 16},
        #     {"prompt": "A girl, child, portrait, 10 years old", "frames": 16},
        #     {"prompt": "A girl, teen, portrait, 20 years old", "frames": 16},
        #     {"prompt": "A girl, woman, portrait, 30 years old", "frames": 16},
        #     {"prompt": "A girl, woman, portrait, 50 years old", "frames": 16},
        #     {"prompt": "A girl, old woman, portrait, 70 years old", "frames": 16},
        # ],
        prompts=[
            {
                "prompt": "A forest in early spring, young green leaves, morning light filtering through trees",
                "frames": 16,
            },
            {
                "prompt": "A forest in full summer, dense greenery, sunlight dappling through the canopy",
                "frames": 16,
            },
            {
                "prompt": "A forest in autumn, vibrant fall colors, leaves beginning to carpet the ground",
                "frames": 16,
            },
            {
                "prompt": "A forest in winter, bare trees with snow-laden branches, soft winter light",
                "frames": 16,
            },
            {
                "prompt": "A forest after a wildfire, scorched trees, smoldering ashes",
                "frames": 16,
            },
            {
                "prompt": "A forest regrowing after the fire, small plants sprouting, early signs of life",
                "frames": 16,
            },
        ],
        negative_prompt="(worst quality:2), (low quality:2), (normal quality:2), ((monochrome)), ((grayscale)), bad hands",
    )
    with open("video/test2.mp4", "wb") as f:
        f.write(res.video_bytes[0])


def read_file_as_paragraph(filename: str) -> str:
    """Read the file content and return it as a single paragraph."""
    paragraph = ""
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            # Join lines into a single paragraph, adding a space between lines
            paragraph = " ".join(line.strip() for line in lines)
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    return paragraph


def save_description_and_title_to_file(content: str, filename: str) -> None:
    """Save the description and title to a file."""
    try:
        with open(filename, "w") as file:
            file.write(content)
        print(f"\nDesc and Title successfully saved to {filename}")
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")


# search_query = "Latest News on Robotics"

# article_text = web_scrapper(search_query)
# sentences = generate_summary(article_text)
# save_sentences_to_file(sentences, "output/script.txt")

# print(f"Summarized version:\n{sentences}\n")

# paragraph = read_file_as_paragraph("output/script.txt")
# print(paragraph)

# desc_and_title = generate_youtube_description_and_title(paragraph)
# save_description_and_title_to_file(desc_and_title, "output/description_and_title.txt")


# for idx, sentence in enumerate(sentences):
#     photo_description = generate_photo_prompt(sentence)
#     print(f"Photo Description for photo {idx}:\n{photo_description}\n")
#     generate_photo(photo_description, idx)
#     generate_voice(sentence, idx)
#     amplify_audio(idx)
#     print(f"Photo #{idx} done\n")


def add_centered_subtitles(
    input_clip,
    text,
    duration,
    position=("center", "bottom"),
    chunk_size=8,
    word_delay=0.8,
):
    """
    Adds centered subtitles to a video clip with groups of words appearing and fading out.

    Parameters:
    - input_clip: The video or image clip to add subtitles to.
    - text: The subtitle text.
    - duration: Duration for the full subtitle to appear.
    - position: Position of the subtitle on the screen.
    - chunk_size: Number of words per chunk.
    - word_delay: Delay between each chunk appearance.

    Returns:
    - A video clip with grouped word subtitle overlays, centered on the screen.
    """
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]
    chunk_clips = []

    # Set y-position based on desired subtitle position
    screen_width, screen_height = input_clip.size
    y_position = (
        screen_height - 100
        if position[1] == "bottom"
        else screen_height // 2 if position[1] == "center" else 100
    )

    for i, chunk in enumerate(chunks):
        chunk_start_time = i * word_delay
        chunk_clip = (
            mp.TextClip(chunk, font="Arial", fontsize=32, color="blue")
            .set_position(("center", y_position))
            .set_start(chunk_start_time)
            .set_duration(word_delay)
            .crossfadein(0.2)
            .crossfadeout(0.2)
        )
        chunk_clips.append(chunk_clip)

    # Composite the chunks into the input clip
    return mp.CompositeVideoClip([input_clip] + chunk_clips).set_duration(duration)


def slow_zoom_effect(input_clip, zoom_factor=1.5, duration=5):
    """
    Creates a slow zoom-in effect on a video clip.

    Parameters:
    - input_clip: The video or image clip to zoom.
    - zoom_factor: The amount to zoom by (e.g., 1.5 means 50% larger).
    - duration: Duration over which the zoom effect will happen.

    Returns:
    - A video clip with the slow zoom effect applied.
    """
    # Apply the zoom effect by resizing gradually
    zoomed_clip = input_clip.resize(
        lambda t: 1 + (zoom_factor - 1) * (t / duration)
    ).set_duration(duration)

    return zoomed_clip


def pan_effect(input_clip, direction="right", distance=0.2, duration=5):
    if direction == "right":
        return input_clip.set_position(
            lambda t: (int(distance * t / duration * input_clip.size[0]), "center")
        )
    elif direction == "left":
        return input_clip.set_position(
            lambda t: (-int(distance * t / duration * input_clip.size[0]), "center")
        )
    elif direction == "down":
        return input_clip.set_position(
            lambda t: ("center", int(distance * t / duration * input_clip.size[1]))
        )
    elif direction == "up":
        return input_clip.set_position(
            lambda t: ("center", -int(distance * t / duration * input_clip.size[1]))
        )


def assemble_video(quantity, search_query):
    clips = []
    for i in range(0, quantity):  # Assuming 200 sentences
        audio_clip = mp.AudioFileClip(f"audio/audio{i}.wav")
        img_clip = mp.ImageClip(f"images/photo{i}.png").set_duration(
            audio_clip.duration
        )

        img_clip = img_clip.fadein(1).fadeout(1)

        if i == 0:
            txt_clip = (
                mp.TextClip(
                    f"{search_query}",
                    font="Gill-Sans-Ultra-Bold",
                    fontsize=64,
                    color="Red",
                )
                .set_position("center")
                .set_duration(audio_clip.duration)
            )

            img_clip = img_clip.crossfadein(1).fadeout(1)

            img_clip = mp.CompositeVideoClip([img_clip, txt_clip]).set_audio(audio_clip)

        else:
            # Apply the zoom effect to the image clip
            zoomed_clip = slow_zoom_effect(
                img_clip,
                zoom_factor=round(random.uniform(1, 1.6), 1),
                duration=audio_clip.duration,
            )
            img_clip = zoomed_clip.set_audio(audio_clip)

        clips.append(img_clip)

    final_clip = mp.concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile("output/final_video.mp4", fps=20)


def process_request(search_query):
    # Web scrape, summarize, generate images and audio
    article_text, urls = web_scrapper(search_query)
    sentences = generate_summary(article_text)
    save_sentences_to_file(sentences, "output/script.txt")

    print(f"Summarized version:\n{sentences}\n")

    print(f"There are {len(sentences)} in the summarized parapragraph.\n")

    paragraph = read_file_as_paragraph("output/script.txt")
    print("Paragraph:", paragraph)

    desc_and_title = generate_youtube_description_and_title(paragraph)
    save_description_and_title_to_file(
        desc_and_title, "output/description_and_title.txt"
    )

    output_images = []
    output_video = []

    # Use actual sentences here; this is just a placeholder
    for idx, sentence in enumerate(sentences):  # Replace with actual sentences
        photo_description = generate_photo_prompt(sentence)
        print(f"\nPhoto Description for photo {idx}:\n{photo_description}\n")
        generate_photo(photo_description, idx)
        generate_voice(sentence, idx)
        amplify_audio(idx)
        print(f"Photo #{idx} done\n")

        # Create a tuple for each image: (image path, caption)
        image_path = f"images/photo{idx}.png"
        caption = f"Generated image for search query: '{search_query}'"
        output_images.append((image_path, caption))

    assemble_video(len(sentences), search_query)
    video_path = f"output/finalvideo.mp4"
    caption = f"Generated video for search query: '{search_query}'"
    output_video.append((video_path, caption))

    return output_images, paragraph, output_video, desc_and_title, urls


import gradio as gr


# Gradio Interface function
def gradio_interface(search_query):
    images, paragraph, output_video, desc_and_title, urls = process_request(
        search_query
    )  # This should return a list of tuples
    return (
        images,
        paragraph,
        output_video,
        desc_and_title,
        urls,
    )  # Return the list of tuples (image_path, caption)


# Setting up Gradio UI components
with gr.Blocks() as demo:
    gr.Markdown("# Auto Video Generator")
    search_input = gr.Textbox(
        label="Enter Search Query", placeholder="Latest News on Robotics"
    )
    submit_button = gr.Button("Generate")

    urls = gr.Textbox(label="URLs of the articles used in the video", interactive=False)

    # Textbox to display the generated paragraph
    paragraph_output = gr.Textbox(label="Generated Paragraph", interactive=False)

    desc_and_title = gr.Textbox(
        label="Generated Description and Title", interactive=False
    )

    # Gallery to display generated images
    image_output = gr.Gallery(label="Generated Images", show_label=True, scale=3)

    video_output = gr.Gallery(label="Generated Video", show_label=True, scale=3)

    # Connect the button click to the interface function
    submit_button.click(
        fn=gradio_interface,
        inputs=search_input,
        outputs=[image_output, paragraph_output, video_output, desc_and_title, urls],
    )

# Launch the Gradio app
demo.launch()
