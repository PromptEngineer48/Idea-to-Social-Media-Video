## Created a virtual environment called AUTOVIDEOS


import os
import requests
import time
from exa_py import Exa
from dotenv import load_dotenv
from openai import OpenAI
from typing import Union, List
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment

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

    return result_items


def generate_summary(article_text):
    # Generate a list containing individual sentences

    chat_completion_res = client.chat.completions.create(
        model=generate_summary_model,
        messages=[
            {
                "role": "system",
                "content": "Give me a youtube script in 40-50 sentences. Give me the script direclty. No requirement of saying things like here is a scripts etc. Also, add a thank you statement at the end and ask for subscription for the youtube channel. Keep the content engaging and funny.",
            },
            {
                "role": "user",
                "content": f"{article_text}",
            },
        ],
        max_tokens=1048,
    )

    results = chat_completion_res.choices[0].message.content
    sentences = sent_tokenize(results)
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
                "content": "Generate a catchy youtube title and an excellent description for my youtube video based on the script given below. Also, include lots of hashtags. Remember you are creating a youtube video title and description.",
            },
            {
                "role": "user",
                "content": f"{script}",
            },
        ],
        max_tokens=1048,
    )

    results = chat_completion_res.choices[0].message.content
    sentences = sent_tokenize(results)
    sentences = [sentence for sentence in sentences if sentence]

    return sentences


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
        print(f"Content successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


search_query = "Latest News on Robotics"

article_text = web_scrapper(search_query)
sentences = generate_summary(article_text)
save_sentences_to_file(sentences, "output/script.txt")

print(f"Summarized version:\n{sentences}\n")

paragraph = read_file_as_paragraph("output/script.txt")
print(paragraph)

desc_and_title = generate_youtube_description_and_title(paragraph)
save_description_and_title_to_file(desc_and_title, "output/description_and_title.txt")


for idx, sentence in enumerate(sentences):
    photo_description = generate_photo_prompt(sentence)
    print(f"Photo Description for photo {idx}:\n{photo_description}\n")
    generate_photo(photo_description, idx)
    generate_voice(sentence, idx)
    amplify_audio(idx)
    print(f"Photo #{idx} done\n")
