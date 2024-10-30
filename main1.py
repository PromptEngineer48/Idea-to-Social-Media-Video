import moviepy.editor as mp
from PIL import Image
import random


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


def assemble_video():
    clips = []
    for i in range(0, 15):  # Assuming 200 sentences
        audio_clip = mp.AudioFileClip(f"audio/audio{i}.wav")
        img_clip = mp.ImageClip(f"images/photo{i}.png").set_duration(
            audio_clip.duration
        )

        img_clip = img_clip.fadein(1).fadeout(1)

        if i == 0:
            txt_clip = (
                mp.TextClip(
                    f"Latest News on Robotics",
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
    final_clip.write_videofile("output/final_video20.mp4", fps=20)


assemble_video()
