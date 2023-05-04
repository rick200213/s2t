import gradio as gr
import os
import subprocess
import tempfile
import whisper
from whisper.utils import write_vtt

model = whisper.load_model("medium")

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"


def translate(input_video):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert input video to mp3
        audio_file = os.path.join(tmpdir, "audio.mp3")
        subprocess.call(["ffmpeg", "-y", "-i", input_video, audio_file], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)

        # Transcribe audio using Whisper
        options = dict(beam_size=5, best_of=5, fp16=False)
        translate_options = dict(task="translate", **options)
        result = model.transcribe(audio_file, **translate_options)

        # Write transcribed text to VTT file
        subtitle = os.path.join(tmpdir, "subtitle.vtt")
        with open(subtitle, "w") as vtt:
            write_vtt(result["segments"], file=vtt)

        # Add subtitles to input video
        output_video = os.path.join(tmpdir, "output.mp4")
        subprocess.call(["ffmpeg", "-y", "-i", input_video, "-vf", f"subtitles={subtitle}", output_video], 
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)

        # Return path to the subtitled video
        return output_video


title = "Add Text/Caption to your YouTube Shorts - MultiLingual"

block = gr.Blocks()

with block:

    with gr.Group():
        with gr.Box(): 
            with gr.Row().style():
                inp_video = gr.Video(
                    label="Input Video",
                    type="filepath",
                    mirror_webcam=False
                )
                op_video = gr.Video()
            btn = gr.Button("Generate Subtitle Video")

        btn.click(translate, inputs=[inp_video], outputs=[op_video])

        gr.HTML('''
            Model by OpenAI - Gradio App by rick200213
        ''')

block.launch()
