import random
import torch
import sys
import os

# from diffusers import AutoPipelineForText2Image
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import huggingface_hub as hf_hub
from monsterapi import MClient
import gradio as gr

from utils import download_model_and_unzip

class WhisperGradio():
    def __init__(self):
        self.basemodel_path = os.environ.get("basemodel_path")
        self.loramodel_path = os.environ.get("loramodel_path", None)
        hf_token = os.environ.get("hf_token", "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")
        self.timeout_default = 200

        try:
            self.client = MClient()
        except Exception as e:
            pass

        self.__setup_models()

        if hf_token != None:
            self.__set_hf_login(hf_token)

        self.pipe = self.load_model(self.basemodel_path)
# whisper model: 'OpenAI/whisper-large-v3','OpenAI/whisper-large', 'OpenAI/whisper-base', 'OpenAI/whisper-tiny', 'OpenAI/whisper-small.en', 'OpenAI/whisper-tiny.en, OpenAI/whisper-medium', 'OpenAI/whisper-small' 'OpenAI/whisper-medium.en', 'OpenAI/whisper-large-v2', 'OpenAI/whisper-base.en', 'distil-whisper/distil-small.en', 'distil-whisper/distil-medium.en', 'distil-whisper/distil-large-v2'

    def __setup_models(self):
        if self.basemodel_path not in ['OpenAI/whisper-large-v3','OpenAI/whisper-large', 'OpenAI/whisper-base', 'OpenAI/whisper-tiny', 'OpenAI/whisper-small.en', 'OpenAI/whisper-tiny.en, OpenAI/whisper-medium', 'OpenAI/whisper-small' 'OpenAI/whisper-medium.en', 'OpenAI/whisper-large-v2', 'OpenAI/whisper-base.en', 'distil-whisper/distil-small.en', 'distil-whisper/distil-medium.en', 'distil-whisper/distil-large-v2']:
            raise ValueError(f"Invalid basemodel path: {self.basemodel_path}!")

        if self.loramodel_path == "":
            self.loramodel_path = None
        else:
            self.loramodel_path = self.loramodel_path

        if self.loramodel_path != None:
            if self.loramodel_path.startswith("https"):
                self.loramodel_path = download_model_and_unzip(self.loramodel_path)
                assert os.path.exists(self.loramodel_path)
        

    def __set_hf_login(self, hf_token):
        hf_hub.login(hf_token)

    def generate_and_display_whisper_transcriptions(self, audio_file, transcription_format, language, diarize, num_speakers, do_sample, repetition_penalty, top_p, top_k, temperature):
        """
        Transcribe audio from MonsterAPI using 'whisper' model and display transcription in Gradio interface.
        """
        try:
            response = self.client.get_response('whisper', {
                "file": audio_file,
                "transcription_format": transcription_format,
                "language": language,
                "diarize": diarize,
                "num_speakers": num_speakers,
                "do_sample": do_sample,
                "repetition_penalty": repetition_penalty,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature
            })
            output = self.client.wait_and_get_result(response['process_id'], timeout = self.timeout_default)
            if 'output' in output:
                transcription_text = output['output']
                return gr.Textbox(value=transcription_text, label="Transcription")
            else:
                return "No output available."
        except Exception as e:
            return f"Error occurred: {str(e)}"

    def load_model(self, model_id):    
        # pipe = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda") # device
        # if self.loramodel_path != None:
        #     pipe.load_lora_weights(self.loramodel_path)
        # return pipe

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to("cuda")
        return pipeline("automatic-speech-recognition", model=model, processor=processor)

    def inference(self, audio_file, transcription_format, language, diarize, num_speakers, do_sample, repetition_penalty, top_p, top_k, temperature):
        transcription = self.pipe(audio_file, transcription_format=transcription_format, language=language, diarize=diarize, num_speakers=num_speakers, do_sample=do_sample, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, temperature=temperature)
        return transcription["text"]

    def run_inference_and_display_transcriptions(self, audio_file, transcription_format, language, diarize, num_speakers, do_sample, repetition_penalty, top_p, top_k, temperature):
        # Call inference function
        transcription = self.inference(audio_file, transcription_format, language, diarize, num_speakers, do_sample, repetition_penalty, top_p, top_k, temperature)
        return transcription


def main():
    obj = WhisperGradio()
    with gr.Blocks() as demo:
        gr.HTML("<h2 style=\"font-size: 2em; font-weight: bold\" align=\"center\">Speech To Text - MonsterAPI</h2>")
        with gr.Row():
            with gr.Column():
                audio_file = gr.Audio(source="upload", type="filepath", label="Upload Audio File")
                transcription_format = gr.Dropdown(label="Transcription Format", choices=["text", "srt"], value="text")
                language = gr.Textbox(label="Language (Optional)")
                diarize = gr.Checkbox(label="Diarize", value=False)
                num_speakers = gr.Slider(label="Number of Speakers", minimum=1, maximum=5, step=1)
                do_sample = gr.Checkbox(label="Do Sample", value=True)
                repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
                top_p = gr.Slider(label="Top P", minimum=0.5, maximum=1.0, value=0.9, step=0.1)
                top_k = gr.Slider(label="Top K", minimum=10, maximum=100, value=50, step=5)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
                run = gr.Button(value="Run")
            with gr.Column():
                gr.HTML("<h3 style=\"font-size: 1.5em; font-weight: bold\" align=\"center\">Transcription Output</h3>")
                transcription_output = gr.Textbox(label="Transcription", interactive=False)

        run.click(obj.run_inference_and_display_transcriptions, inputs=[audio_file, transcription_format, language, diarize, num_speakers, do_sample, repetition_penalty, top_p, top_k, temperature], outputs=[transcription_output])
        # gr.Examples([["a photo of sky toy riding a bicycle", 1,1]], [prompt,samples], gallery_sdxl, obj.inference, cache_examples=False)

    demo.queue()
    demo.launch(server_name="0.0.0.0", share=True)
    
    
if __name__ == "__main__":
    main()