import random
import torch
import sys
import os

from diffusers import AutoPipelineForText2Image
import huggingface_hub as hf_hub
from monsterapi import MClient
from PIL import Image
import gradio as gr

from utils import download_model_and_unzip

class SDXLLoRAGradio():
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

    def __setup_models(self):
        if self.basemodel_path not in ['stabilityai/stable-diffusion-xl-base-1.0', 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1']:
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

    def generate_and_display_ga_sdxl_base_images(self, prompt, num_samples, num_inference_steps, guidance_scale):
        """
        Generate images from MonsterAPI using 'sdxl-base' model and display them in Gradio interface.
        """
        try:
            response = self.client.get_response('sdxl-base', {
                "prompt": prompt,
                "samples": num_samples,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale        })
            output = self.client.wait_and_get_result(response['process_id'], timeout = self.timeout_default)
            if 'output' in output:
                image_url = output['output']
                return gr.Image(image_url)
            else:
                return "No output available."
        except Exception as e:
            return f"Error occurred: {str(e)}"

    def load_model(self, model_id):    
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        if self.loramodel_path != None:
            pipe.load_lora_weights(self.loramodel_path)
        return pipe

    def inference(self, prompt, num_samples,num_inference_steps,guidance_scale):
        all_images = []
        images = self.pipe(prompt, num_images_per_prompt=num_samples, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
        all_images.extend(images)
        return all_images

    def run_inference_and_display_images(self, prompt, samples, num_inference_steps, guidance_scale):
        # Call inference function
        images = self.inference(prompt, samples, num_inference_steps, guidance_scale)
        # Call generate_and_display_images function
        #output_image = self.generate_and_display_ga_sdxl_base_images(prompt, samples, num_inference_steps, guidance_scale)
        return images


def main():
    obj = SDXLLoRAGradio()
    with gr.Blocks() as demo:
        gr.HTML("<h2 style=\"font-size: 2em; font-weight: bold\" align=\"center\">Stable Diffusion Dreambooth - MonsterAPI</h2>")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="prompt")
                guidance_scale = gr.Slider(label="Guidance Scale",value=12.5,minimum=7.5,maximum=30.0,step=0.1)
                num_inference_steps = gr.Slider(label="Steps",value=50,minimum=20,maximum=60,step=1)
                samples = gr.Slider(label="Samples",value=1, minimum=1,maximum=3,step=1)
                run = gr.Button(value="Run")
            with gr.Column():
                gr.HTML("<h3 style=\"font-size: 1.5em; font-weight: bold\" align=\"center\">Finetuned Model Output</h3>")
                gallery_sdxl = gr.Gallery(show_label=False)
                # gr.HTML("<h3 style=\"font-size: 1.5em; font-weight: bold\" align=\"center\">SDXL Original</h3>")
                # gallery_monster = gr.Gallery(show_label=False)


        run.click(obj.run_inference_and_display_images, inputs=[prompt,samples,num_inference_steps, guidance_scale], outputs=[gallery_sdxl])
        gr.Examples([["a photo of sky toy riding a bicycle", 1,1]], [prompt,samples], gallery_sdxl, obj.inference, cache_examples=False)

    demo.queue()
    demo.launch(server_name="0.0.0.0",share=True)
    
    
if __name__ == "__main__":
    main()