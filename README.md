# weblog-of-Thesis
This is the weblog of my thesis, where I document my iterative development and progress on a weekly basis

## 23, Oct - 27, Oct
### Used RunPod to train model
I watched this tutorial to learned how to install Stable Diffusion on RunPod: [Tutorial]( https://www.youtube.com/watch?v=a8WESfPwlYw&ab_channel=SECourses )
Also trained a DreamBooth model. I first picked RTX 3090 as GPU to deploy tamplete, then changed to RTX 4090 since it's significantly quicker and more powerful than the RTX 3090.

### used GPT4 to generated prompts
This week, I also trained some LoRA models via the [Google Colab Notebook](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb) wtching this [tutorial](https://www.youtube.com/watch?v=oksoqMsVpaY&t=4s&ab_channel=Code%26bird).
LoRA model needs paired dataset which include images and corresponding textual prompts. I used GPT4 to help me generated the prompt then pasted the prompts into .txt file.
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/Screenshot%202023-11-13%20at%2014.23.11.png" alt="RTX 4070" width="400"/>



## 23, Oct - 27, Oct
### Switching to RunPod with Virtual GPU
I have a MacBook Pro with an M2 Pro chip. Although I successfully installed Stable Diffusion locally, I encountered persistent issues while attempting to train the LoRA due to the lack of an Nvidia GPU:
**error:raise AssertionError("Torch not compiled with CUDA enabled")AssertionError: Torch not compiled with CUDA enabled**

### Attempt 1
I installed Mambaforge, which is designed specifically for Apple Silicon (M1, M2 chips). To verify the installation, I ran  
```python -c "import torch; print(torch.cuda.is_available())" ```, and the output was **arm64**, indicating that PyTorch is running on Apple Silicon. However, I still faced the Torch not compiled with CUDA enabled error.

### Attempt 2
I tried editing the file at "/Users/hsun/Desktop/SDW/stable-diffusion-webui/venv/lib/python3.10/site-packages/diffusers/pipelines/pipeline_utils.py"
changing line 1273 from ```model.to(device)``` to ```model.to('cpu')```. Unfortunately, this didn't resolve the issue.

### Attempt 3
Luckily, there's a [cyber cafe](https://maps.app.goo.gl/wpHhDhVvc8A6ESWa8 "cyber cafe") near my home that provides computers with Windows systems and Nvidia RTX 4070 GPUs 🥳.
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/IMG_1296%20(1).jpg?raw=true" alt="RTX 4070" width="400"/>

I followed a tutorial to install Stable Diffusion: (https://www.youtube.com/watch?v=onmqbI5XPH8 )
and everything ran very smoothly and quickly. However, I was not allowed to install **xformer**, a library designed for Transformer neural network architectures, due to an **Access Denied** error.

Consequently, I decided to use RunPod, a service where I can rent remote virtual machines with high-performance GPUs suitable for compute-intensive tasks such as machine learning and deep learning.
