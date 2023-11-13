# weblog-of-Thesis
This is the weblog of my thesis, where I document my iterative development and progress on a weekly basis

## July
### Thesis reading 

This month, I spent a lot of time researching a model for generating images in the FishChief drawing style (FishChief is my illustration brand, which I've been running since 2017 and have accumulated hundreds of illustrations). Initially, I found information on Image-to-Image techniques, which I was familiar with from my end-of-semester project that involved converting line drawings into FishChief style fish. My original idea was that a user could upload a photo, and the model would transform it into an image in the FishChief drawing style. This is the paper I initially referred to: https://phillipi.github.io/pix2pix/. In addition to more formal papers, I also looked at some interesting online works, like a Japanese website that transformed fried chicken into paintings in the style of Van Gogh or Cezanne. During this period, my focus was primarily on Image-to-Image as a starting point.

Besides Conditional Adversarial Nets, I also looked into Cycle-Consistent Adversarial Networks, with the paper: https://junyanz.github.io/CycleGAN/. This also aligned with my initial concept of image style transformation.

However, during a meeting with my teaching assistant, they mentioned that these were relatively older techniques and that there is a newer invention called the Diffusion Model. They suggested I look into it.

After researching, I found that many popular image generation models, like DALL-E 2, Midjourney, Imagen, GLIDE, and Stable Diffusion, utilize this technology. Believing that it's crucial to stay up-to-date with the rapidly evolving field of machine learning, I decided to focus on the Diffusion Model. Since Stable Diffusion is open-source, offering a larger scope for customization and many extensions to suit personal preferences, I decided to make Stable Diffusion the main technology for my thesis.


## August
### Have rough methodology idea of the thesis -- ControlNet

This month, I focused on researching the Diffusion model and stable diffusion. When I was doing research, I found a technique called ControlNet, which fine-tunes diffusion models using a relatively small dataset. Typically, diffusion models require hundreds of millions of images for training, but ControlNet only needs tens of thousands. Although this is still a very large number for most people, compared to hundreds of millions of images, tens of thousands are relatively more achievable.

This is the GitHub link I referred to initially: https://github.com/lllyasviel/ControlNet/blob/main/README.md

It showcases the various possibilities of ControlNet. Apart from what I'm most familiar with, transforming canny edges into images, it also includes transforming certain objects (the example given is shoes) into various styles, and the use of skeletons, allowing control over the actions of characters in the generated images.

I plan to first try expanding my dataset using data augmentation, and then use it to train a version of the ControlNet fine-tuning method developed for FishChief.

Also this month, I started considering text-to-image as the main theme for my thesis. Text-to-image uses textual descriptions to generate images that match those descriptions. I'm contemplating how this approach could help FishChief in the future, using such a model to generate images and products.

## 4, Sep - 15, Sep
### Follow the practice in Github
In this GitHub repository about ControlNet (https://github.com/lllyasviel/ControlNet/blob/main/README.md), I found a section discussing how to train our own dataset (https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md). Not only does it include a link to a Google Colab Notebook, but it also provides a dataset. I think this is an excellent exercise, so I decided to follow this section to see if I could achieve some results using the provided dataset.

I've been trying to run this Google Colab Notebook for the past two weeks, but I keep encountering various problems. For example:
```Source file does not exist: /content/drive/MyDrive/CCI/230817ControlNet/fill50k/source/source/0.png```
It can't find certain image files, but when I check Google Drive, those files indeed exist;
Or ```Failed to read images: /content/drive/MyDrive/CCI/230817ControlNet/fill50k/source/12246.png, /content/drive/MyDrive/CCI/230817ControlNet/fill50k/target/12246.png```
It can't read certain images, but upon checking, they can be opened, the permissions are fine, and the path should be correct. Then, running the training cell again produces another error;
```RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx```
Even after installing the NVIDIA driver from http://www.nvidia.com/Download/index.aspx as instructed, the error still occurs.

After trying for two full weeks, I only managed to successfully get results once. During this time, I planned to try to get this notebook running and prepare a large dataset suitable for ControlNet using augmentation, while also searching for simpler ways to achieve my goal of training a model with a small dataset.


## 18, Sep - 29, Sep
### Data augmentation
In order to have more data to train ControlNet, I did data augmentation by rotating, flipping, and scaling to generate more images.
I used this [Google Colab Notebook](https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/230919data_augmentation.ipynb) to do data augmentation

In the begining I only have 158 images of marine creatures, after data augmentation, I have 158*4*6=3792 images.

<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/data%20augmentation%203972%20files.png" alt="RTX 4070" width="800"/>

After completing the data augmentation, I intend to test the ControlNet with Canny Edge. My final project for the semester involves generating FishChief-style fish images using Canny Edge, a technique I'm more familiar with. That's why I want to start with ControlNet with Canny Edge as an introductory exercise. I modified the [code](https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/230919data_augmentation.ipynb) from last semester's assignment with the help of ChatGPT, enabling it to generate the Canny Edge for each fish image.

## 2, Oct - 6, Oct
### Struggled with running ControlNet on Google Colab Notebook
ControlNet requires high VRAM to run, so I upgraded to Colab Premium. I think maybe a dataset with 5k images and 5k prompt file is too big, I was still struggled with run the notebook successfully, so I decreased the dataset to 100 images with 100 corresponding textual prompts.

<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/2%2C%20Oct.png" alt="RTX 4070" width="800"/>

Meanwhile, I kept searching for other methods meet my intension, found fine-tuning is better for people who only have small dataset. (approximatelly 10-50 images)
[Artical](https://new.qq.com/rain/a/20230403A020C800) I read introducing 4 ways to approach fine-tuning. I decided to try DreamBooth first.

## 9, Oct - 13, Oct

### Train DreamBooth of myself and Bobo my cat
According to the research I did last week, I found [this tutorial](https://www.youtube.com/watch?v=kCcXrmVk1F0&t=505s&ab_channel=MattWolfe) to train DreamBooth via Google Colab Notebook, I practiced to train portrait fo myself and Bobo, my cat, first to check the quality of the result. I used 20 images as train images, the result looks quite nice!
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/inject%20Amy.png" alt="RTX 4070" width="800"/>

### Collected dataset of my artworks
After the successful portrait model, I started to  collected images to build my drawing dataset from my previous artwork from 2015. I created a .psd file in Adobe Photoshop with file size as 512*512 pixels.

<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/FC%20dataset%20PS%20file.png" alt="RTX 4070" width="800"/>
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/73dataset.jpg" alt="RTX 4070" width="800"/>

### Train DreamBooth which present my drawing style
I first used a dataset which only included 10 images:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/DreamBooth%20dataset%2010%20images.png" alt="RTX 4070" width="800"/>

Here's the outcome:

<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/first%20DB%20outcome.png" alt="RTX 4070" width="800"/>

The outcome is very amazing! So I decided to dive into fine-tuning technoque.


### Installed Stable Diffusion Locally
If I decided to use fine-tuning technique, I would need to use Stable Diffusion since this technique is base on this. I can run Stable Diffusion on Google Colab Notebook and install it locally, I think install it locally might be more stable, so I followed [this tutorial](https://hossie.notion.site/Stable-Diffusion-MacBook-M1-M2-dda94dc6d59943ea8bc4108897642637) to install Stable Diffusion locally using terminal:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/install%20SD%20locally.png" alt="RTX 4070" width="800"/>


## 16, Oct - 20, Oct
### Train LoRA and generate suitable prompts
Last week I tried DreamBooth and found the result is quite nice! So this week I tried another way to do fine-tuning -- LoRA. I didn't choose Texture Inversion and Hypernetworks beacuse they are old techniques in terms of fine-tuning.
first I tryed many ways to generate suitable prompts for the image, prompts paired dataset in order to train LoRA model.
I followed [this tutorial](https://www.youtube.com/watch?v=fH1jf8juA8Y&t=242s&ab_channel=%E9%97%B9%E9%97%B9%E4%B8%8D%E9%97%B9), I couldn't installed the tagger extension mentioned in the video, I recorded what did I try in my diary:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/tagger%20extension.png" alt="RTX 4070" width="800"/>

Then I followed [this tutorial](https://www.youtube.com/watch?v=RgyOR5NiFMY&t=120s&ab_channel=%E6%8A%98%E9%A8%B0%E5%96%B5), tried 4 different ways using [Comparing image captioning models](https://huggingface.co/spaces/nielsr/comparing-captioning-models) online to generate prompts for this image:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/different%20prompt%20generator.png" alt="RTX 4070" width="800"/>
and also tried img2img Deep Booru in SD webui to generated prompt using same image as above, here's what Deep Booru generated:
**black hair, bubble, cat, dress, fish, multiple boys, ocean, planet, short hair, star \(sky\), starry sky, whale** 🤦🏻‍♀️🤦🏻‍♀️🤦🏻‍♀️

I would like the prompt to contain more details, so I eventually use GPT4's new function which can import image in communication box, here's the prompt generated by GPT4 with same image: **underwater, jellyfish, people, aquarium, sitting, watching, blue, marine life, glowing, bubbles, group of people, large window, ocean scene, colorful, tranquility, deep sea, sea creatures, mesmerized, stone arch, illumination, audience, relaxation, serene, aquatic** which is moch better.

So I use this method to complete my paired dataset.
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/FishChief%20LoRA%20Dataset.png" alt="RTX 4070" width="800"/>

After having a proper dataset, I trained a series of LoRA models:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/LoRA%20model%20test1.png" alt="RTX 4070" width="800"/>
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/LoRA%20model%20test2.png" alt="RTX 4070" width="800"/>
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/LoRA%20model%20test3.png" alt="RTX 4070" width="800"/>
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/LoRA%20model%20test4.png" alt="RTX 4070" width="800"/>

The outcome is not similar to my drawing style, but at least I can now train my own LoRA model.


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


## 30, Oct - 3, Nov
### Used RunPod to train model
I watched this tutorial to learned how to install Stable Diffusion on RunPod: [Tutorial]( https://www.youtube.com/watch?v=a8WESfPwlYw&ab_channel=SECourses )
Also trained a DreamBooth model on virtual environment of Stable Diffusion on RunPod. I first picked RTX 3090 as GPU to deploy tamplete, then changed to RTX 4090 since it's significantly quicker and more powerful than the RTX 3090.

I followed [this tutorial](https://www.youtube.com/watch?v=g0wXIcRhkJk&t=890s&ab_channel=SECourses) to train DreamBooth on Runpod by using dataset I built in week 9, Oct - 13, Oct, following images are images generated by those DreamBooth checkpoints:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/DB%20checkpoint%20output.png" alt="RTX 4070" width="800"/>

### Used GPT4 to generated prompts
This week, I also trained some LoRA models via the [Google Colab Notebook](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb) wtching this [tutorial](https://www.youtube.com/watch?v=oksoqMsVpaY&t=4s&ab_channel=Code%26bird).
LoRA model needs paired dataset which include images and corresponding textual prompts. I used GPT4 to help me generated the prompt then pasted the prompts into .txt file.
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/spermwhale%20dataset.png" alt="RTX 4070" width="800"/>

### Train Sperm whale LoRA and Killer whale LoRA
I collected species 's images from CC0 website such as [pixabay](https://pixabay.com/), [Pexels](https://www.pexels.com/), pasted them into a .psd file in Adobe Photoshop:
<img src="" alt="RTX 4070" width="400"/>

then use ChatGPT to help me write the code in Google Colab Notebook to export images from .psd file:
```
!pip install psd-tools
!pip install psd-tools Pillow

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import os
from psd_tools import PSDImage

# 讀取PSD檔案
psd_path = '/content/drive/MyDrive/Thesis/fishchief_style.psd'
psd = PSDImage.open(psd_path)

# 確保輸出目錄存在
output_dir = '/content/drive/MyDrive/Thesis/1102LoRA dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

canvas_width, canvas_height = 512, 512

# 導出每一個圖層
for i, layer in enumerate(psd, start=1):
    # 取得圖層的圖像
    img = layer.topil()

    # 若圖層為透明或沒有圖片，跳過此圖層
    if img is None:
        continue

    # 如果圖層帶有透明度，則將透明部分填充為白色
    if img.mode == 'RGBA':
        # 創建一個白色背景
        white_background = Image.new("RGBA", img.size, "WHITE")
        # 組合圖層與白色背景
        img = Image.alpha_composite(white_background, img)

    # 轉換合成後的圖片為RGB
    img_rgb = img.convert("RGB")

    # 創建一個512x512的白色背景圖像
    white_bg = Image.new("RGB", (canvas_width, canvas_height), "WHITE")

    # 考慮圖層的位置，確保圖層保持在原來的位置
    paste_position = (layer.left, layer.top)

    # 粘貼圖層到背景上
    white_bg.paste(img_rgb, paste_position)

    # 設定輸出路徑
    output_path = os.path.join(output_dir, f"{i}.jpg")

    # 導出圖像
    white_bg.save(output_path, 'JPEG')

    # 創建對應的空白TXT檔案
    txt_output_path = os.path.join(output_dir, f"{i}.txt")
    with open(txt_output_path, 'w') as fp:
        pass  # 'pass'意味著不做任何事情，留下一個空文件

print(f"Layers exported to {output_dir}")
```
Then I followed [this tutorial](https://www.youtube.com/watch?v=oksoqMsVpaY&t=4s&ab_channel=Code%26bird) to train LoRA for secific species.
I trained loads of LoRA and saved them in drive:
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/LoRA%20models%20in%20drive.png" alt="RTX 4070" width="800"/>

### Combine DreamBooth checkpoint with LoRA and do some experimentations
I used FishChief checkpoint as base model and combined with species LoRA, FishChief stylish LoRA to generate images. By using different parameters, optimizers, sampling steps and prompts, I generated loads of images similar to FishChief's illustration style.
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/experimentations.jpg" alt="RTX 4070" width="800"/>

<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/Final%20outcome%20of%20the%20thesis.png" alt="RTX 4070" width="800"/>

### Learned how to use img2img tag function
img2img function in stable diffusion allow users to modify, enhance, or transform an existing image based on a given textual prompt. I used __inpainted__ function in this tag to re-generate whale's tail if its not correct.
[tutorial](https://www.youtube.com/watch?v=hMvZsAaF7Gs&ab_channel=%E5%B0%8F%E9%BB%91Leo)

### Other new things I learned this week
__Hi-res__: Many images in CIVITAI have an icon saying "Hi-res", it refers to a technique used to improve the quality of the generated images, especially at higher resolutions.
(https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/6509)

__Manage LoRA models__: LoRA can only be activated when the trigger words are correct being used in the text, [lora-prompt-tool](https://github.com/a2569875/lora-prompt-tool) is a useful tool have following features: Automatic add trigger words to prompts/ Prompt search/filtering/ Editing and managing prompts/ Batch import of prompts which could be useful when having many LoRA models.








