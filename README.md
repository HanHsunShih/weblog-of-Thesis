# weblog-of-Thesis
This is the weblog of my thesis, where I document my iterative development and progress on a weekly basis

## 23, Oct - 27, Oct
### Change to Runpod with virtual GPU
My device is Mac book Pro with M2 pro chip, I installed the Stable Diffusion locally, but when I tried to train the LoRA, I kept encountered this error:
**error:raise AssertionError("Torch not compiled with CUDA enabled")AssertionError: Torch not compiled with CUDA enabled**
Since I don't have Nvidia GPU.

## Attempt 1
I installed Mambaforge, which is specific for Apple sillicon(M1, M2), I checked if the output is **arm64** when I ran this code: 
```python -c "import torch; print(torch.cuda.is_available())" ```, the output shows **arm64**, it means pytorch is running on Apple sillicon.
But the **Torch not compiled with CUDA enabled** is still there.

## Attempt 2
Open the file: "/Users/hsun/Desktop/SDW/stable-diffusion-webui/venv/lib/python3.10/site-packages/diffusers/pipelines/pipeline_utils.py"
change line 1273, from ```model.to(device)``` to ```model.to('cpu')```, It doesn't work as well.

## Attempt 3
Fortunately there's a [cyber cafe](https://maps.app.goo.gl/wpHhDhVvc8A6ESWa8 "cyber cafe") near my home provide computer which is windows system with Nvidia GPU RTX 4070🥳
<img src="https://github.com/HanHsunShih/weblog-of-Thesis/blob/main/images/IMG_1296%20(1).jpg?raw=true" alt="RTX 4070" width="400"/>




