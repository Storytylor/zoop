# !pip install -q diffusers transformers accelerate torch

from diffusers import DDPMPipeline
import matplotlib.pyplot as plt

model = DDPMPipeline.from_pretrained("1aurent/ddpm-mnist")

images = model(batch_size=4).images

plt.figure(figsize=(8,2))

for i,image in enumerate(images):
    plt.subplot(1,4,i+1)
    plt.imshow(image,cmap="gray")
    plt.axis("off")

plt.suptitle("Generated MNIST Images using Pre-trained DDPM")
plt.show()