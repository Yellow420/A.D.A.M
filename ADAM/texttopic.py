#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import torch
import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
from PIL import Image
from torchvision import transforms

def text_to_pic(text):
    # Load pre-trained BigGAN model
    biggan_model = torch.hub.load('huggingface/pytorch-pretrained-BigGAN', 'BigGAN-deep-128')

    # Generate image from text using BigGAN
    with torch.no_grad():
        noise_vector = torch.randn(1, 128)
        text_embedding = biggan_model.module.embeddings(torch.LongTensor([biggan_model.module.vocab[text]]))
        output = biggan_model.module.decoder([noise_vector, text_embedding], 0.7)

    generated_image = transforms.ToPILImage()(output[0]/2.0 + 0.5)  # Convert to PIL Image

    # Convert PIL Image to NumPy array for DeepDream
    image_array = np.array(generated_image)

    # Load pre-trained InceptionV3 model for DeepDream
    inception_model = tfhub.load('https://tfhub.dev/google/deepdream/inception_v3/1')

    # Perform DeepDream on the generated image
    dream_img = tf.image.resize(np.expand_dims(image_array, axis=0), (224, 224))
    dream_img = inception_model(dream_img)['mixed3']

    # Save the DeepDream image
    dream_img = tf.squeeze(dream_img)
    dream_img = tf.image.resize(dream_img, (generated_image.size[1], generated_image.size[0]))
    image = Image.fromarray(np.array(dream_img))

  

    return image