import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


from skimage import io
import streamlit as st


def compress_image(image, top_k):
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(shape=image.shape)
    np.fill_diagonal(sigma, sing_values)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]
    compressed_image = trunc_U @ trunc_sigma @ trunc_V
    return compressed_image

# Загрузка изображения по URL и сжатие
def load_and_compress_image(url, top_k):
    image = io.imread(url)[:, :, 0]
    compressed_image = compress_image(image, top_k)
    return compressed_image

# Заголовок и описание
st.title('Приложение для сжатия изображений методом SVD')
st.write('Это приложение сжимает изображения с использованием сингулярного разложения (SVD).')

# Поля для ввода URL изображения и количества сингулярных чисел
url = st.text_input('Введите URL изображения:')
top_k = st.slider('Выберите количество сингулярных чисел (k):', min_value=1, max_value=400, value=20, step=1)

# Кнопка для загрузки и отображения сжатого изображения
if st.button('Сжать изображение'):
    try:
        compressed_image = load_and_compress_image(url, top_k)
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(io.imread(url), cmap='gray')
        ax[0].title.set_text('Оригинальное изображение')
        ax[1].imshow(compressed_image, cmap='gray')
        ax[1].title.set_text('Сжатое изображение (k={})'.format(top_k))
        ax[0].axis('off')
        ax[1].axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error('Error: {}'.format(e))