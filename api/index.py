import io
import base64
import numpy as np
from PIL import Image
from flask_cors import CORS
from scipy.fftpack import dct, idct
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

QUALITY_TO_BLOCK_SIZE = {
    25: 8,
    50: 6,
    75: 4,
    100: 2
}

# Fungsi untuk memadatkan gambar ke dalam blok
def pad_image(img_array, block_size):
    height, width, channels = img_array.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_img = np.pad(img_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    return padded_img

# Fungsi untuk mengkuantisasi blok
def quantize(block, quality):
    quantized_block = np.zeros_like(block, dtype=np.float32)
    quantized_block[:quality, :quality] = block[:quality, :quality]
    return quantized_block

# Fungsi untuk melakukan kompresi gambar
def compress_image(img_data, quality=75):
    # Mendapatkan gambar asli dari data bytes dan mengonversinya ke mode RGB
    pre_processing = Image.open(io.BytesIO(img_data)).convert("RGB")
    image_block = np.array(pre_processing, dtype=np.float32)

    # Mendapatkan ukuran blok berdasarkan kualitas yang diberikan (default = 50)
    block_size = QUALITY_TO_BLOCK_SIZE.get(quality, 2)

    # Mendapatkan tinggi, lebar, dan jumlah saluran warna gambar
    height, width, channels = image_block.shape

    # Inisialisasi matriks untuk gambar terkompresi
    compressed_img = np.zeros((height, width, channels), dtype=np.float32)

    # Loop pertama: Iterasi melalui blok-blok dalam gambar
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Mengambil blok gambar berdasarkan ukuran blok
            block = image_block[y:y+block_size, x:x+block_size, :]

            # Inisialisasi matriks untuk blok hasil DCT
            dct_block = np.zeros_like(block)

            # Loop kedua: Iterasi melalui saluran warna (misalnya, R, G, B)
            for c in range(channels):
                # Melakukan Transformasi DCT pada setiap saluran warna blok
                dct_block[:, :, c] = dct(block[:, :, c], type=2, norm="ortho")

            # Melakukan kuantisasi pada blok DCT
            quantized_block = quantize(dct_block, quality)

            # Inisialisasi matriks untuk blok terkompresi
            compressed_block = np.zeros_like(quantized_block)

            # Loop ketiga: Iterasi melalui saluran warna (misalnya, R, G, B)
            for c in range(channels):
                # Melakukan transformasi IDCT pada setiap saluran warna blok terkompresi
                compressed_block[:, :, c] = idct(quantized_block[:, :, c], type=2, norm="ortho")

            # Menyimpan blok terkompresi dalam gambar terkompresi
            compressed_img[y:y+block_size, x:x+block_size, :] = np.clip(compressed_block, 0, 255)

    # Mengonversi gambar terkompresi ke format gambar JPEG dengan kualitas yang diberikan
    compressed_img = Image.fromarray(compressed_img.astype(np.uint8))
    img_bytes = io.BytesIO()
    compressed_img.save(img_bytes, format="JPEG", quality=quality)
    img_bytes.seek(0)

    # Mengukur ukuran gambar asli dan gambar terkompresi dalam bytes
    original_size = len(img_data)
    compressed_size = len(img_bytes.getvalue())

    # Mengembalikan data gambar terkompresi, ukuran asli, dan ukuran terkompresi
    return img_bytes.getvalue(), original_size, compressed_size

@app.route('/compress', methods=['POST'])
def compress_endpoint():
    quality = int(request.form.get('quality', 75))

    image_files = request.files.getlist('image')

    results = []

    for image_file in image_files:
        img_data = image_file.read()

        compressed_image_data, original_size, compressed_size = compress_image(
            img_data, quality=quality)

        results.append({
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compressed_image_base64": base64.b64encode(compressed_image_data).decode('utf-8')
        })

    return jsonify(results)
