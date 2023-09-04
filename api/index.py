import io
import base64
import numpy as np
from PIL import Image
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

# Fungsi untuk memadatkan gambar ke dalam blok berukuran block_size x block_size
def pad_image(img_array, block_size):
    height, width, channels = img_array.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_img = np.pad(img_array, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    return padded_img

# Fungsi untuk mengkuantisasi blok dengan mempertahankan bagian real dari quality frekuensi pertama
def quantize(block, quality):
    quantized_block = np.zeros_like(block, dtype=np.complex128)
    quantized_block.real[:quality, :quality] = block.real[:quality, :quality]
    return quantized_block

# Fungsi untuk memetakan nilai quality menjadi ukuran blok yang sesuai dalam DCT
def switch(qty):
    if qty == 25:
        return 8
    elif qty == 50:
        return 6
    elif qty == 75:
        return 4
    
# compress_image(img_data, quality=50): Fungsi ini mengompresi gambar dengan kualitas yang diberikan (default = 50). Prosesnya adalah sebagai berikut:
# a. Memuat gambar dari data bytes menggunakan PIL dan mengonversinya ke mode RGB.
# b. Mengatur ukuran blok berdasarkan nilai quality.
# c. Memadatkan gambar ke ukuran blok yang sesuai.
# d. Melakukan transformasi DCT pada gambar dalam bentuk blok.
# e. Mengkuantisasi hasil transformasi DCT.
# f. Melakukan transformasi IDCT pada gambar terkompresi.
# g. Membatasi nilai-nilai piksel pada rentang yang valid (0-255).
# h. Mengonversi gambar hasil kompresi ke dalam format base64.

# Fungsi untuk mengompresi gambar dengan kualitas yang diberikan
def compress_image(img_data, quality=50):
    # Pre Processing
    pre_processing = Image.open(io.BytesIO(img_data)).convert("RGB")

    # Pembagian citra menjadi block block
    image_block = np.array(pre_processing, dtype=np.float32)
    
    # Pengurangan nilai rata rata
    nilai_rata_rata = switch(qty=quality)
    padded_img_array = pad_image(image_block, nilai_rata_rata)
    height, width, channels = padded_img_array.shape
    compressed_img = np.zeros((height, width, channels), dtype=np.float32)

    # Transformasi Discrete Cosine
    for y in range(0, height, nilai_rata_rata):
        for x in range(0, width, nilai_rata_rata):
            block = padded_img_array[y:y+nilai_rata_rata, x:x+nilai_rata_rata, :]

            # Melakukan transformasi DCT pada setiap blok
            dct_block = np.fft.fft2(block, axes=(0, 1), norm="ortho")

            # Mengkuantisasi blok hasil DCT
            quantized_block = quantize(dct_block, quality=50)

            # Menyimpan bagian real hasil kuantisasi ke dalam gambar terkompresi
            compressed_img[y:y+nilai_rata_rata, x:x+nilai_rata_rata, :] = quantized_block.real

    # Kuantisasi Invers
    decompressed_img = np.zeros((height, width, channels), dtype=np.float32)

    for y in range(0, height, nilai_rata_rata):
        for x in range(0, width, nilai_rata_rata):
            quantized_block = compressed_img[y:y+nilai_rata_rata, x:x+nilai_rata_rata, :]

            # Melakukan transformasi IDCT pada setiap blok terkompresi
            idct_block = np.fft.ifft2(quantized_block, axes=(0, 1), norm="ortho").real
            decompressed_img[y:y+nilai_rata_rata, x:x+nilai_rata_rata, :] = idct_block

    # Rekonstruksi gambar
    decompressed_img = np.clip(decompressed_img, 0, 255)

    # Encoding
    compressed_img = Image.fromarray(decompressed_img.astype(np.uint8))
    img_bytes = io.BytesIO()
    compressed_img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    compressed_image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    return compressed_image_base64

# Fungsi compress_endpoint(): Route ("/compress") yang menerima POST request untuk mengompresi gambar. Langkah-langkahnya adalah sebagai berikut:
# a. Mendapatkan nilai quality dari data form.
# b. Menerima daftar file gambar dari data form.
# c. Untuk setiap file gambar, membaca data gambar, mengukur ukuran gambar asli dalam bytes.
# d. Melakukan kompresi pada gambar menggunakan fungsi compress_image() dengan kualitas yang diberikan.
# e. Mengukur ukuran gambar hasil kompresi dalam bytes.
# f. Menyimpan hasil kompresi dalam format base64 bersama dengan informasi ukuran asli dan ukuran terkompresi.
# g. Mengembalikan hasil dalam format JSON yang berisi informasi ukuran gambar asli, ukuran gambar terkompresi, dan data gambar terkompresi dalam format base64.

# Route untuk endpoint kompresi
@app.route('/compress', methods=['POST'])
def compress_endpoint():
    # Mendapatkan nilai quality dari data form
    quality = int(request.form.get('quality'))

    # Mendapatkan daftar file gambar dari data form
    image_files = request.files.getlist('image')

    # List untuk menyimpan hasil kompresi setiap gambar
    results = []

    # Untuk setiap file gambar dalam daftar, lakukan kompresi
    for image_file in image_files:
        # Mendapatkan data gambar dari data form
        img_data = image_file.read()

        # Mendapatkan ukuran gambar asli dalam bytes
        original_size = len(img_data)

        # Langkah 10: Melakukan kompresi pada gambar
        compressed_image_base64 = compress_image(img_data, quality=quality)

        # Mendapatkan ukuran gambar terkompresi dalam bytes
        compressed_size = len(base64.b64decode(compressed_image_base64))

        # Langkah 11: Menambahkan hasil kompresi untuk gambar ini ke dalam list hasil
        results.append({
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compressed_image_base64": compressed_image_base64
        })

    # Mengembalikan hasil dalam format JSON yang berisi informasi ukuran gambar asli, ukuran gambar terkompresi, dan data gambar terkompresi dalam format base64.
    return jsonify(results)
