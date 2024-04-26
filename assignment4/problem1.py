import numpy as np
import scipy.fftpack as fftpack
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt


Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

def jpeg_compress(image):
    
    height, width = image.shape
    print(height, width)
    assert height % 8 == 0 and width % 8 == 0, "Image dimensions must be divisible by 8"

    compressed_blocks = []

    # Process image in 8x8 blocks
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            
            block = image[y:y+8, x:x+8]


            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')


            quantized_block = np.round(dct_block / Q + 0.5)


            compressed_blocks.append(quantized_block)
    print(len(compressed_blocks))
    return compressed_blocks

def jpeg_decompress(compressed_blocks):
    total_pix = len(compressed_blocks) * 8 * 8#, len(compressed_blocks[0])
    height = width = int(np.sqrt(total_pix))
    decompressed_image = np.zeros((height, width))
    print(height, width)
    block_idx = 0

    # Reconstruct image from compressed blocks
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if block_idx < len(compressed_blocks):
                
                quantized_block = compressed_blocks[block_idx]


                dequantized_block = quantized_block * Q


                idct_block = fftpack.idct(fftpack.idct(dequantized_block.T, norm='ortho').T, norm='ortho')
                # print(idct_block.shape)
                
                decompressed_image[y:y+8, x:x+8] = idct_block

                block_idx += 1

    return decompressed_image

def load_image(filename):

    img = Image.open(filename)
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img)
    return img

def calculate_file_size(compressed_blocks):

    num_blocks = len(compressed_blocks)
    bits_per_coefficient = 8  # Assuming each coefficient is represented by 8 bits (can be adjusted)
    total_bits = num_blocks * 64 * bits_per_coefficient  # Each block has 64 DCT coefficients
    
    return total_bits


def calculate_mse(original_image, reconstructed_image):

    mse = np.mean((original_image - reconstructed_image) ** 2)
    return mse


def encode_quantized_indices(compressed_blocks):

    huffman_table = {
        (0,0):"0",
        (-1,1): "10",
        (-3, -2, 2, 3): "110",
        (-7, -6, -5, -4, 4, 5, 6, 7): "1110",
        (-15, -14, -13, -12, -11, -10, -9, -8, 8, 9, 10, 11, 12, 13, 14, 15): "11110",
        tuple(range(-31, -15 + 1)) + tuple(range(16, 31 + 1)): "111110",
        tuple(range(-63, -31 + 1)) + tuple(range(32, 63 + 1)): "1111110",
        tuple(range(-127, -63 + 1)) + tuple(range(64, 127 + 1)): "11111110",
        tuple(range(-255, -127 + 1)) + tuple(range(128, 255 + 1)): "111111110",
        tuple(range(-511, -255 + 1)) + tuple(range(256, 511 + 1)): "1111111110",
        tuple(range(-1023, -511 + 1)) + tuple(range(512, 1023 + 1)): "11111111110",
        tuple(range(-2047, -1023 + 1)) + tuple(range(1024, 2047 + 1)): "111111111110",
        # Add more entries for other quantized indices as needed
    }

    encoded_bitstream = ""

    for block in compressed_blocks:
        for quantized_value in block.flatten():
            found_code = False
            # print(quantized_value)
            for indices, code in huffman_table.items():
                # print(indices)
                if quantized_value in indices:
                    encoded_bitstream += code
                    found_code = True
                    break
            if not found_code:
                raise ValueError(f"No Huffman code found for quantized value: {quantized_value}")

    return encoded_bitstream



# Example usage
# Load a grayscale image (you can use any image here)
input_image = load_image('cameraman.tif')


# Compress image
compressed_blocks = jpeg_compress(input_image)

output_file_size = calculate_file_size(compressed_blocks)


# Decompress image
reconstructed_image = jpeg_decompress(compressed_blocks)

# print(len(compressed_blocks), len(reconstructed_image))
mse = calculate_mse(input_image, reconstructed_image)

compressed_pil_image = Image.fromarray(reconstructed_image.astype(np.uint8))
output_filename = "compressed_custom.jpg"
compressed_pil_image.save(output_filename, quality=80)



# Save or use the encoded bitstream as needed
encoded_bitstream = encode_quantized_indices(compressed_blocks)
print(f"Encoded bitstream length: {len(encoded_bitstream)} bits")
# Calculate the compression ratio (input image size in bits / output file size in bits)
# print(input_image.size, input_image.shape[0]*input_image.shape[1])
input_image_size_bits = input_image.size * 8  # Assume each pixel is 8 bits (grayscale)
compression_ratio = input_image_size_bits / len(encoded_bitstream)

print(input_image_size_bits, len(encoded_bitstream),input_image_size_bits/len(encoded_bitstream) )
# Print results
print(f"Output file size: {output_file_size} bits")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Compression Ratio: {compression_ratio}")









quality = 80  # Adjust quality factor (0-100)
output_filename_default = "compressed_default.jpg"
Image.fromarray(input_image).save(output_filename_default, quality=quality)

# Load and decompress the default compressed image
compressed_image_default = np.array(Image.open(output_filename_default))
reconstructed_image_default = compressed_image_default

# Calculate MSE for default compression
mse_default = calculate_mse(input_image, reconstructed_image_default)

# Compare MSE and file sizes
print(f"Custom JPEG Compression - MSE: {mse}, File Size: {os.path.getsize('compressed_custom.jpg')} bytes")
print(f"Default JPEG Compression - MSE: {mse_default}, File Size: {os.path.getsize('compressed_default.jpg')} bytes")
