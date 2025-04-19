from PIL import Image
import numpy as np

# Load and convert image to grayscale
img = Image.open("./Data/7_a.png").convert("L")

# Resize to 27x27 just in case
img = img.resize((27, 27))

# Convert to numpy array (integer values 0-255)
pixels = np.array(img)

# Invert the grayscale values (optional)
pixels = 255 - pixels

# Flatten to 729 input values
pixels = pixels.flatten()

# Format the output as a Python array
formatted_pixels = np.array2string(pixels, separator=",", threshold=np.inf)

print(formatted_pixels)
