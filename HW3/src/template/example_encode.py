from pathlib import Path
import cv2
import skimage.io as sio
import matplotlib.pyplot as plt

from utils import encode_mask, decode_maskobj


image_subdir = './sample-image/'
image_subdir = Path(image_subdir)

image_path = image_subdir / 'image.tif'
mask_path = image_subdir / 'class2.tif'


image = cv2.imread(str(image_path))
mask = sio.imread(mask_path)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
plt.imshow(mask)
plt.axis('off')

plt.tight_layout()
plt.show()



# Encode a mask to RLE format -- Assume you have a predicted mask

binary_mask = mask == 10
plt.figure(figsize=(4, 3))
plt.imshow(binary_mask)
plt.show()

rle_mask = encode_mask(binary_mask=binary_mask)
rle_mask


# Convert it back to binark mask

decoded_mask = decode_maskobj(rle_mask)
plt.figure(figsize=(4, 3))
plt.imshow(decoded_mask)
plt.show()