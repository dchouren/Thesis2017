
import vision_utils as vutils


image_path = '/Users/daway/Documents/Princeton/Thesis2017/4392232665_90c251d33c.jpg'
x = vutils.load_and_preprocess_image(image_path, preprocess=False, rescale=True)

print(x.shape)