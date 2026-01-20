from PIL import Image
from PIL import ImageOps;
from torch.utils.data import Dataset
import os

def mnistify(pil_img):
    """Convert a PIL image to MNIST format: 28x28 grayscale, inverted colors, autocontrasted.
    Arguments:
        pil_img: PIL.Image - input image
        output_path: str or None - if given, save the processed image to this path
    Returns:
        PIL.Image - processed image
    """
    if pil_img.mode == "RGBA":
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3]) # paste without alpha channel
        img = background
    else:
        img = pil_img
    img = img.convert("L");
    img = ImageOps.autocontrast(img, cutoff=1);
    img = img.point(lambda x: 255 if x>210 else x);
    img = ImageOps.invert(img);
    bbox = img.getbbox();
    if bbox:
        img = img.crop(bbox);
    img = img.resize((28, 28), Image.Resampling.LANCZOS);
    img = ImageOps.autocontrast(img, cutoff=1);

    return img;

if __name__ == "__main__":
    # Example usage
    img = Image.open("./data/myFashion/7.1.png")
    processed_img = mnistify(img)
    processed_img.show();


class CustomDataset(Dataset):
    """A custom dataset for loading images from a directory, processing them to MNIST format,
    and providing labels based on filenames.
    Assumes that images are named with their label as the first character (e.g., '3_image1.png' for label 3).
    Arguments:
        path: str - directory containing images
        transform: callable or None - optional transform to apply to images
        img_process: callable - function to process images (default: mnistify)
    """
    def __init__(self, path, transform=None, img_process=mnistify):
        self.path = path
        self.img_process = img_process
        self.transform = transform
        self.images = [f for f in os.listdir(path) if f.endswith(".png")]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        label = int(img_name[0]) # Assuming the label is the first digit of the filename
        image = Image.open(os.path.join(self.path, img_name))
        image = self.img_process(image)
        if self.transform:
            image = self.transform(image)
        
        return image, label

