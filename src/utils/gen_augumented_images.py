import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T


def random_augment(img):
    """Apply random augmentation with small/random parameters."""
    
    # 1. Random resize / crop
    if random.random() < 0.3:
        scale = (random.uniform(0.9, 1.0), 1.0)
        img = T.RandomResizedCrop(size=224, scale=scale)(img)
    
    # 2. Random flip 
    if random.random() < 0.2:
        img = T.RandomHorizontalFlip(p=1.0)(img)
    if random.random() < 0.1:
        img = T.RandomVerticalFlip(p=1.0)(img)
    
    # 3. Random rotation 
    if random.random() < 0.3:
        angle = random.randint(-10, 10)
        img = T.functional.rotate(img, angle)
    
    # 4. Random color jitter 
    if random.random() < 0.5:
        brightness = random.uniform(0.9, 1.1)
        contrast   = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        hue_range  = random.uniform(0.0, 0.05)
        img = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=(-hue_range, hue_range)
        )(img)
    
    # 5. Random blur/sharpen
    if random.random() < 0.2:
        radius = random.uniform(0.1, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif random.random() < 0.1:
        img = img.filter(ImageFilter.SHARPEN)
    
    # 6. Random brightness/contrast
    if random.random() < 0.2:
        factor = random.uniform(0.9, 1.1)
        img = ImageEnhance.Brightness(img).enhance(factor)
    if random.random() < 0.2:
        factor = random.uniform(0.9, 1.1)
        img = ImageEnhance.Contrast(img).enhance(factor)
    
    return img



def augment_folder(folder_path, save_path=None, num_augments=5):
    """
    Load all images in folder and create random augmented versions with random params.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    results = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert("RGB")
            
            augmented_imgs = []
            for i in range(num_augments):
                aug_img = random_augment(img.copy())
                augmented_imgs.append(aug_img)
                
                if save_path:
                    out_name = f"{os.path.splitext(fname)[0]}_aug{i}.jpg"
                    aug_img.save(os.path.join(save_path, out_name))
            
            results[fname] = augmented_imgs
    
    return results


# Example usage
if __name__ == "__main__":
    folder = "src/sample_images_gen/input_images"
    out_folder = "data/raw"
    augmented = augment_folder(folder, save_path=out_folder, num_augments=20)
    print(f"Done! Augmented {len(augmented)} images.")
