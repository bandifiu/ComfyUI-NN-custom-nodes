import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from comfy_api.latest import io
import gc

from .input_resolution import aspect_ratios, get_ratio
from .utils import get_device, round_to, get_node_id, get_node_category, offload_device

def min_tensor(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(dim=0)[0]
    return torch.clamp(mn, min=0)

def max_tensor(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(dim=0)[0]
    return torch.clamp(mx, max=1)

# Ensure the tensors are in the correct format
def prepare_tensor(image):        
    if image.dim() == 3:  # If the tensor does not have a batch dimension
        image = image.unsqueeze(0)
    
    return image

def split_rgb_alpha(image):
    if image.shape[-1] == 4:
        return image[..., :3], image[..., 3:4]
    return image, None


def merge_rgb_alpha(rgb, alpha):
    if alpha is not None:
        return torch.cat([rgb, alpha], dim=-1)
    return rgb

def adjust_color_channels(img_tensor, red=0, green=0, blue=0):     
    gains = torch.tensor(
        [1 + red, 1 + green, 1 + blue],
        device=img_tensor.device,
        dtype=img_tensor.dtype
    )

    if img_tensor.shape[-1] == 4:
        rgb = img_tensor[..., :3] * gains
        alpha = img_tensor[..., 3:4]
        return torch.cat([rgb, alpha], dim=-1)

    return img_tensor * gains

def adjust_brightness(img, brightness=0):
    return img + brightness


def adjust_contrast(img, contrast=0):
    luma = (
        0.2126 * img[..., 0] +
        0.7152 * img[..., 1] +
        0.0722 * img[..., 2]
    )

    mean_luma = luma.mean(dim=(1, 2), keepdim=True)
    factor = 1 + contrast

    rgb, alpha = split_rgb_alpha(img)
    rgb = (rgb - mean_luma.unsqueeze(-1)) * factor + mean_luma.unsqueeze(-1)

    if alpha is not None:        
        return torch.cat([rgb, alpha], dim=-1)    

    return rgb

def adjust_saturation(img, saturation=1.0):
    """
    img_tensor: (B, H, W, 3), values in [0,1]
    saturation:
        1.0 = unchanged
        0.0 = grayscale
        >1.0 = increase saturation
    """

    rgb, alpha = split_rgb_alpha(img)    

    luma = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    ).unsqueeze(-1)

    rgb = luma + saturation * (rgb - luma)

    if alpha is not None:        
        return torch.cat([rgb, alpha], dim=-1)

    return rgb

def adjust_contrast_ref(images, ref_image, multiplier=1.0):
    """
    Adjust contrast to match reference image WITHOUT shifting colors.
    Contrast is computed in luminance space and applied uniformly.
    """

    image = prepare_tensor(images[0])
    ref_image = prepare_tensor(ref_image)

    image = image.clamp(0, 1)
    ref_image = ref_image.clamp(0, 1)

    # RGB → luminance weights (ITU-R BT.709)
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device)

    image_rgb = image[..., :3]
    ref_rgb = ref_image[..., :3]

    image_lum = torch.sum(image_rgb * weights, dim=-1, keepdim=True)
    ref_lum = torch.sum(ref_rgb * weights, dim=-1, keepdim=True)

    # Luminance statistics
    image_mean = image_lum.mean()
    image_std = image_lum.std()
    ref_mean = ref_lum.mean()
    ref_std = ref_lum.std()

    eps = 1e-6
    contrast_value = ref_std / (image_std + eps)
    adjusted_contrast = 1 + (contrast_value - 1) * multiplier

    # Apply contrast uniformly to all channels
    output = (images - image_mean) * adjusted_contrast + ref_mean

    return output.clamp(0, 1)

def adjust_contrast_ref_batch(images, ref_images, multiplier=1.0):
    """
    Batch-safe contrast matching using luminance.
    images:      (B, H, W, 3)
    ref_images:  (B, H, W, 3) or (H, W, 3)
    """

    images = images.clamp(0, 1)
    ref_images = ref_images.clamp(0, 1)

    # Broadcast ref if single image
    if ref_images.dim() == 3:
        ref_images = ref_images.unsqueeze(0).expand_as(images)

    weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        device=images.device
    )

    # Luminance (B, H, W, 1)
    image_lum = (images[..., :3] * weights).sum(dim=-1, keepdim=True)
    ref_lum = (ref_images[..., :3] * weights).sum(dim=-1, keepdim=True)

    # Per-image statistics
    image_mean = image_lum.mean(dim=(1, 2, 3), keepdim=True)
    image_std = image_lum.std(dim=(1, 2, 3), keepdim=True)

    ref_mean = ref_lum.mean(dim=(1, 2, 3), keepdim=True)
    ref_std = ref_lum.std(dim=(1, 2, 3), keepdim=True)

    eps = 1e-6
    contrast = ref_std / (image_std + eps)
    contrast = 1 + (contrast - 1) * multiplier

    output = (images - image_mean) * contrast + ref_mean

    return output.clamp(0, 1)


def process_chunks(image, chunk_size, device, output_shape=None, process_func=None, *func_args, **func_kwargs):
    """
    Generic chunk processing function
    """
    if process_func is None:
        return image
    
    # Pre-allocate output tensor
    if output_shape is not None:
        output_tensor = torch.empty(output_shape, device=offload_device, dtype=image.dtype)
    else:
        output_tensor = torch.empty_like(image, device=offload_device)
    
    # Process in chunks
    with torch.no_grad():
        for start in range(0, len(image), chunk_size):
            end = min(start + chunk_size, len(image))
            chunk = image[start:end]
            # Move chunk to device for processing
            chunk_gpu = chunk.to(device)

            chunk_result = process_func(chunk_gpu, *func_args, **func_kwargs)

            chunk_cpu = chunk_result.to(offload_device)

            output_tensor[start:end].copy_(chunk_cpu, non_blocking=True)
            
            gc.collect()
    
    return output_tensor

def extend_range_per_channel(img, min_vals=(16,16,16), max_vals=(235,235,235)):

    rgb, alpha = split_rgb_alpha(img)       

    min_vals_tensor = torch.tensor(
        min_vals, device=img.device, dtype=img.dtype
    )
    max_vals_tensor = torch.tensor(
        max_vals, device=img.device, dtype=img.dtype
    )

    if torch.any(min_vals_tensor >= max_vals_tensor):
        raise ValueError(
            f"All min_vals must be < max_vals. "
            f"Got min={min_vals}, max={max_vals}"
        )

    mins = min_vals_tensor / 255.0
    maxs = max_vals_tensor / 255.0

    scale = 1.0 / (maxs - mins)

    rgb = (rgb - mins) * scale
    rgb = torch.clamp(rgb, 0.0, 1.0)

    if alpha is not None:       
        return torch.cat([rgb, alpha], dim=-1)

    return rgb

def extend_range(img, min_val=16, max_val=235):    
    return extend_range_per_channel(
        img,
        min_vals=(min_val,)*3,
        max_vals=(max_val,)*3
    )

# From https://github.com/cubiq/ComfyUI_essentials/blob/main/image.py
class ImageCASharpening(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageCASharpening"),
            display_name="Image CAS harpening (Batch)",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"), 
                io.Float.Input('amount', default=1.0, min=0, max=1, step=0.05, round=0.05),
                io.Int.Input("batch_size", default=1, min=1, max=256, step=1), 
                io.Combo.Input('device', default="gpu", options=["gpu", "cpu"]),               
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )  

    @classmethod
    def execute(self, image, amount, batch_size, device):
        if not amount:
            return io.NodeOutput(image)      

        actual_device = get_device(device) 
        
        output = process_chunks(
            image=image,
            chunk_size=batch_size,
            device=actual_device,
            process_func=self.cas,
            amount=amount
        )

        return io.NodeOutput(output)  
    
    @classmethod
    def cas(cls, image, amount): 

        rgb, alpha = split_rgb_alpha(image)                   
          
        epsilon = 1e-5
        img = F.pad(rgb.permute([0, 3, 1, 2]), pad=(1, 1, 1, 1))
        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]
        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_tensor(cross)
        mx = max_tensor(cross)
        diag = (a, c, g, i)
        mn2 = min_tensor(diag)
        mx2 = max_tensor(diag)
        mx = mx + mx2
        mn = mn + mn2
        # Computing local weight
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))
        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4 * w)
        output = ((b + d + f + h) * w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)
        output = output.permute([0, 2, 3, 1])

        del a, b, c, d, e, f, g, h, i

        if alpha is not None:
            return torch.cat([output, alpha], dim=-1) 
        
        return output
    
class ImageInsetCropPercent(io.ComfyNode):
    """Image Inset Crop, with percentages."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageInsetCropPercent"),
            display_name="Image Inset Crop",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"), 
                io.Float.Input('top', default=0, min=0, max=99.9, step=0.1, round=0.1),
                io.Float.Input('left', default=0, min=0, max=99.9, step=0.1, round=0.1),
                io.Float.Input('right', default=0, min=0, max=99.9, step=0.1, round=0.1),
                io.Float.Input('bottom', default=0, min=0, max=99.9, step=0.1, round=0.1),
                io.Boolean.Input("enable", default=False), 
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )      
    
    @classmethod
    def get_new_bounds(cls, width, height, left, right, top, bottom):
        """Returns the new bounds for an image with inset crop data."""
        left = 0 + left
        right = width - right
        top = 0 + top
        bottom = height - bottom

        return (left, right, top, bottom)
  
    @classmethod
    def execute(cls, image, top, left, right, bottom, enable=True):
        if not enable:
            return io.NodeOutput(image)
                
        _, height, width, _ = image.shape
        
        rounding = 2
        left = round_to(width - (width * (100 - left) / 100), rounding)
        right = round_to(width - (width * (100 - right) / 100), rounding)
        top = round_to(height - (height * (100 - top) / 100), rounding)
        bottom = round_to(height - (height * (100 - bottom) / 100), rounding)

        inset_left, inset_right, inset_top, inset_bottom = cls.get_new_bounds(width, height, left, right,
                                                                        top, bottom)
        if inset_top > inset_bottom:
            raise ValueError(f"Invalid cropping dimensions top ({inset_top}) exceeds bottom ({inset_bottom})")
        if inset_left > inset_right:
            raise ValueError(f"Invalid cropping dimensions left ({inset_left}) exceeds right ({inset_right})")
    
        cropped_frames = image[:, inset_top:inset_bottom, inset_left:inset_right, :]

        return io.NodeOutput(cropped_frames)
    
class ImageUpscaleBy(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageUpscaleBy"),
            display_name="Image Upscale By",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"), 
                io.Float.Input('scale_by', default=1.0, min=0.01, max=16, step=0.01, round=0.01),
                io.Int.Input("batch_size", default=1, min=1, max=256, step=1), 
                io.Combo.Input('device', default="gpu", options=["gpu", "cpu"]),               
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )  
            
    @classmethod
    def scale(cls, chunk, new_width, new_height):        
        image_rgb = chunk.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
        # For Lanczos-like interpolation, use bicubic with antialiasing
        upsampled = F.interpolate(
            image_rgb, 
            size=(new_height, new_width), 
            mode='bicubic', 
            align_corners=False,
            antialias=True 
        )
    
        result = upsampled.permute(0, 2, 3, 1)  # BCHW -> BHWC

        return io.NodeOutput(result)
    

    @classmethod
    def execute(cls, image, scale_by, batch_size, device):

        if scale_by == 1:
            return io.NodeOutput(image)
                
        actual_device = get_device(device)

        length, height, width, channels = image.shape
    
        # Calculate new dimensions
        new_height = int(height * scale_by)
        new_width = int(width * scale_by)

        output_shape = (length, new_height, new_width, channels)   
        
        output = process_chunks(
            image=image,
            chunk_size=batch_size,
            device=actual_device,
            output_shape=output_shape,
            process_func=cls.scale,
            new_height=new_height,
            new_width=new_width,
        )

        return io.NodeOutput(output)
    
class ImagePreviewGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImagePreviewGate"),
            display_name="Image Preview Gate",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image", optional=True), 
                io.Int.Input("width", default=256, min=64, max=8192, step=2),
                io.Int.Input("height", default=256, min=64, max=8192, step=2),                
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )        

    @classmethod
    def create_no_preview_image(cls, width=256, height=256):
        # Create an empty black image
        image = Image.new('RGB', (width, height), color='black')
        
        # Get the drawing context
        draw = ImageDraw.Draw(image)
        
        # Define text properties
        font_size = 40
        text = "No preview"
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # You may need to specify a path to the font file
        except IOError:
            # Fallback font if 'arial.ttf' is not found
            font = ImageFont.load_default()
        
        # Get the size of the text
        text_width = draw.textlength(text, font=font)
        text_height = font_size
        
        # Calculate position to center the text
        x = (width - text_width) / 2
        y = (height - text_height) / 2
        
        # Draw the text on the image
        draw.text((x, y), text, fill='white', font=font)
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return image

    @classmethod
    def execute(cls, image=None, width=256, height=256): 
        if image is None:
            image = cls.create_no_preview_image(width, height)
        
        return io.NodeOutput(image)
    
class ImageColorRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageColorRange"),
            display_name="Image Color Range",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"),        
                io.Int.Input("min", default=0, min=0, max=127, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("max", default=255, min=128, max=255, step=1, display_mode=io.NumberDisplay.number),                
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )  

    @classmethod
    def execute(cls, image, min, max):            
        if min > 0 or max < 255:                  
            result_image = extend_range(image, min_val=min, max_val=max)

            return io.NodeOutput(result_image)
           
        return io.NodeOutput(image)
    
class ImageAdjustContrast(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageAdjustContrast"),
            display_name="Image Adjust Contrast/Brightness",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"),        
                io.Float.Input("contrast", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("brightness", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),                             
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
    
    @classmethod
    def execute(cls, image, contrast=0, brightness=0):  
        result_image = image        
        if contrast or brightness:                  
            result_image = adjust_brightness(result_image, brightness=brightness)
            result_image = adjust_contrast(result_image, contrast=contrast)
            result_image = torch.clamp(result_image, min=0, max=1.0)
        
        return io.NodeOutput(result_image)
    
class ImageAdjustColor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageAdjustColor"),
            display_name="Image Adjust Color",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"),                        
                io.Float.Input("red", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("green", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("blue", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),              
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
        
    @classmethod
    def execute(cls, image, red, green, blue):        
        result_image = image        
        if red or green or blue:   
            result_image = adjust_color_channels(result_image, red=red, green=green, blue=blue)

        result_image = torch.clamp(result_image, 0.0, 1.0)
        
        return io.NodeOutput(result_image)
    
class ImageAdjust(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageAdjust"),
            display_name="Image Adjust",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"),        
                io.Float.Input("contrast", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("brightness", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("saturation", default=1, min=-1, max=2, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("red", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("green", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),
                io.Float.Input("blue", default=0, min=-0.5, max=0.5, step=0.01, round=0.01, display_mode=io.NumberDisplay.number),              
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image, contrast, brightness, saturation, red, green, blue):        
        result_image = image        
        if brightness:   
            result_image = adjust_brightness(result_image, brightness=brightness)

        if contrast:               
            result_image = adjust_contrast(result_image, contrast=contrast)

        if saturation != 1:               
            result_image = adjust_saturation(result_image, saturation=saturation)
        
        if red or green or blue:   
            result_image = adjust_color_channels(result_image, red=red, green=green, blue=blue)

        result_image = torch.clamp(result_image, 0.0, 1.0)
       
        return io.NodeOutput(result_image)
    
class ContrastAdjustRef(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ContrastAdjustRef"),
            display_name="Contrast Adjust (Ref)",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("image"),        
                io.Image.Input("ref_image", optional=True),        
                io.Float.Input("multiplier", default=1.0, min=0, max=2, step=0.01, round=0.01, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        ) 
    
    @classmethod
    def execute(cls, image, ref_image=None, multiplier=1.0):   
        if ref_image is None:            
            return io.NodeOutput(image)   
        
        result_image = adjust_contrast_ref(image, ref_image, multiplier)
                
        return io.NodeOutput(result_image)


class ImageGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageGrid"),
            display_name="Image Grid",
            category=get_node_category("image"),
            is_input_list=True,
            inputs=[    
                io.Image.Input("images"),    
                io.Int.Input("columns", default=4, min=1, max=99),
                io.String.Input('aspect_ratio', default="1:1"),
                io.Int.Input("max_resolution", default=2048, min=64, max=16384, step=4),
            ],
            outputs=[
                io.Image.Output(display_name="grid_image"),
            ],
        )     
    
    @classmethod
    def execute(cls, images, columns, aspect_ratio, max_resolution):   
        background=0.0     
        # is_input_list=True makes every input a list
        columns = columns[0]
        aspect_ratio = aspect_ratio[0]
        max_resolution = max_resolution[0]
        all_images = []
        for img in images:
            if img.ndim != 4:
                raise ValueError("Each input must be [B, H, W, C]")
            for i in range(img.shape[0]):
                all_images.append(img[i])

        if len(all_images) == 0:
            raise ValueError("No images provided")

        # --- Parse aspect ratio ---
        target_ratio = 1
        try:
            target_ratio, *_ = get_ratio(aspect_ratio)
        except:
            first = all_images[0]
            h, w, _ = first.shape
            target_ratio = w / h


        n = len(all_images)
        rows = math.ceil(n / columns)

        c = all_images[0].shape[2]
        device = all_images[0].device
        dtype = all_images[0].dtype

        # --- Center crop to target aspect ratio ---
        cropped = []
        for img in all_images:
            h, w, _ = img.shape
            ratio = w / h

            if ratio > target_ratio:  # crop width
                new_w = int(h * target_ratio)
                x0 = (w - new_w) // 2
                img = img[:, x0:x0 + new_w]
            else:  # crop height
                new_h = int(w / target_ratio)
                y0 = (h - new_h) // 2
                img = img[y0:y0 + new_h, :]

            cropped.append(img)

        # --- Determine cell size ---
        cell_h = max(img.shape[0] for img in cropped)
        cell_w = max(img.shape[1] for img in cropped)

        grid_h = rows * cell_h
        grid_w = columns * cell_w

        scale = min(max_resolution / grid_w, max_resolution / grid_h, 1.0)
        out_h = int(grid_h * scale)
        out_w = int(grid_w * scale)

        cell_h = out_h // rows
        cell_w = out_w // columns

        # --- Create grid ---
        grid = torch.full((out_h, out_w, c), background, dtype=dtype, device=device)

        for k, img in enumerate(cropped):
            r = k // columns
            cl = k % columns

            img = img.permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(cell_h, cell_w), mode="bilinear", align_corners=False)
            img = img.squeeze(0).permute(1, 2, 0)

            y0 = r * cell_h
            x0 = cl * cell_w
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = img
       
        return io.NodeOutput(grid.unsqueeze(0))
    
class VideoTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("VideoTimeline"),
            display_name="Video Timeline",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("images"),    
                io.Int.Input("columns", default=2, min=1, max=99),
                io.Int.Input("rows", default=2, min=1, max=99),
                io.Int.Input("max_resolution", default=2048, min=64, max=8192, step=2),
            ],
            outputs=[
                io.Image.Output(display_name="timeline"),
                io.Image.Output(display_name="frames"),
            ],
        )     
    @classmethod    
    def execute(cls, images, columns, rows, max_resolution): 
        timeline, frames = cls.make_timeline(images, rows, columns, max_resolution)
        return io.NodeOutput(timeline, frames) 
        
    @classmethod  
    def make_timeline(cls, frames, rows=2, columns=2, max_size=1024, background=1.0):
        """
        Build a timeline grid image from frames.

        Args:
            frames: torch.Tensor [b, h, w, c]
            rows, cols: grid size
            max_size: max width or height of output image
            background: fill value for empty cells

        Returns:
            timeline: torch.Tensor [1, H, W, C]
            displayed_frames: list of torch.Tensor [h, w, c] frames actually used
        """
        b, h, w, c = frames.shape
        device = frames.device
        dtype = frames.dtype

        total_slots = rows * columns

        # Compute target grid size
        grid_h = rows * h
        grid_w = columns * w

        scale = min(max_size / grid_w, max_size / grid_h, 1.0)
        out_h = int(grid_h * scale)
        out_w = int(grid_w * scale)

        cell_h = out_h // rows
        cell_w = out_w // columns

        # Create output with background color
        timeline = torch.full((out_h, out_w, c), background, dtype=dtype, device=device)

        # Choose frame indices
        if b >= total_slots:
            indices = torch.linspace(0, b - 1, total_slots, device=device).long()
        else:
            indices = torch.arange(b, device=device)

        displayed_frames = []

        for k, idx in enumerate(indices):
            r = k // columns
            cl = k % columns

            img = frames[idx]                        # [h, w, c]
            displayed_frames.append(img.clone())     # store original-size frame

            img_resized = img.permute(2, 0, 1).unsqueeze(0)  # [1, c, h, w]
            img_resized = F.interpolate(img_resized, size=(cell_h, cell_w), mode="bilinear", align_corners=False)
            img_resized = img_resized.squeeze(0).permute(1, 2, 0)  # [cell_h, cell_w, c]

            y0 = r * cell_h
            x0 = cl * cell_w
            timeline[y0:y0+cell_h, x0:x0+cell_w] = img_resized

        timeline = timeline.unsqueeze(0)  # [1, H, W, C]

        return timeline, torch.stack(displayed_frames)
    
class ImageResizeBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageResizeBatch"),
            display_name="Image Resize (Batch)",
            category=get_node_category("image"),
            inputs=[    
                io.Image.Input("images"),   
                io.Int.Input("width", default=1024, min=0, max=8192, step=2),
                io.Int.Input("height", default=1024, min=0, max=8192, step=2), 
                io.Float.Input('scale_by', default=1.0, min=0.01, max=16, step=0.01),
                io.Int.Input("divisible_by", default=2, min=2, max=256, step=2),
                io.Int.Input("max_resolution", default=2048, min=64, max=8192, step=2), 
                io.Int.Input("batch_size", default=1, min=1, max=256),
                io.Combo.Input('device', default="gpu", options=["gpu", "cpu"]),  
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )     
    @classmethod    
    def execute(cls, images, width, height, scale_by, divisible_by, max_resolution, batch_size, device):

        new_width = width * scale_by
        new_height = height * scale_by

        longest_side = max(new_width, new_height)

        if longest_side > max_resolution:
            limit_scale = max_resolution / longest_side
            new_width *= limit_scale
            new_height *= limit_scale
        
        new_width = round_to(new_width, divisible_by)
        new_height = round_to(new_height, divisible_by)

        if width == new_width and height == new_height:
            return io.NodeOutput(images) 
                
        actual_device = get_device(device)

        length, _, _, channels = images.shape                 

        output_shape = (length, new_height, new_width, channels)
        
        output = process_chunks(
            image=images,
            chunk_size=batch_size,
            device=actual_device,
            output_shape=output_shape,
            process_func=cls.scale,
            new_height=new_height,
            new_width=new_width,
        )
        
        return io.NodeOutput(output)   
       

    @classmethod 
    def scale(cls, chunk, new_width, new_height):        
        image_rgb = chunk.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
        # For Lanczos-like interpolation, use bicubic with antialiasing
        upsampled = F.interpolate(
            image_rgb, 
            size=(new_height, new_width), 
            mode='bicubic', 
            align_corners=False,
            antialias=True 
        )
    
        result = upsampled.permute(0, 2, 3, 1)  # BCHW -> BHWC

        return result



    