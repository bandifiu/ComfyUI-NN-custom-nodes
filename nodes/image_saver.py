import torch
import numpy as np
import PIL.Image as Image
from PIL.PngImagePlugin import PngInfo
import json
from comfy_api.latest import UI
import folder_paths
import os
from datetime import datetime
from comfy_api.latest import io

from .utils import get_node_id, get_node_category


SUPPORTED_FORMATS = ['png', 'jpeg', 'webp']

def get_unique_filename(filepath):
    """
    Returns a unique filename by appending a number to the basename if the file already exists.
    """
    # Get the directory and filename
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Split filename and extension
    name, ext = os.path.splitext(filename)   
    
    # Check if the file already exists
    if not os.path.exists(filepath):
        return filepath
    
    # File exists, find a unique name
    counter = 1
    while True:
        new_filename = f"{name}_{counter:05d}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1

def make_filename(output_dir, filename, format, batch_index = None):
    now = datetime.now()  
    filename = filename.replace("%date", now.strftime("%Y_%m_%d"))
    filename = filename.replace("%timestamp", now.strftime("%Y-%m-%d_%H-%M-%S"))
    filename = filename.replace("%time", now.strftime("%H-%M-%S"))    

    if batch_index is not None:
        filename = f"{filename}_{batch_index:02d}.{format}"     
    else:
        filename = f"{filename}.{format}"     

    output_path = os.path.join(output_dir, filename)
    output_path = get_unique_filename(output_path)
    
    return output_path

def save_image(image, filename, format, metadata, quality, lossless):
    """
    Save torch tensor image to disk with optional metadata    
    """
    output_dir = folder_paths.output_directory
    save_filename = make_filename(output_dir, filename, format)

    # Convert torch tensor to numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    files = []
    # Handle batch dimension
    if len(image.shape) == 4:  # [b, h, w, c]
        batch_size = image.shape[0]
        for i in range(batch_size):
            single_image = image[i]
            # Generate filename with batch index
            if batch_size > 1:                
                save_filename = make_filename(output_dir, filename, format, i + 1)                
            
            path = save_single_image(single_image, save_filename, format, metadata, quality, lossless)
            files.append(path)

    else:  # [h, w, c]
        path = save_single_image(image, save_filename, format, metadata, quality, lossless)
        files.append(path)

    return files 

def save_single_image(image, filepath, format, metadata, quality=90, lossless=False):
    """
    Save a single image (now with RGBA support for PNG)
    """
    filepath = os.path.normpath(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Normalize image range
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).round().astype(np.uint8)

    # Determine image mode    
    channels = image.shape[2]

    if channels == 4:
        pil_image = Image.fromarray(image, mode='RGBA')
    else:
        pil_image = Image.fromarray(image, mode='RGB')

    fmt = format.lower()
    
    if fmt == 'png':
        pnginfo = PngInfo()
        if metadata:
            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))
        pil_image.save(filepath, format='PNG', pnginfo=pnginfo)

    elif fmt in ('jpg', 'jpeg'):
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        exif_data = pil_image.getexif()
        pil_image.save(filepath, format='JPEG', quality=quality, exif=exif_data, subsampling=0)

    elif fmt == 'webp':
        save_args = {
            "format": "WEBP",
            "quality": quality,
            "method": 6,
            "lossless": lossless
        }

        # Attach metadata (stored as EXIF in WebP)
        if metadata:
            exif = pil_image.getexif()
            if "prompt" in metadata:
                exif[0x0110] = "prompt:{}".format(metadata['prompt'])
            if "workflow" in metadata:
                exif[0x010f] = "workflow:{}".format(metadata['workflow']) 
            save_args["exif"] = exif

        pil_image.save(filepath, **save_args)

    else:
        pil_image.save(filepath, format=format.upper())    

    output_dir = folder_paths.output_directory
    return os.path.relpath(filepath, output_dir)


class ImageSaver(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageSaver"),
            display_name="Image Saver",
            category=get_node_category(),
            is_output_node=True,
            inputs=[    
                io.Image.Input("image"),        
                io.String.Input("filename", default="%date/%timestamp"),
                io.Combo.Input('format', default="webp", options=SUPPORTED_FORMATS), 
                io.Int.Input("quality", default=90, min=10, max=100, tooltip="Image quality for jpeg and webp"),        
                io.Boolean.Input("lossless", default=False, tooltip="Save as lossless webp"),     
                io.Boolean.Input("save_metadata", default=True, tooltip="Include metadata / workflow in png and webp"),                     
            ],            
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="files", is_output_list=True),
            ],
        )
    
    @classmethod
    def execute(cls, image, filename, format, quality, lossless, save_metadata):                
        metadata = None
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo

        if save_metadata:
            metadata = {}
            
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)

            if extra_pnginfo is not None and "workflow" in extra_pnginfo:
                metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])
                    
        files = save_image(image, filename, format, metadata, quality, lossless)       
        
        return io.NodeOutput(image, files, ui={"images": map(lambda file: {"filename": os.path.basename(file), "subfolder": os.path.dirname(file), "type": 'output'}, files)})

class PathInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("PathInfo"),
            display_name="Path Info",
            category=get_node_category(),
            inputs=[    
                io.String.Input("path"),        
                io.Int.Input("index", default=0, min=0),        
                io.Boolean.Input("extension", default=False),     
            ],
            outputs=[
                io.String.Output(display_name="path"),
                io.String.Output(display_name="filename"),
                io.String.Output(display_name="directory"),
            ],
        )

    @classmethod
    def execute(cls, path, index, extension):

        relpath = path
        if isinstance(path, list):
            relpath = path[index] if index < len(path) else path[-1]


        directory = os.path.dirname(relpath)

        filename = os.path.basename(relpath)
        name, ext = os.path.splitext(filename) 

        if not extension:
            relpath = os.path.join(directory, name)
            filename = name

        return io.NodeOutput(relpath, filename, directory)