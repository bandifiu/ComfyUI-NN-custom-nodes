import math
from comfy_api.latest import io
from .utils import get_node_id, get_node_category, round_to

aspect_ratios = ["auto", "1:1", "9:7", "5:4", "4:3", "3:2", "16:9", "2:1", "21:9"]
frame_rates = [16, 24, 30]

def get_ratio(aspect_ratio: str) -> tuple[float, int, int]:
    if aspect_ratio == "auto":
        return 1.0, 1, 1
    
    w, h = map(int, aspect_ratio.split(":"))
    ratio = w / h
    return ratio, w, h

def get_resolution(width: int, height: int, aspect_ratio: str, portrait: bool, divisible_by: int) -> tuple[int, int]:
    if width and height:
        if height == -1:
                height = width
        if aspect_ratio != "auto":            
            _, w_ratio, h_ratio = get_ratio(aspect_ratio)
            total_pixels = width * height        
            width = math.sqrt(total_pixels * w_ratio / h_ratio)
            height = math.sqrt(total_pixels * h_ratio / w_ratio) 
    
    elif width and not height:            
        ratio, *_ = get_ratio(aspect_ratio)           
        if portrait:
            height = width * ratio
        else: 
            height = width / ratio

    elif height and not width:
        ratio, *_ = get_ratio(aspect_ratio)
        if portrait:
            width = height / ratio
        else: 
            width = height * ratio

    else:
        DEFAULT_RES = 1024
        _, w_ratio, h_ratio = get_ratio(aspect_ratio)
        total_pixels = DEFAULT_RES * DEFAULT_RES                    
        width = math.sqrt(total_pixels * w_ratio / h_ratio)
        height = math.sqrt(total_pixels * h_ratio / w_ratio)

    width = round_to(width, divisible_by)
    height = round_to(height, divisible_by)    
    
    if portrait:
        if width > height:
            width, height = height, width
    else:
        if width < height:
            width, height = height, width

    return width, height

class ImageResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageResolution"),
            display_name="Image Resolution",
            category=get_node_category(),
            inputs=[    
                io.Int.Input("width", default=1024, min=0, max=8192, step=1),
                io.Int.Input("height", default=-1, min=-1, max=8192, step=1),  
                io.Combo.Input('aspect_ratio', options=aspect_ratios),
                io.Int.Input("divisible_by", default=2, min=2, max=256, step=2),
                io.Boolean.Input("portrait", default=False),             
            ],
            outputs=[
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
            ],
        )  

    @classmethod
    def execute(cls, width, height, aspect_ratio="auto", divisible_by=16, portrait=False):        
        width, height = get_resolution(width, height, aspect_ratio, portrait, divisible_by)

        return io.NodeOutput(width, height, ui={"text": (width, height)})
        
    
class VideoResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("VideoResolution"),
            display_name="Video Resolution",
            category=get_node_category(),
            inputs=[    
                io.Int.Input("width", default=1024, min=0, max=8192, step=1),
                io.Int.Input("height", default=-1, min=-1, max=8192, step=1),  
                io.Combo.Input('aspect_ratio', options=aspect_ratios),
                io.Float.Input('length', default=5.0, min=0, max=120, step=0.1, tooltip="Length in seconds"),
                io.Combo.Input('frame_rate', options=frame_rates),
                io.Int.Input("divisible_by", default=2, min=2, max=256, step=2),
                io.Boolean.Input("portrait", default=False),             
            ],
            outputs=[
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
                io.Int.Output(display_name="Frames"),
            ],
        ) 

    @classmethod
    def execute(cls, width, height, aspect_ratio, length=5.0, frame_rate=16, divisible_by=16, portrait=False): 

        width, height = get_resolution(width, height, aspect_ratio, portrait, divisible_by)
        frames = round_to(length * frame_rate, 4) + 1          

        return io.NodeOutput(width, height, frames, ui={"text": (width, height, frames)})

    
class ResolutionScale(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ResolutionScale"),
            display_name="Resolution Scale",
            category=get_node_category(),
            inputs=[    
                io.Int.Input("width", default=1024, min=64, max=8192, step=2),
                io.Int.Input("height", default=1024, min=64, max=8192, step=2), 
                io.Float.Input('scale_by', default=1.0, min=0.01, max=16, step=0.01),
                io.Int.Input("divisible_by", default=2, min=2, max=256, step=2),
                io.Int.Input("max_resolution", default=2048, min=64, max=8192, step=2),
            ],
            outputs=[
                io.Int.Output(display_name="Width"),
                io.Int.Output(display_name="Height"),
            ],
        )  
    @classmethod
    def execute(cls, width, height, scale_by, divisible_by, max_resolution): 

        new_width = width * scale_by
        new_height = height * scale_by

        longest_side = max(new_width, new_height)


        if longest_side > max_resolution:
            limit_scale = max_resolution / longest_side
            new_width *= limit_scale
            new_height *= limit_scale
        
        new_width = round_to(new_width, divisible_by)
        new_height = round_to(new_height, divisible_by)
                
        return io.NodeOutput(new_width, new_height, ui={"text": (new_width, new_height)})
       