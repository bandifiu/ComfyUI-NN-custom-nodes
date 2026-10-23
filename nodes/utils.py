import gc
import comfy.model_management as mm
from comfy_api.latest import io

main_device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

def get_device(device):
    return main_device if device in ["gpu", 'cuda'] else offload_device 

def round_to(a, d = 16):
    return int(a // d * d)

def get_node_id(name: str) -> str:
    return f"NN{name}"

def get_node_category(group: str | None = None, sub_group: str | None = None) -> str:
    category_name = "NN"
    if group is not None:
        category_name = f"{category_name}/{group}"
    if sub_group is not None:
        category_name = f"{category_name}/{sub_group}"
        
    return category_name

class SoftClearCache(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("SoftClearCache")
        return io.Schema(
            node_id=get_node_id("SoftClearCache"),
            display_name="Soft Clear Cache",
            category=get_node_category("util"),
            inputs=[    
                io.MatchType.Input("any", template=template),
                io.Boolean.Input("unload_models", default=True),        
                io.Boolean.Input("clear_cache", default=True),            
            ],
            outputs=[
                io.MatchType.Output(display_name="any", template=template),
            ],
        )  
    @classmethod    
    def execute(cls, any, unload_models, clear_cache):        
        if unload_models:
            mm.unload_all_models()
            
        if clear_cache:
            mm.soft_empty_cache()
            
        gc.collect()        
    
        return io.NodeOutput(any)
    
class ImageOptimize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("ImageOptimize"),
            display_name="Image Optimize",
            category=get_node_category("util"),
            inputs=[    
                io.Image.Input("images"),        
                io.Boolean.Input("contiguous", default=True),     
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images, contiguous):        
        if contiguous and not images.is_contiguous():
            images = images.contiguous()
        
        return io.NodeOutput(images)
    
class InputCheck(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("InputCheck")
        return io.Schema(
            node_id=get_node_id("InputCheck"),
            display_name="InputCheck",
            category=get_node_category("util"),
            inputs=[    
                io.MatchType.Input("any", optional=True, template=template),        
            ],
            outputs=[
                io.MatchType.Output(display_name="any", template=template),
                io.Boolean.Output(display_name="is_none"),
            ],
        )  
       
    @classmethod
    def execute(self, any=None):  
        return io.NodeOutput(any, any is None) 
    
class AnySwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("AnySwitch")
        return io.Schema(
            node_id=get_node_id("AnyBoolSwitchTyped"),
            display_name="Bool Switch",
            category=get_node_category("util"),
            inputs=[    
                io.MatchType.Input("any_true", lazy=True, template=template),        
                io.MatchType.Input("any_false", lazy=True, template=template),        
                io.Boolean.Input("select", default=True),     
            ],
            outputs=[
                io.MatchType.Output(display_name="any", template=template),
            ],
        )    

    @classmethod
    def check_lazy_status(cls, any_true=None, any_false=None, select=True):  
        if select and any_true is None:
            return ["any_true"]
        
        if not select and any_false is None:
            return ["any_false"]
              
        return []
    @classmethod
    def execute(cls, any_true=None, any_false=None, select=True):
        selected = any_true if select else any_false

        return io.NodeOutput(selected)

    