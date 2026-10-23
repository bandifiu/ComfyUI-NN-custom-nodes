from typing_extensions import override
from comfy_api.latest import ComfyExtension

from .nodes.image import (
    ImageCASharpening, 
    ImageInsetCropPercent,
    ImagePreviewGate, 
    ImageAdjust, 
    ImageAdjustColor, 
    ImageAdjustContrast, 
    ContrastAdjustRef, 
    ImageAdjustColor, 
    ImageUpscaleBy, 
    ImageGrid,
    ImageColorRange,
    VideoTimeline,
    ImageResizeBatch,
    ImageAlphaTransparency
)
from .nodes.utils import SoftClearCache, ImageOptimize, AnySwitch, InputCheck
from .nodes.wildcards import WildcardProcessor
from .nodes.image_saver import ImageSaver, PathInfo
from .nodes.input_resolution import ImageResolution, VideoResolution, ResolutionScale
from .nodes.profiler import install_hook

WEB_DIRECTORY = "./web"

class NNCustomNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        nodes = [
            ImageCASharpening, 
            ImageInsetCropPercent,
            ImagePreviewGate, 
            ImageAdjust, 
            ImageAdjustColor, 
            ImageAdjustContrast, 
            ContrastAdjustRef, 
            ImageAdjustColor, 
            ImageUpscaleBy, 
            ImageGrid,
            ImageColorRange,
            VideoTimeline,
            ImageAlphaTransparency,

            SoftClearCache, 
            ImageOptimize, 
            AnySwitch, 
            InputCheck,

            WildcardProcessor,

            ImageSaver, PathInfo,

            ImageResolution, 
            VideoResolution, 
            ResolutionScale,
            ImageResizeBatch
        ]  

        return nodes


async def comfy_entrypoint(): 
    install_hook()
    return NNCustomNodesExtension()
