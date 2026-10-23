import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "NNCustomNodes",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    switch (nodeData.name) {
      case "NNVideoResolution":
      case "NNVideoResolutionV2":
      case "NNImageResolution":
      case "NNResolutionScale": {
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const r = onExecuted?.apply(this, arguments);

          const [width, height, frames] = message["text"];

          this.outputs[0]["label"] = `Width (${width})`;
          this.outputs[1]["label"] = `Height (${height})`;
          if (this.outputs[2]) {
            this.outputs[2]["label"] = `Frames (${frames})`;
          }

          return r;
        };
        break;
      }
    }
  },
});
