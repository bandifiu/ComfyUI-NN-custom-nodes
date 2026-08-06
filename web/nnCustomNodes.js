import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "NNCustomNodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {	
		switch(nodeData.name) {			
			case "NNVideoResolution":
			case "NNVideoResolutionV2":
			case "NNImageResolution":
			case "NNResolutionScale": {
				const onExecuted  = nodeType.prototype.onExecuted;
				nodeType.prototype.onExecuted  = function(message) {
					const r = onExecuted?.apply(this, arguments)

					const [width, height, frames] = message["text"];	

					this.outputs[0]["label"] = `Width (${width})`
					this.outputs[1]["label"] = `Height (${height})` 
					if (this.outputs[2]) {
						this.outputs[2]["label"] = `Frames (${frames})`
					}
					
					return r
				}
				break;
			}		
			case "NNAnyBoolSwitchTyped": 
			case "NNInputCheck": 
			case "NNSoftClearCache": {					
				const onConnectionsChange  = nodeType.prototype.onConnectionsChange;
				nodeType.prototype.onConnectionsChange  = function(side, slot, connect, link_info, output) {			
					const r = onConnectionsChange?.apply(this, arguments)

					this.NNUpdateType?.()
					
					return r
				}	
				break;
			}	
		}	
	}, 	
	
	async nodeCreated(node, app) {	
		const nodeType = (await node).type		
		switch(nodeType) {	
			case "NNAnyBoolSwitchTyped": 
			case "NNInputCheck": 
			case "NNSoftClearCache": {	
				
				node.NNUpdateType = () => {
					const types = [
						node.getInputDataType(0),
						node.getInputDataType(1),
					]

					const outputLink = node.outputs[0]?.links?.[0]
					if (outputLink) {
						const outType = app.graph.links[outputLink]?.type
						if (outType) {
							types.push(outType)
						}
					}

					const newType = types.find(t => !!t && t != '*') ?? '*';

					if (nodeType === "NNAnyBoolSwitchTyped") {
						node.inputs[0].type = newType
						node.inputs[0].label = `${newType}_true`
						node.inputs[1].type = newType
						node.inputs[1].label = `${newType}_false`
					} else {
						node.inputs[0].type = newType
						node.inputs[0].label = `${newType}`
					}  

					node.outputs[0].type = newType
					node.outputs[0].label = `${newType}`
				}		

				node.NNUpdateType()		
				break
			}	
		}
	}	
});
