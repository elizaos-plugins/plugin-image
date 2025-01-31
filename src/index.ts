
import { describeImage } from "./actions/describe-image";
import {
    ImageDescriptionService,
} from "./services/image";

const imagePlugin = {
        name: "default",
        description: "Default plugin, with basic actions and evaluators",
        services: [
            new ImageDescriptionService() as any
        ],
        actions: [describeImage],
    } 

export default imagePlugin;
