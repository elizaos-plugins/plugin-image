import type { Plugin } from "@elizaos/core";

import { describeImage } from "./actions/describe-image.ts";
import {
    ImageDescriptionService,
} from "./services/image";

const browserPlugin: Plugin = {
        name: "default",
        description: "Default plugin, with basic actions and evaluators",
        services: [
            new ImageDescriptionService(),
        ],
        actions: [describeImage],
    } 

export default browserPlugin;
