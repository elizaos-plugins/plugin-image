import type { Plugin } from "@elizaos/core";

import { describeImage } from "./actions/describe-image.ts";
import {
    ImageDescriptionService,
} from "./services/image";

export type NodePlugin = ReturnType<typeof createNodePlugin>;

export function createNodePlugin() {
    return {
        name: "default",
        description: "Default plugin, with basic actions and evaluators",
        services: [
            new ImageDescriptionService(),
        ],
        actions: [describeImage],
    } as const satisfies Plugin;
}
