import * as _elizaos_core from '@elizaos/core';
import { Service, IImageDescriptionService, ServiceType, IAgentRuntime } from '@elizaos/core';

declare class ImageDescriptionService extends Service implements IImageDescriptionService {
    static serviceType: ServiceType;
    private initialized;
    private runtime;
    private provider;
    getInstance(): IImageDescriptionService;
    initialize(runtime: IAgentRuntime): Promise<void>;
    private initializeProvider;
    private loadImageData;
    private convertImageDataToFormat;
    private fetchImage;
    describeImage(imageUrlOrPath: string): Promise<{
        title: string;
        description: string;
    }>;
}

type NodePlugin = ReturnType<typeof createNodePlugin>;
declare function createNodePlugin(): {
    readonly name: "default";
    readonly description: "Default plugin, with basic actions and evaluators";
    readonly services: [ImageDescriptionService];
    readonly actions: [_elizaos_core.Action];
};

export { type NodePlugin, createNodePlugin };
