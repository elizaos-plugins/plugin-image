// src/actions/describe-image.ts
import {
  composeContext,
  generateObject,
  ModelClass,
  elizaLogger,
  ServiceType
} from "@elizaos/core";

// src/templates.ts
var getFileLocationTemplate = `
{{recentMessages}}

extract the file location from the users message or the attachment in the message history that they are referring to.
your job is to infer the correct attachment based on the recent messages, the users most recent message, and the attachments in the message
image attachments are the result of the users uploads, or images you have created.
only respond with the file location, no other text.
typically the file location is in the form of a URL or a file path.

\`\`\`json
{
    "fileLocation": "file location text goes here"
}
\`\`\`
`;

// src/types.ts
import { z } from "zod";
var FileLocationResultSchema = z.object({
  fileLocation: z.string().min(1)
});
function isFileLocationResult(obj) {
  return FileLocationResultSchema.safeParse(obj).success;
}

// src/actions/describe-image.ts
var describeImage = {
  name: "DESCRIBE_IMAGE",
  similes: ["DESCRIBE_PICTURE", "EXPLAIN_PICTURE", "EXPLAIN_IMAGE"],
  validate: async (_runtime, _message) => {
    return true;
  },
  description: "Describe an image",
  handler: async (runtime, message, state, _options, callback) => {
    const getFileLocationContext = composeContext({
      state,
      template: getFileLocationTemplate
    });
    const fileLocationResultObject = await generateObject({
      runtime,
      context: getFileLocationContext,
      modelClass: ModelClass.SMALL,
      schema: FileLocationResultSchema,
      stop: ["\n"]
    });
    if (!isFileLocationResult(
      fileLocationResultObject?.object ?? fileLocationResultObject
    )) {
      elizaLogger.error("Failed to generate file location");
      return false;
    }
    let fileLocation = fileLocationResultObject?.object?.fileLocation;
    fileLocation ??= fileLocationResultObject;
    const { description } = await runtime.getService(ServiceType.IMAGE_DESCRIPTION).describeImage(fileLocation);
    runtime.messageManager.createMemory({
      userId: message.agentId,
      agentId: message.agentId,
      roomId: message.roomId,
      content: {
        text: description
      }
    });
    callback({
      text: description
    });
    return true;
  },
  examples: [
    [
      {
        user: "{{user1}}",
        content: {
          text: "Can you describe this image for me?"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "Let me analyze this image for you...",
          action: "DESCRIBE_IMAGE"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "I see an orange tabby cat sitting on a windowsill. The cat appears to be relaxed and looking out the window at birds flying by. The lighting suggests it's a sunny afternoon."
        }
      }
    ],
    [
      {
        user: "{{user1}}",
        content: {
          text: "What's in this picture?"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "I'll take a look at that image...",
          action: "DESCRIBE_IMAGE"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "The image shows a modern kitchen with stainless steel appliances. There's a large island counter in the center with marble countertops. The cabinets are white with sleek handles, and there's pendant lighting hanging above the island."
        }
      }
    ],
    [
      {
        user: "{{user1}}",
        content: {
          text: "Could you tell me what this image depicts?"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "I'll describe this image for you...",
          action: "DESCRIBE_IMAGE"
        }
      },
      {
        user: "{{user2}}",
        content: {
          text: "This is a scenic mountain landscape at sunset. The peaks are snow-capped and reflected in a calm lake below. The sky is painted in vibrant oranges and purples, with a few wispy clouds catching the last rays of sunlight."
        }
      }
    ]
  ]
};

// src/services/image.ts
import {
  elizaLogger as elizaLogger2,
  getEndpoint,
  ModelProviderName,
  models,
  Service,
  ServiceType as ServiceType2
} from "@elizaos/core";
import {
  AutoProcessor,
  AutoTokenizer,
  env,
  Florence2ForConditionalGeneration,
  RawImage
} from "@huggingface/transformers";
import sharp from "sharp";
import fs from "fs";
import os from "os";
import path from "path";
var IMAGE_DESCRIPTION_PROMPT = "Describe this image and give it a title. The first line should be the title, and then a line break, then a detailed description of the image. Respond with the format 'title\\ndescription'";
var convertToBase64DataUrl = (imageData, mimeType) => {
  const base64Data = imageData.toString("base64");
  return `data:${mimeType};base64,${base64Data}`;
};
var handleApiError = async (response, provider) => {
  const responseText = await response.text();
  elizaLogger2.error(
    `${provider} API error:`,
    response.status,
    "-",
    responseText
  );
  throw new Error(`HTTP error! status: ${response.status}`);
};
var parseImageResponse = (text) => {
  const [title, ...descriptionParts] = text.split("\n");
  return { title, description: descriptionParts.join("\n") };
};
var LocalImageProvider = class {
  model = null;
  processor = null;
  tokenizer = null;
  modelId = "onnx-community/Florence-2-base-ft";
  async initialize() {
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.backends.onnx.logLevel = "fatal";
    env.backends.onnx.wasm.proxy = false;
    env.backends.onnx.wasm.numThreads = 1;
    elizaLogger2.info("Downloading Florence model...");
    this.model = await Florence2ForConditionalGeneration.from_pretrained(
      this.modelId,
      {
        device: "gpu",
        progress_callback: (progress) => {
          if (progress.status === "downloading") {
            const percent = (progress.loaded / progress.total * 100).toFixed(1);
            const dots = ".".repeat(
              Math.floor(Number(percent) / 5)
            );
            elizaLogger2.info(
              `Downloading Florence model: [${dots.padEnd(20, " ")}] ${percent}%`
            );
          }
        }
      }
    );
    elizaLogger2.info("Downloading processor...");
    this.processor = await AutoProcessor.from_pretrained(
      this.modelId
    );
    elizaLogger2.info("Downloading tokenizer...");
    this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
    elizaLogger2.success("Image service initialization complete");
  }
  async describeImage(imageData, mimeType) {
    if (!this.model || !this.processor || !this.tokenizer) {
      throw new Error("Model components not initialized");
    }
    const blob = new Blob([imageData], { type: mimeType });
    const image = await RawImage.fromBlob(blob);
    const visionInputs = await this.processor(image);
    const prompts = this.processor.construct_prompts("<DETAILED_CAPTION>");
    const textInputs = this.tokenizer(prompts);
    elizaLogger2.log("Generating image description");
    const generatedIds = await this.model.generate({
      ...textInputs,
      ...visionInputs,
      max_new_tokens: 256
    });
    const generatedText = this.tokenizer.batch_decode(generatedIds, {
      skip_special_tokens: false
    })[0];
    const result = this.processor.post_process_generation(
      generatedText,
      "<DETAILED_CAPTION>",
      image.size
    );
    const detailedCaption = result["<DETAILED_CAPTION>"];
    return { title: detailedCaption, description: detailedCaption };
  }
};
var AnthropicImageProvider = class {
  constructor(runtime) {
    this.runtime = runtime;
  }
  async initialize() {
  }
  async describeImage(imageData, mimeType) {
    const endpoint = getEndpoint(ModelProviderName.ANTHROPIC);
    const apiKey = this.runtime.getSetting("ANTHROPIC_API_KEY");
    const content = [
      { type: "text", text: IMAGE_DESCRIPTION_PROMPT },
      {
        type: "image",
        source: {
          type: "base64",
          media_type: mimeType,
          data: imageData.toString("base64")
        }
      }
    ];
    const response = await fetch(`${endpoint}/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01"
      },
      body: JSON.stringify(
        {
          model: "claude-3-haiku-20240307",
          max_tokens: 1024,
          messages: [{ role: "user", content }]
        }
      )
    });
    if (!response.ok) {
      await handleApiError(response, "Anthropic");
    }
    const data = await response.json();
    return parseImageResponse(data.content[0].text);
  }
};
var OpenAIImageProvider = class {
  constructor(runtime) {
    this.runtime = runtime;
  }
  async initialize() {
  }
  async describeImage(imageData, mimeType) {
    const imageUrl = convertToBase64DataUrl(imageData, mimeType);
    const content = [
      { type: "text", text: IMAGE_DESCRIPTION_PROMPT },
      { type: "image_url", image_url: { url: imageUrl } }
    ];
    const endpoint = this.runtime.imageVisionModelProvider === ModelProviderName.OPENAI ? getEndpoint(this.runtime.imageVisionModelProvider) : "https://api.openai.com/v1";
    const response = await fetch(endpoint + "/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.runtime.getSetting("OPENAI_API_KEY")}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [{ role: "user", content }],
        max_tokens: 500
      })
    });
    if (!response.ok) {
      await handleApiError(response, "OpenAI");
    }
    const data = await response.json();
    return parseImageResponse(data.choices[0].message.content);
  }
};
var GroqImageProvider = class {
  constructor(runtime) {
    this.runtime = runtime;
  }
  async initialize() {
  }
  async describeImage(imageData, mimeType) {
    const imageUrl = convertToBase64DataUrl(imageData, mimeType);
    const content = [
      { type: "text", text: IMAGE_DESCRIPTION_PROMPT },
      { type: "image_url", image_url: { url: imageUrl } }
    ];
    const endpoint = this.runtime.imageVisionModelProvider === ModelProviderName.GROQ ? getEndpoint(this.runtime.imageVisionModelProvider) : "https://api.groq.com/openai/v1/";
    const response = await fetch(endpoint + "/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.runtime.getSetting("GROQ_API_KEY")}`
      },
      body: JSON.stringify({
        model: (
          /*this.runtime.imageVisionModelName ||*/
          "llama-3.2-90b-vision-preview"
        ),
        messages: [{ role: "user", content }],
        max_tokens: 1024
      })
    });
    if (!response.ok) {
      await handleApiError(response, "Groq");
    }
    const data = await response.json();
    return parseImageResponse(data.choices[0].message.content);
  }
};
var GoogleImageProvider = class {
  constructor(runtime) {
    this.runtime = runtime;
  }
  async initialize() {
  }
  async describeImage(imageData, mimeType) {
    const endpoint = getEndpoint(ModelProviderName.GOOGLE);
    const apiKey = this.runtime.getSetting("GOOGLE_GENERATIVE_AI_API_KEY");
    const response = await fetch(
      `${endpoint}/v1/models/gemini-1.5-pro:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                { text: IMAGE_DESCRIPTION_PROMPT },
                {
                  inline_data: {
                    mime_type: mimeType,
                    data: imageData.toString("base64")
                  }
                }
              ]
            }
          ]
        })
      }
    );
    if (!response.ok) {
      await handleApiError(response, "Google Gemini");
    }
    const data = await response.json();
    return parseImageResponse(data.candidates[0].content.parts[0].text);
  }
};
var ImageDescriptionService = class _ImageDescriptionService extends Service {
  static serviceType = ServiceType2.IMAGE_DESCRIPTION;
  initialized = false;
  runtime = null;
  provider = null;
  getInstance() {
    return _ImageDescriptionService.getInstance();
  }
  async initialize(runtime) {
    elizaLogger2.log("Initializing ImageDescriptionService");
    this.runtime = runtime;
  }
  async initializeProvider() {
    if (!this.runtime) {
      throw new Error("Runtime is required for image recognition");
    }
    const availableModels = [
      ModelProviderName.LLAMALOCAL,
      ModelProviderName.ANTHROPIC,
      ModelProviderName.GOOGLE,
      ModelProviderName.OPENAI,
      ModelProviderName.GROQ
    ].join(", ");
    const model = models[this.runtime?.character?.modelProvider];
    if (this.runtime.imageVisionModelProvider) {
      if (this.runtime.imageVisionModelProvider === ModelProviderName.LLAMALOCAL || this.runtime.imageVisionModelProvider === ModelProviderName.OLLAMA) {
        this.provider = new LocalImageProvider();
        elizaLogger2.debug("Using local provider for vision model");
      } else if (this.runtime.imageVisionModelProvider === ModelProviderName.ANTHROPIC) {
        this.provider = new AnthropicImageProvider(this.runtime);
        elizaLogger2.debug("Using anthropic for vision model");
      } else if (this.runtime.imageVisionModelProvider === ModelProviderName.GOOGLE) {
        this.provider = new GoogleImageProvider(this.runtime);
        elizaLogger2.debug("Using google for vision model");
      } else if (this.runtime.imageVisionModelProvider === ModelProviderName.OPENAI) {
        this.provider = new OpenAIImageProvider(this.runtime);
        elizaLogger2.debug("Using openai for vision model");
      } else if (this.runtime.imageVisionModelProvider === ModelProviderName.GROQ) {
        this.provider = new GroqImageProvider(this.runtime);
        elizaLogger2.debug("Using Groq for vision model");
      } else {
        elizaLogger2.warn(
          `Unsupported image vision model provider: ${this.runtime.imageVisionModelProvider}. Please use one of the following: ${availableModels}. Update the 'imageVisionModelProvider' field in the character file.`
        );
        return false;
      }
    } else if (model === models[ModelProviderName.LLAMALOCAL] || model === models[ModelProviderName.OLLAMA]) {
      this.provider = new LocalImageProvider();
      elizaLogger2.debug("Using local provider for vision model");
    } else if (model === models[ModelProviderName.ANTHROPIC]) {
      this.provider = new AnthropicImageProvider(this.runtime);
      elizaLogger2.debug("Using anthropic for vision model");
    } else if (model === models[ModelProviderName.GOOGLE]) {
      this.provider = new GoogleImageProvider(this.runtime);
      elizaLogger2.debug("Using google for vision model");
    } else if (model === models[ModelProviderName.GROQ]) {
      this.provider = new GroqImageProvider(this.runtime);
      elizaLogger2.debug("Using groq for vision model");
    } else {
      elizaLogger2.debug("Using default openai for vision model");
      this.provider = new OpenAIImageProvider(this.runtime);
    }
    try {
      await this.provider.initialize();
    } catch {
      elizaLogger2.error(
        `Failed to initialize the image vision model provider: ${this.runtime.imageVisionModelProvider}`
      );
      return false;
    }
    return true;
  }
  async loadImageData(imageUrlOrPath) {
    let loadedImageData;
    let loadedMimeType;
    const { imageData, mimeType } = await this.fetchImage(imageUrlOrPath);
    const skipConversion = mimeType === "image/jpeg" || mimeType === "image/jpg" || mimeType === "image/png";
    if (skipConversion) {
      loadedImageData = imageData;
      loadedMimeType = mimeType;
    } else {
      const converted = await this.convertImageDataToFormat(
        imageData,
        "png"
      );
      loadedImageData = converted.imageData;
      loadedMimeType = converted.mimeType;
    }
    if (!loadedImageData || loadedImageData.length === 0) {
      throw new Error("Failed to fetch image data");
    }
    return { data: loadedImageData, mimeType: loadedMimeType };
  }
  async convertImageDataToFormat(data, format = "png") {
    const tempFilePath = path.join(
      os.tmpdir(),
      `tmp_img_${Date.now()}.${format}`
    );
    try {
      await sharp(data).toFormat(format).toFile(tempFilePath);
      const { imageData, mimeType } = await this.fetchImage(tempFilePath);
      return {
        imageData,
        mimeType
      };
    } finally {
      fs.unlinkSync(tempFilePath);
    }
  }
  async fetchImage(imageUrlOrPath) {
    let imageData;
    let mimeType;
    if (fs.existsSync(imageUrlOrPath)) {
      imageData = fs.readFileSync(imageUrlOrPath);
      const ext = path.extname(imageUrlOrPath).slice(1).toLowerCase();
      mimeType = ext ? `image/${ext}` : "image/jpeg";
    } else {
      const response = await fetch(imageUrlOrPath);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch image: ${response.statusText}`
        );
      }
      imageData = Buffer.from(await response.arrayBuffer());
      mimeType = response.headers.get("content-type") || "image/jpeg";
    }
    return { imageData, mimeType };
  }
  async describeImage(imageUrlOrPath) {
    if (!this.initialized) {
      this.initialized = await this.initializeProvider();
    }
    if (this.initialized) {
      try {
        const { data, mimeType } = await this.loadImageData(imageUrlOrPath);
        return await this.provider.describeImage(data, mimeType);
      } catch (error) {
        elizaLogger2.error("Error in describeImage:", error);
        throw error;
      }
    }
  }
};

// src/index.ts
function createNodePlugin() {
  return {
    name: "default",
    description: "Default plugin, with basic actions and evaluators",
    services: [
      new ImageDescriptionService()
    ],
    actions: [describeImage]
  };
}
export {
  createNodePlugin
};
//# sourceMappingURL=index.js.map