import { appendFileSync, mkdirSync } from "fs";
import OpenAI from "openai";
import type {
	Tool as OpenAITool,
	ResponseCreateParamsStreaming,
	ResponseFunctionToolCall,
	ResponseInput,
	ResponseInputContent,
	ResponseInputImage,
	ResponseInputText,
	ResponseOutputMessage,
	ResponseReasoningItem,
} from "openai/resources/responses/responses.js";
import { dirname } from "path";
import { calculateCost } from "../models.js";
import { getEnvApiKey } from "../stream.js";
import type {
	Api,
	AssistantMessage,
	Context,
	Model,
	StopReason,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { parseStreamingJson } from "../utils/json-parse.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { transformMessages } from "./transorm-messages.js";

/** Fast deterministic hash to shorten long strings */
function shortHash(str: string): string {
	let h1 = 0xdeadbeef;
	let h2 = 0x41c6ce57;
	for (let i = 0; i < str.length; i++) {
		const ch = str.charCodeAt(i);
		h1 = Math.imul(h1 ^ ch, 2654435761);
		h2 = Math.imul(h2 ^ ch, 1597334677);
	}
	h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
	h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
	return (h2 >>> 0).toString(36) + (h1 >>> 0).toString(36);
}

type FetchFn = typeof fetch;
type FetchRequestInfo = Parameters<FetchFn>[0];
type FetchRequestInit = NonNullable<Parameters<FetchFn>[1]>;
type FetchBodyInit = FetchRequestInit["body"];
type FetchHeadersInit = FetchRequestInit["headers"];

let httpRequestSequence = 0;

function nextHttpRequestId(): string {
	httpRequestSequence += 1;
	return `http_${Date.now()}_${httpRequestSequence}`;
}

function writeHttpLog(logPath: string, entry: Record<string, unknown>): void {
	try {
		mkdirSync(dirname(logPath), { recursive: true });
		appendFileSync(logPath, `${JSON.stringify(entry)}\n`);
	} catch {
		// Best-effort logging only.
	}
}

function headersToRecord(headers?: FetchHeadersInit | Headers): Record<string, string> {
	const record: Record<string, string> = {};
	if (!headers) {
		return record;
	}
	const normalized = new Headers(headers);
	normalized.forEach((value, key) => {
		record[key] = value;
	});
	return record;
}

function isReadableStream(value: unknown): value is ReadableStream<Uint8Array> {
	if (!value || typeof value !== "object") {
		return false;
	}
	const candidate = value as { getReader?: unknown };
	return typeof candidate.getReader === "function";
}

function serializeBody(body?: FetchBodyInit | null): { bodyText?: string; bodyType?: string } {
	if (body === undefined || body === null) {
		return {};
	}
	if (typeof body === "string") {
		return { bodyText: body };
	}
	if (body instanceof URLSearchParams) {
		return { bodyText: body.toString() };
	}
	if (body instanceof ArrayBuffer) {
		return { bodyText: new TextDecoder().decode(new Uint8Array(body)) };
	}
	if (ArrayBuffer.isView(body)) {
		return { bodyText: new TextDecoder().decode(new Uint8Array(body.buffer, body.byteOffset, body.byteLength)) };
	}
	if (isReadableStream(body)) {
		return { bodyType: "ReadableStream" };
	}
	if (typeof body === "object" && "constructor" in body) {
		const ctor = (body as { constructor?: { name?: string } }).constructor;
		return { bodyType: ctor?.name ?? "object" };
	}
	return { bodyType: typeof body };
}

function requestInfoToUrl(input: FetchRequestInfo): string {
	if (typeof input === "string") return input;
	if (input instanceof URL) return input.toString();
	return input.url;
}

function requestInfoToMethod(input: FetchRequestInfo, init?: FetchRequestInit): string {
	if (init?.method) return init.method;
	if (input instanceof Request) return input.method;
	return "GET";
}

function maybeGetLoggedFetch(model: Model<"openai-responses">): FetchFn | undefined {
	const logPath = process.env.PI_LOG_HTTP_PATH;
	if (!logPath) {
		return undefined;
	}

	const baseFetch = globalThis.fetch as FetchFn | undefined;
	if (!baseFetch) {
		return undefined;
	}

	return async (input: FetchRequestInfo, init?: FetchRequestInit): Promise<Response> => {
		const requestId = nextHttpRequestId();
		const startedAt = Date.now();
		const url = requestInfoToUrl(input);
		const method = requestInfoToMethod(input, init);
		const headers = headersToRecord(init?.headers ?? (input instanceof Request ? input.headers : undefined));
		const { bodyText, bodyType } = serializeBody(init?.body ?? (input instanceof Request ? input.body : undefined));

		writeHttpLog(logPath, {
			timestamp: new Date().toISOString(),
			requestId,
			direction: "request",
			provider: model.provider,
			model: model.id,
			method,
			url,
			headers,
			...(bodyText !== undefined ? { bodyText } : {}),
			...(bodyType ? { bodyType } : {}),
		});

		const response = await baseFetch(input, init);
		const responseHeaders = headersToRecord(response.headers);
		const responseInfo = {
			timestamp: new Date().toISOString(),
			requestId,
			direction: "response",
			provider: model.provider,
			model: model.id,
			status: response.status,
			statusText: response.statusText,
			durationMs: Date.now() - startedAt,
			headers: responseHeaders,
		};

		void response
			.clone()
			.text()
			.then((body: string) => {
				writeHttpLog(logPath, { ...responseInfo, bodyText: body });
			})
			.catch((error: unknown) => {
				const message = error instanceof Error ? error.message : String(error);
				writeHttpLog(logPath, { ...responseInfo, error: message });
			});

		return response;
	};
}

// OpenAI Responses-specific options
export interface OpenAIResponsesOptions extends StreamOptions {
	reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh";
	reasoningSummary?: "auto" | "detailed" | "concise" | null;
}

/**
 * Generate function for OpenAI Responses API
 */
export const streamOpenAIResponses: StreamFunction<"openai-responses"> = (
	model: Model<"openai-responses">,
	context: Context,
	options?: OpenAIResponsesOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	// Start async processing
	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: "openai-responses" as Api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			// Create OpenAI client
			const apiKey = options?.apiKey || getEnvApiKey(model.provider) || "";
			const headers = buildHeaders(model, context);
			const client = createClient(model, context, apiKey, headers);
			const params = buildParams(model, context, options);
			maybeLogRequest(model, params, headers);
			const openaiStream = await client.responses.create(params, { signal: options?.signal });
			stream.push({ type: "start", partial: output });

			let currentItem: ResponseReasoningItem | ResponseOutputMessage | ResponseFunctionToolCall | null = null;
			let currentBlock: ThinkingContent | TextContent | (ToolCall & { partialJson: string }) | null = null;
			const blocks = output.content;
			const blockIndex = () => blocks.length - 1;

			for await (const event of openaiStream) {
				// Handle output item start
				if (event.type === "response.output_item.added") {
					const item = event.item;
					if (item.type === "reasoning") {
						currentItem = item;
						currentBlock = { type: "thinking", thinking: "" };
						output.content.push(currentBlock);
						stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
					} else if (item.type === "message") {
						currentItem = item;
						currentBlock = { type: "text", text: "" };
						output.content.push(currentBlock);
						stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
					} else if (item.type === "function_call") {
						currentItem = item;
						currentBlock = {
							type: "toolCall",
							id: `${item.call_id}|${item.id}`,
							name: item.name,
							arguments: {},
							partialJson: item.arguments || "",
						};
						output.content.push(currentBlock);
						stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
					}
				}
				// Handle reasoning summary deltas
				else if (event.type === "response.reasoning_summary_part.added") {
					if (currentItem && currentItem.type === "reasoning") {
						currentItem.summary = currentItem.summary || [];
						currentItem.summary.push(event.part);
					}
				} else if (event.type === "response.reasoning_summary_text.delta") {
					if (
						currentItem &&
						currentItem.type === "reasoning" &&
						currentBlock &&
						currentBlock.type === "thinking"
					) {
						currentItem.summary = currentItem.summary || [];
						const lastPart = currentItem.summary[currentItem.summary.length - 1];
						if (lastPart) {
							currentBlock.thinking += event.delta;
							lastPart.text += event.delta;
							stream.push({
								type: "thinking_delta",
								contentIndex: blockIndex(),
								delta: event.delta,
								partial: output,
							});
						}
					}
				}
				// Add a new line between summary parts (hack...)
				else if (event.type === "response.reasoning_summary_part.done") {
					if (
						currentItem &&
						currentItem.type === "reasoning" &&
						currentBlock &&
						currentBlock.type === "thinking"
					) {
						currentItem.summary = currentItem.summary || [];
						const lastPart = currentItem.summary[currentItem.summary.length - 1];
						if (lastPart) {
							currentBlock.thinking += "\n\n";
							lastPart.text += "\n\n";
							stream.push({
								type: "thinking_delta",
								contentIndex: blockIndex(),
								delta: "\n\n",
								partial: output,
							});
						}
					}
				}
				// Handle text output deltas
				else if (event.type === "response.content_part.added") {
					if (currentItem && currentItem.type === "message") {
						currentItem.content = currentItem.content || [];
						// Filter out ReasoningText, only accept output_text and refusal
						if (event.part.type === "output_text" || event.part.type === "refusal") {
							currentItem.content.push(event.part);
						}
					}
				} else if (event.type === "response.output_text.delta") {
					if (currentItem && currentItem.type === "message" && currentBlock && currentBlock.type === "text") {
						const lastPart = currentItem.content[currentItem.content.length - 1];
						if (lastPart && lastPart.type === "output_text") {
							currentBlock.text += event.delta;
							lastPart.text += event.delta;
							stream.push({
								type: "text_delta",
								contentIndex: blockIndex(),
								delta: event.delta,
								partial: output,
							});
						}
					}
				} else if (event.type === "response.refusal.delta") {
					if (currentItem && currentItem.type === "message" && currentBlock && currentBlock.type === "text") {
						const lastPart = currentItem.content[currentItem.content.length - 1];
						if (lastPart && lastPart.type === "refusal") {
							currentBlock.text += event.delta;
							lastPart.refusal += event.delta;
							stream.push({
								type: "text_delta",
								contentIndex: blockIndex(),
								delta: event.delta,
								partial: output,
							});
						}
					}
				}
				// Handle function call argument deltas
				else if (event.type === "response.function_call_arguments.delta") {
					if (
						currentItem &&
						currentItem.type === "function_call" &&
						currentBlock &&
						currentBlock.type === "toolCall"
					) {
						currentBlock.partialJson += event.delta;
						currentBlock.arguments = parseStreamingJson(currentBlock.partialJson);
						stream.push({
							type: "toolcall_delta",
							contentIndex: blockIndex(),
							delta: event.delta,
							partial: output,
						});
					}
				}
				// Handle output item completion
				else if (event.type === "response.output_item.done") {
					const item = event.item;

					if (item.type === "reasoning" && currentBlock && currentBlock.type === "thinking") {
						currentBlock.thinking = item.summary?.map((s) => s.text).join("\n\n") || "";
						currentBlock.thinkingSignature = JSON.stringify(item);
						stream.push({
							type: "thinking_end",
							contentIndex: blockIndex(),
							content: currentBlock.thinking,
							partial: output,
						});
						currentBlock = null;
					} else if (item.type === "message" && currentBlock && currentBlock.type === "text") {
						currentBlock.text = item.content.map((c) => (c.type === "output_text" ? c.text : c.refusal)).join("");
						currentBlock.textSignature = item.id;
						stream.push({
							type: "text_end",
							contentIndex: blockIndex(),
							content: currentBlock.text,
							partial: output,
						});
						currentBlock = null;
					} else if (item.type === "function_call") {
						const toolCall: ToolCall = {
							type: "toolCall",
							id: `${item.call_id}|${item.id}`,
							name: item.name,
							arguments: JSON.parse(item.arguments),
						};

						stream.push({ type: "toolcall_end", contentIndex: blockIndex(), toolCall, partial: output });
					}
				}
				// Handle completion
				else if (event.type === "response.completed") {
					const response = event.response;
					if (response?.usage) {
						const cachedTokens = response.usage.input_tokens_details?.cached_tokens || 0;
						output.usage = {
							// OpenAI includes cached tokens in input_tokens, so subtract to get non-cached input
							input: (response.usage.input_tokens || 0) - cachedTokens,
							output: response.usage.output_tokens || 0,
							cacheRead: cachedTokens,
							cacheWrite: 0,
							totalTokens: response.usage.total_tokens || 0,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						};
					}
					calculateCost(model, output.usage);
					// Map status to stop reason
					output.stopReason = mapStopReason(response?.status);
					if (output.content.some((b) => b.type === "toolCall") && output.stopReason === "stop") {
						output.stopReason = "toolUse";
					}
				}
				// Handle errors
				else if (event.type === "error") {
					throw new Error(`Error Code ${event.code}: ${event.message}` || "Unknown error");
				} else if (event.type === "response.failed") {
					throw new Error("Unknown error");
				}
			}

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}

			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error("An unkown error ocurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			for (const block of output.content) delete (block as any).index;
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

function createClient(
	model: Model<"openai-responses">,
	context: Context,
	apiKey?: string,
	headers?: Record<string, string>,
) {
	if (!apiKey) {
		if (!process.env.OPENAI_API_KEY) {
			throw new Error(
				"OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as an argument.",
			);
		}
		apiKey = process.env.OPENAI_API_KEY;
	}

	const resolvedHeaders = headers ?? buildHeaders(model, context);
	const loggedFetch = maybeGetLoggedFetch(model);

	return new OpenAI({
		apiKey,
		baseURL: model.baseUrl,
		dangerouslyAllowBrowser: true,
		defaultHeaders: resolvedHeaders,
		...(loggedFetch ? { fetch: loggedFetch } : {}),
	});
}

function buildHeaders(model: Model<"openai-responses">, context: Context): Record<string, string> {
	const headers = { ...model.headers };
	if (model.provider === "github-copilot") {
		// Copilot expects X-Initiator to indicate whether the request is user-initiated
		// or agent-initiated (e.g. follow-up after assistant/tool messages). If there is
		// no prior message, default to user-initiated.
		const messages = context.messages || [];
		const lastMessage = messages[messages.length - 1];
		const isAgentCall = lastMessage ? lastMessage.role !== "user" : false;
		headers["X-Initiator"] = isAgentCall ? "agent" : "user";
		headers["Openai-Intent"] = "conversation-edits";

		// Copilot requires this header when sending images
		const hasImages = messages.some((msg) => {
			if (msg.role === "user" && Array.isArray(msg.content)) {
				return msg.content.some((c) => c.type === "image");
			}
			if (msg.role === "toolResult" && Array.isArray(msg.content)) {
				return msg.content.some((c) => c.type === "image");
			}
			return false;
		});
		if (hasImages) {
			headers["Copilot-Vision-Request"] = "true";
		}
	}

	return headers;
}

function buildParams(model: Model<"openai-responses">, context: Context, options?: OpenAIResponsesOptions) {
	const useInstructions = model.provider.toLowerCase() === "codex";
	const messages = convertMessages(model, context, useInstructions);

	const params: ResponseCreateParamsStreaming & { instructions?: string; store?: boolean } = {
		model: model.id,
		input: messages,
		stream: true,
	};

	if (options?.maxTokens && !useInstructions) {
		params.max_output_tokens = options?.maxTokens;
	}

	if (options?.temperature !== undefined) {
		params.temperature = options?.temperature;
	}

	if (context.tools) {
		params.tools = convertTools(context.tools, useInstructions);
	}

	if (useInstructions) {
		params.store = false;
		if (context.systemPrompt) {
			params.instructions = sanitizeSurrogates(context.systemPrompt);
		}
	}

	if (model.reasoning) {
		if (options?.reasoningEffort || options?.reasoningSummary) {
			params.reasoning = {
				effort: options?.reasoningEffort || "medium",
				summary: options?.reasoningSummary || "auto",
			};
			params.include = ["reasoning.encrypted_content"];
		} else if (!useInstructions) {
			if (model.name.startsWith("gpt-5")) {
				// Jesus Christ, see https://community.openai.com/t/need-reasoning-false-option-for-gpt-5/1351588/7
				messages.push({
					role: "developer",
					content: [
						{
							type: "input_text",
							text: "# Juice: 0 !important",
						},
					],
				});
			}
		}
	}

	return params;
}

function convertMessages(model: Model<"openai-responses">, context: Context, useInstructions: boolean): ResponseInput {
	const messages: ResponseInput = [];

	const transformedMessages = transformMessages(context.messages, model);

	if (context.systemPrompt && !useInstructions) {
		const role = model.reasoning ? "developer" : "system";
		messages.push({
			role,
			content: sanitizeSurrogates(context.systemPrompt),
		});
	}

	let msgIndex = 0;
	for (const msg of transformedMessages) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				if (useInstructions) {
					messages.push({
						type: "message",
						role: "user",
						content: [{ type: "input_text", text: sanitizeSurrogates(msg.content) }],
					});
				} else {
					messages.push({
						role: "user",
						content: [{ type: "input_text", text: sanitizeSurrogates(msg.content) }],
					});
				}
			} else {
				const content: ResponseInputContent[] = msg.content.map((item): ResponseInputContent => {
					if (item.type === "text") {
						return {
							type: "input_text",
							text: sanitizeSurrogates(item.text),
						} satisfies ResponseInputText;
					} else {
						return {
							type: "input_image",
							detail: "auto",
							image_url: `data:${item.mimeType};base64,${item.data}`,
						} satisfies ResponseInputImage;
					}
				});
				const filteredContent = !model.input.includes("image")
					? content.filter((c) => c.type !== "input_image")
					: content;
				if (filteredContent.length === 0) continue;
				if (useInstructions) {
					messages.push({
						type: "message",
						role: "user",
						content: filteredContent,
					});
				} else {
					messages.push({
						role: "user",
						content: filteredContent,
					});
				}
			}
		} else if (msg.role === "assistant") {
			const output: ResponseInput = [];

			for (const block of msg.content) {
				// Do not submit thinking blocks if the completion had an error (i.e. abort)
				if (block.type === "thinking" && msg.stopReason !== "error" && !useInstructions) {
					if (block.thinkingSignature) {
						const reasoningItem = JSON.parse(block.thinkingSignature);
						output.push(reasoningItem);
					}
				} else if (block.type === "text") {
					const textBlock = block as TextContent;
					if (useInstructions) {
						output.push({
							type: "message",
							role: "assistant",
							content: [{ type: "output_text", text: sanitizeSurrogates(textBlock.text) }],
						} as any);
					} else {
						// OpenAI requires id to be max 64 characters
						let msgId = textBlock.textSignature;
						if (!msgId) {
							msgId = `msg_${msgIndex}`;
						} else if (msgId.length > 64) {
							msgId = `msg_${shortHash(msgId)}`;
						}
						output.push({
							type: "message",
							role: "assistant",
							content: [{ type: "output_text", text: sanitizeSurrogates(textBlock.text), annotations: [] }],
							status: "completed",
							id: msgId,
						} satisfies ResponseOutputMessage);
					}
					// Do not submit toolcall blocks if the completion had an error (i.e. abort)
				} else if (block.type === "toolCall" && msg.stopReason !== "error") {
					const toolCall = block as ToolCall;
					const [callId, itemId] = toolCall.id.split("|");
					if (useInstructions) {
						output.push({
							type: "function_call",
							call_id: callId,
							name: toolCall.name,
							arguments: JSON.stringify(toolCall.arguments),
						} as any);
					} else {
						output.push({
							type: "function_call",
							id: itemId,
							call_id: callId,
							name: toolCall.name,
							arguments: JSON.stringify(toolCall.arguments),
						});
					}
				}
			}
			if (output.length === 0) continue;
			messages.push(...output);
		} else if (msg.role === "toolResult") {
			// Extract text and image content
			const textResult = msg.content
				.filter((c) => c.type === "text")
				.map((c) => (c as any).text)
				.join("\n");
			const hasImages = msg.content.some((c) => c.type === "image");

			// Always send function_call_output with text (or placeholder if only images)
			const hasText = textResult.length > 0;
			messages.push({
				type: "function_call_output",
				call_id: msg.toolCallId.split("|")[0],
				output: sanitizeSurrogates(hasText ? textResult : "(see attached image)"),
			});

			// If there are images and model supports them, send a follow-up user message with images
			if (hasImages && model.input.includes("image")) {
				const contentParts: ResponseInputContent[] = [];

				// Add text prefix
				contentParts.push({
					type: "input_text",
					text: "Attached image(s) from tool result:",
				} satisfies ResponseInputText);

				// Add images
				for (const block of msg.content) {
					if (block.type === "image") {
						contentParts.push({
							type: "input_image",
							detail: "auto",
							image_url: `data:${(block as any).mimeType};base64,${(block as any).data}`,
						} satisfies ResponseInputImage);
					}
				}

				messages.push({
					role: "user",
					content: contentParts,
					...(useInstructions ? { type: "message" } : {}),
				} as any);
			}
		}
		msgIndex++;
	}

	return messages;
}

function convertTools(tools: Tool[], useInstructions: boolean): OpenAITool[] {
	return tools.map((tool) => {
		const base = {
			type: "function",
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters as any, // TypeBox already generates JSON Schema
		};
		if (useInstructions) {
			return base as OpenAITool;
		}
		return { ...base, strict: null } as OpenAITool;
	});
}

function maybeLogRequest(
	model: Model<"openai-responses">,
	params: ResponseCreateParamsStreaming & { instructions?: string; store?: boolean },
	headers: Record<string, string>,
): void {
	const logPath = process.env.PI_LOG_REQUESTS_PATH;
	if (!logPath) {
		return;
	}
	try {
		const redactedParams = JSON.parse(
			JSON.stringify(params, (key, value) => {
				if (key === "image_url" && typeof value === "string" && value.startsWith("data:")) {
					const base64Index = value.indexOf("base64,");
					const size = base64Index === -1 ? value.length : value.length - (base64Index + "base64,".length);
					return `data:<omitted base64 ${size} chars>`;
				}
				return value;
			}),
		);
		const redactedHeaders: Record<string, string> = {};
		for (const [key, value] of Object.entries(headers)) {
			if (/authorization/i.test(key)) {
				redactedHeaders[key] = "<redacted>";
			} else {
				redactedHeaders[key] = value;
			}
		}
		const entry = {
			timestamp: new Date().toISOString(),
			provider: model.provider,
			model: model.id,
			baseUrl: model.baseUrl,
			headers: redactedHeaders,
			params: redactedParams,
		};
		mkdirSync(dirname(logPath), { recursive: true });
		appendFileSync(logPath, `${JSON.stringify(entry)}\n`);
	} catch {
		// Best-effort logging only.
	}
}

function mapStopReason(status: OpenAI.Responses.ResponseStatus | undefined): StopReason {
	if (!status) return "stop";
	switch (status) {
		case "completed":
			return "stop";
		case "incomplete":
			return "length";
		case "failed":
		case "cancelled":
			return "error";
		// These two are wonky ...
		case "in_progress":
		case "queued":
			return "stop";
		default: {
			const _exhaustive: never = status;
			throw new Error(`Unhandled stop reason: ${_exhaustive}`);
		}
	}
}
