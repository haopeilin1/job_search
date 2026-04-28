/**
 * API 客户端 —— 与求职雷达 Agent 后端对接
 *
 * 当前实现：
 * - 直接调用后端 REST API（/api/v1/chat）
 * - 文件上传采用简化方案：前端读取 File 为 base64 data URL，作为 attachments 数组传入
 * TODO: 后续接入真实文件上传服务（OSS / 临时存储）后，改为传递 file_url
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// ──────────────────────────── 类型定义 ────────────────────────────

export type IntentType = "match_single" | "global_match" | "rag_qa" | "general";
export type RouteLayer = "rule" | "llm" | "llm_fallback";
export type ReplyType = "match_report" | "global_ranking" | "rag_answer" | "text";

export interface ChatAttachment {
  filename: string;
  content_type: string;
  url?: string;
  data?: string;  // base64 编码的文件内容（图片直接上传时使用）
}

export interface ChatContextItem {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  session_id?: string;
  message: string;
  type: "text" | "image" | "file";
  attachments?: ChatAttachment[];
  context?: ChatContextItem[];
}

export interface RouteMeta {
  intent: IntentType;
  confidence: number;
  layer: RouteLayer;
  rule?: string;
  reason?: string;
}

export interface RagSource {
  jd_id: string;
  company: string;
  position: string;
  snippet: string;
}

export interface ChatReply {
  type: ReplyType;
  content: string;
  data?: Record<string, any>;
  sources?: RagSource[];
}

export interface ChatResponse {
  session_id: string;
  intent: IntentType;
  route_meta: RouteMeta;
  reply: ChatReply;
}

// ──────────────────────────── 工具函数 ────────────────────────────

/**
 * 将 File 对象转为 base64 data URL，用于简化版附件上传
 * TODO: 后续接入真实文件上传服务后，此函数可替换为上传到 OSS 并返回 file_url
 */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * 统一 fetch 封装，处理 JSON 请求与响应
 */
async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`API Error ${res.status}: ${errText}`);
  }
  return res.json() as Promise<T>;
}

// ──────────────────────────── 核心 API ────────────────────────────

/**
 * 发送对话消息到后端统一入口
 *
 * @param payload 请求体（包含 message / type / attachments / context）
 * @returns 后端返回的 ChatResponse（含意图识别结果与回复内容）
 */
export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  return fetchJson<ChatResponse>(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * 上传简历文件（复用现有 resumes/upload 接口）
 * TODO: 后续接入真实解析后，改为调用 /api/v1/resumes/upload
 */
export async function uploadResume(file: File): Promise<any> {
  const form = new FormData();
  form.append("file", file);
  form.append("force_update", "true");

  const res = await fetch(`${API_BASE}/api/v1/resumes/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Upload Error ${res.status}: ${err}`);
  }
  return res.json();
}
