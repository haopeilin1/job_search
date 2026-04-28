/**
 * Zustand 对话状态管理 —— 求职雷达 Agent
 *
 * 设计决策：
 * - 使用 Zustand 而非 Redux/Context，因为对话状态扁平、更新频繁，Zustand 性能更优
 * - persist 中间件将对话历史缓存到 localStorage，刷新页面不丢失
 * - 每条消息独立存储 routeMeta / replyType，便于前端按类型渲染不同 UI
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { v4 as uuidv4 } from "uuid";

import type {
  IntentType,
  RouteMeta,
  ReplyType,
  ChatResponse,
} from "@/lib/api";

// ──────────────────────────── 消息模型 ────────────────────────────

export interface ChatMessage {
  id: string; // uuid
  role: "user" | "assistant";
  content: string; // 文本内容（用户输入或系统回复）
  intent?: IntentType; // 系统消息携带的意图标签
  routeMeta?: RouteMeta; // 系统消息携带的路由调试信息
  replyType?: ReplyType; // 系统消息渲染类型
  replyData?: Record<string, any>; // 结构化数据（匹配报告、排名等）
  sources?: any[]; // RAG 引用来源
  isLoading?: boolean; // 是否正在生成（骨架屏占位）
  attachments?: any[]; // 用户上传的文件列表
}

// ──────────────────────────── Store 状态 ────────────────────────────

interface ChatStore {
  messages: ChatMessage[];
  sessionId: string | null;
  isLoading: boolean;

  // Actions
  addUserMessage: (content: string, attachments?: any[]) => void;
  addLoadingMessage: () => string; // 返回 loading message id
  setAssistantResponse: (loadingId: string, response: ChatResponse) => void;
  clearMessages: () => void;
}

// ──────────────────────────── 辅助函数 ────────────────────────────

function generateSessionId(): string {
  return uuidv4();
}

// ──────────────────────────── Store 实现 ────────────────────────────

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      messages: [],
      sessionId: null,
      isLoading: false,

      /**
       * 添加用户消息到对话列表末尾
       */
      addUserMessage: (content: string, attachments?: any[]) => {
        const msg: ChatMessage = {
          id: uuidv4(),
          role: "user",
          content,
          attachments,
        };
        set((state) => ({
          messages: [...state.messages, msg],
          isLoading: true,
        }));
      },

      /**
       * 添加 Loading 占位消息，返回其 id 供后续替换
       */
      addLoadingMessage: () => {
        const id = uuidv4();
        const msg: ChatMessage = {
          id,
          role: "assistant",
          content: "",
          isLoading: true,
        };
        set((state) => ({
          messages: [...state.messages, msg],
          isLoading: true,
        }));
        return id;
      },

      /**
       * 用后端返回的数据替换 Loading 消息
       */
      setAssistantResponse: (loadingId: string, response: ChatResponse) => {
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === loadingId
              ? {
                  ...m,
                  isLoading: false,
                  content: response.reply.content,
                  intent: response.intent,
                  routeMeta: response.route_meta,
                  replyType: response.reply.type,
                  replyData: response.reply.data,
                  sources: response.reply.sources,
                }
              : m
          ),
          isLoading: false,
          sessionId: response.session_id,
        }));
      },

      /**
       * 清空对话历史（同时清除 localStorage 缓存）
       */
      clearMessages: () => {
        set({ messages: [], sessionId: null, isLoading: false });
      },
    }),
    {
      name: "career-radar-chat", // localStorage key
      partialize: (state) => ({
        messages: state.messages,
        sessionId: state.sessionId,
      }),
    }
  )
);
