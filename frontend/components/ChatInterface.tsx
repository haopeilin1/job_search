"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import {
  Send,
  Upload,
  Loader2,
  BrainCircuit,
  BookOpen,
  Settings,
  Zap,
  MessageSquare,
  FileText,
  Image as ImageIcon,
} from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";

import { useChatStore, ChatMessage } from "@/store/chatStore";
import {
  sendChatMessage,
  fileToDataUrl,
  ChatRequest,
  ChatContextItem,
  type RouteLayer,
} from "@/lib/api";
import { MatchReportCard } from "./chat/MatchReportCard";

// ──────────────────────────── 辅助组件 ────────────────────────────

/**
 * 意图调试标签 —— 显示在系统消息右下角，用于面试展示路由决策路径
 */
function IntentBadge({ layer, rule, confidence }: { layer: RouteLayer; rule?: string; confidence?: number }) {
  if (layer === "rule") {
    return (
      <Badge className="bg-emerald-50 text-emerald-700 border-emerald-100 text-[10px] font-black px-2 py-0.5 rounded-full">
        ⚡ 规则命中 · {rule || "unknown"}
      </Badge>
    );
  }
  if (layer === "llm") {
    return (
      <Badge className="bg-blue-50 text-blue-700 border-blue-100 text-[10px] font-black px-2 py-0.5 rounded-full">
        🧠 LLM识别 · c={confidence?.toFixed(2)}
      </Badge>
    );
  }
  return (
    <Badge className="bg-amber-50 text-amber-700 border-amber-100 text-[10px] font-black px-2 py-0.5 rounded-full">
      ⚠️ 置信度低 · 降级为对话
    </Badge>
  );
}

/**
 * 用户附件预览
 */
function AttachmentPreview({ attachments }: { attachments: any[] }) {
  if (!attachments || attachments.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {attachments.map((att, idx) => (
        <div
          key={idx}
          className="flex items-center gap-1.5 bg-white/20 px-2.5 py-1 rounded-xl text-[10px] font-bold"
        >
          {att.mime_type?.startsWith("image/") ? <ImageIcon size={12} /> : <FileText size={12} />}
          <span className="truncate max-w-[120px]">{att.filename || `附件${idx + 1}`}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * 全局排名卡片 —— 横向滚动列表
 */
function GlobalRankingRow({ data }: { data: any }) {
  const items = data?.rankings || [];
  return (
    <div className="w-full space-y-3">
      <div className="bg-white p-5 rounded-[2.5rem] border border-gray-100 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <Zap size={14} className="text-gray-500" />
          <span className="text-xs font-black text-gray-400 uppercase tracking-widest">
            全局推荐排名
          </span>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-2 no-scrollbar">
          {items.map((item: any, idx: number) => (
            <div
              key={idx}
              className={`shrink-0 w-52 rounded-2xl p-4 ${
                idx === 0 ? "bg-gray-800 text-white" : "bg-gray-50 text-gray-800"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className={`text-sm font-black ${idx === 0 ? "text-white" : ""}`}>
                  {item.company}
                </span>
                <span
                  className={`text-lg font-black ${
                    idx === 0 ? "text-white" : scoreColor(item.score)
                  }`}
                >
                  {item.score}
                </span>
              </div>
              <div className={`text-xs font-bold ${idx === 0 ? "text-gray-300" : "text-gray-500"}`}>
                {item.title}
              </div>
              <div
                className={`text-[10px] font-bold mt-2 leading-relaxed ${
                  idx === 0 ? "text-gray-400" : "text-gray-400"
                }`}
              >
                {item.reason}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function scoreColor(score: number): string {
  if (score >= 80) return "text-emerald-600";
  if (score >= 60) return "text-amber-500";
  return "text-red-500";
}

/**
 * RAG 问答卡片 —— 带引用来源
 */
function RagAnswerCard({ text, sources }: { text: string; sources?: any[] }) {
  return (
    <div className="max-w-[85%] p-5 rounded-[2.2rem] bg-white text-gray-800 rounded-tl-none border border-gray-100 shadow-sm">
      <p className="text-sm font-bold leading-relaxed whitespace-pre-line mb-3">{text}</p>
      {sources && sources.length > 0 && (
        <div className="pt-3 border-t border-gray-100 space-y-2">
          <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
            引用来源
          </div>
          {sources.map((s, idx) => (
            <div
              key={idx}
              className="bg-gray-50 rounded-xl p-3 text-xs text-gray-600 font-bold leading-relaxed"
            >
              <div className="text-gray-800 font-black">{s.company} · {s.position}</div>
              <div className="text-gray-400 mt-0.5">{s.snippet}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ──────────────────────────── 消息气泡 ────────────────────────────

function MessageBubble({ message }: { message: ChatMessage }) {
  // 用户消息
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] p-5 rounded-[2.2rem] bg-gray-800 text-white rounded-tr-none shadow-sm">
          <p className="text-sm font-bold leading-relaxed whitespace-pre-line">{message.content}</p>
          {message.attachments && <AttachmentPreview attachments={message.attachments} />}
        </div>
      </div>
    );
  }

  // Loading 骨架屏
  if (message.isLoading) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] p-5 rounded-[2.2rem] bg-white text-gray-800 rounded-tl-none border border-gray-100 shadow-sm space-y-3">
          <div className="flex items-center gap-2">
            <Loader2 size={16} className="animate-spin text-gray-400" />
            <span className="text-sm font-bold text-gray-500">正在分析意图...</span>
          </div>
          <Skeleton className="h-4 w-48 rounded-full" />
          <Skeleton className="h-4 w-32 rounded-full" />
        </div>
      </div>
    );
  }

  // 系统消息 —— 根据 replyType 渲染不同 UI
  const replyType = message.replyType;

  if (replyType === "match_report") {
    const data = message.replyData || {};
    return (
      <div className="w-full">
        <MatchReportCard
          score={data.score || 0}
          verdict={data.label || "匹配分析"}
          dimensions={data.dimensions}
          gapAnalysis={data.gap_analysis}
          questions={data.questions}
        />
        <div className="flex justify-end mt-2">
          {message.routeMeta && (
            <IntentBadge
              layer={message.routeMeta.layer}
              rule={message.routeMeta.rule}
              confidence={message.routeMeta.confidence}
            />
          )}
        </div>
      </div>
    );
  }

  if (replyType === "global_ranking") {
    return (
      <div className="w-full">
        <GlobalRankingRow data={message.replyData || {}} />
        <div className="flex justify-end mt-2">
          {message.routeMeta && (
            <IntentBadge
              layer={message.routeMeta.layer}
              rule={message.routeMeta.rule}
              confidence={message.routeMeta.confidence}
            />
          )}
        </div>
      </div>
    );
  }

  if (replyType === "rag_answer") {
    return (
      <div className="w-full">
        <RagAnswerCard text={message.content} sources={message.sources} />
        <div className="flex justify-end mt-2">
          {message.routeMeta && (
            <IntentBadge
              layer={message.routeMeta.layer}
              rule={message.routeMeta.rule}
              confidence={message.routeMeta.confidence}
            />
          )}
        </div>
      </div>
    );
  }

  // 默认 general 文本
  return (
    <div className="w-full">
      <div className="flex justify-start">
        <div className="max-w-[85%] p-5 rounded-[2.2rem] bg-white text-gray-800 rounded-tl-none border border-gray-100 shadow-sm">
          <p className="text-sm font-bold leading-relaxed whitespace-pre-line">{message.content}</p>
        </div>
      </div>
      <div className="flex justify-end mt-2">
        {message.routeMeta && (
          <IntentBadge
            layer={message.routeMeta.layer}
            rule={message.routeMeta.rule}
            confidence={message.routeMeta.confidence}
          />
        )}
      </div>
    </div>
  );
}

// ──────────────────────────── 主组件 ────────────────────────────

export default function ChatInterface() {
  const { messages, sessionId, isLoading, addUserMessage, addLoadingMessage, setAssistantResponse } =
    useChatStore();

  const [inputValue, setInputValue] = useState("");
  const [pendingAttachments, setPendingAttachments] = useState<any[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  /**
   * 从 Store 中提取最近 2 轮对话作为上下文
   * 规则：取最后 1 条 user + 1 条 assistant（排除 loading）
   */
  const buildContext = useCallback((): ChatContextItem[] => {
    const validMessages = messages.filter((m) => !m.isLoading);
    const context: ChatContextItem[] = [];
    // 取最后 2 轮（最多 4 条消息：user + assistant + user + assistant）
    const recent = validMessages.slice(-4);
    for (const m of recent) {
      context.push({
        role: m.role === "user" ? "user" : "assistant",
        content: m.content,
      });
    }
    return context;
  }, [messages]);

  /**
   * 发送消息主流程
   */
  const handleSend = useCallback(async () => {
    if (!inputValue.trim() && pendingAttachments.length === 0) return;
    const text = inputValue.trim() || "请分析附件内容";

    // 1. 添加用户消息到 Store
    addUserMessage(text, pendingAttachments);
    setInputValue("");
    setPendingAttachments([]);

    // 2. 添加 Loading 占位
    const loadingId = addLoadingMessage();

    // 3. 组装请求
    // 提取纯 base64（去掉 data:image/jpeg;base64, 前缀）
    const extractBase64 = (dataUrl: string): string => {
      const idx = dataUrl.indexOf(",");
      return idx >= 0 ? dataUrl.substring(idx + 1) : dataUrl;
    };

    const payload: ChatRequest = {
      session_id: sessionId || uuidv4(),
      message: text,
      type: pendingAttachments.length > 0 ? "file" : "text",
      attachments: pendingAttachments.map((a) => ({
        filename: a.filename,
        content_type: a.mimeType,
        data: a.dataUrl ? extractBase64(a.dataUrl) : undefined,
      })),
      context: buildContext(),
    };

    // 4. 调用后端
    try {
      const response = await sendChatMessage(payload);
      setAssistantResponse(loadingId, response);
    } catch (err) {
      console.error("[ChatInterface] send failed:", err);
      setAssistantResponse(loadingId, {
        session_id: payload.session_id!,
        intent: "general",
        route_meta: {
          intent: "general",
          confidence: 0,
          layer: "llm_fallback",
          reason: String(err),
        },
        reply: {
          type: "text",
          content: "❌ 发送失败，请检查网络连接或后端服务是否正常运行。",
        },
      });
    }
  }, [inputValue, pendingAttachments, sessionId, buildContext, addUserMessage, addLoadingMessage, setAssistantResponse]);

  /**
   * 处理文件选择
   */
  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      e.target.value = "";

      try {
        const dataUrl = await fileToDataUrl(file);
        setPendingAttachments((prev) => [
          ...prev,
          { filename: file.name, mimeType: file.type, dataUrl },
        ]);
      } catch (err) {
        console.error("[ChatInterface] file read failed:", err);
      }
    },
    []
  );

  /**
   * 键盘事件：Enter 发送，Shift+Enter 换行
   */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#FAFAFA] relative overflow-hidden">
      {/* 顶部 Header */}
      <header className="px-6 pt-6 pb-3 flex items-center justify-between border-b border-gray-100 bg-white/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-gray-700 to-gray-800 w-10 h-10 rounded-2xl flex items-center justify-center shadow-md shadow-gray-200 border-2 border-white">
            <BrainCircuit size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-lg font-black text-gray-800 leading-none">求职雷达</h1>
            <p className="text-[10px] text-gray-400 font-bold mt-0.5">Career Intelligence</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="w-10 h-10 rounded-2xl bg-white shadow-sm border border-gray-100"
          >
            <Settings size={18} className="text-gray-500" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="w-10 h-10 rounded-2xl bg-white shadow-sm border border-gray-100"
          >
            <BookOpen size={18} className="text-gray-600" />
          </Button>
        </div>
      </header>

      {/* 消息流区域 */}
      <ScrollArea className="flex-1 px-6 py-5" ref={scrollRef as any}>
        <div className="space-y-5 max-w-3xl mx-auto">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 bg-gray-100 rounded-3xl flex items-center justify-center mb-4">
                <MessageSquare size={28} className="text-gray-400" />
              </div>
              <h2 className="text-lg font-black text-gray-700 mb-1">开始你的求职对话</h2>
              <p className="text-sm text-gray-400 font-bold max-w-xs">
                上传 JD 截图做匹配分析，或询问知识库中的岗位详情
              </p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
        </div>
      </ScrollArea>

      {/* 输入栏 */}
      <div className="px-4 py-4 bg-white border-t border-gray-100 z-10">
        {/* 附件预览 */}
        {pendingAttachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3 px-1">
            {pendingAttachments.map((att, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1.5 bg-gray-100 px-3 py-1.5 rounded-xl text-xs font-bold text-gray-600"
              >
                {att.mimeType?.startsWith("image/") ? <ImageIcon size={12} /> : <FileText size={12} />}
                <span className="truncate max-w-[140px]">{att.filename}</span>
                <button
                  onClick={() =>
                    setPendingAttachments((prev) => prev.filter((_, i) => i !== idx))
                  }
                  className="ml-1 text-gray-400 hover:text-red-500 transition-colors"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 max-w-3xl mx-auto">
          {/* 文件上传按钮 */}
          <Button
            variant="ghost"
            size="icon"
            className="shrink-0 w-11 h-11 rounded-full bg-gray-50 hover:bg-gray-100 text-gray-400 hover:text-gray-600"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={20} />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.txt,.pdf,.doc,.docx"
            className="hidden"
            onChange={handleFileSelect}
          />

          {/* 输入框 */}
          <div className="flex-1 bg-gray-50 rounded-3xl px-5 py-3 border border-gray-100 focus-within:border-gray-300 focus-within:bg-white transition-all">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder="询问职位，或上传 JD 截图 / 文本..."
              className="w-full bg-transparent text-sm text-gray-600 outline-none font-bold placeholder:font-normal resize-none overflow-hidden leading-relaxed"
              style={{ minHeight: "22px", maxHeight: "120px" }}
            />
          </div>

          {/* 发送按钮 */}
          <Button
            size="icon"
            className={`shrink-0 w-11 h-11 rounded-full shadow-lg transition-all ${
              inputValue.trim() || pendingAttachments.length > 0
                ? "bg-gray-800 text-white shadow-gray-200 hover:scale-105"
                : "bg-gray-100 text-gray-300"
            }`}
            onClick={handleSend}
            disabled={isLoading || (!inputValue.trim() && pendingAttachments.length === 0)}
          >
            <Send size={18} />
          </Button>
        </div>
      </div>
    </div>
  );
}
