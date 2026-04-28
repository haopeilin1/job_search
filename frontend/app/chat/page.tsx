"""
求职雷达 Agent —— 对话页面

路由：/chat
技术栈：Next.js 14 App Router + TypeScript + Tailwind CSS + shadcn/ui + Zustand

本页面为对话主入口，核心交互逻辑封装在 ChatInterface 组件中。
TODO: 后续可扩展为多页面路由（/resume, /knowledge-base 等）。
"""

import ChatInterface from "@/components/ChatInterface";

export const metadata = {
  title: "求职雷达 Agent",
  description: "AI 驱动的求职匹配与面试辅导助手",
};

export default function ChatPage() {
  return <ChatInterface />;
}
