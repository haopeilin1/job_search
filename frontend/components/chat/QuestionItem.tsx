"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Target, Zap, Flame } from "lucide-react";
import { Badge } from "@/components/ui/badge";

// ──────────────────────────── 类型 ────────────────────────────

export interface FollowUp {
  question: string;
  intent: string; // 追问意图说明，如"验证量化真实性"
}

export interface QuestionItemData {
  id: number;
  question: string;
  type: string; // 深挖 / 压力 / 拔高
  difficulty: string; // 简单 / 中等 / 困难
  follow_ups: FollowUp[];
  source_jd_item?: string; // 来源 JD 要求
}

interface QuestionItemProps {
  data: QuestionItemData;
}

// ──────────────────────────── 辅助 ────────────────────────────

function difficultyColor(difficulty: string): string {
  switch (difficulty) {
    case "困难":
      return "bg-red-50 text-red-600 border-red-100";
    case "中等":
      return "bg-amber-50 text-amber-600 border-amber-100";
    default:
      return "bg-green-50 text-green-600 border-green-100";
  }
}

function typeIcon(type: string) {
  switch (type) {
    case "深挖":
      return <Target size={12} />;
    case "压力":
      return <Zap size={12} />;
    case "拔高":
      return <Flame size={12} />;
    default:
      return null;
  }
}

// ──────────────────────────── 组件 ────────────────────────────

export function QuestionItem({ data }: QuestionItemProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-100 rounded-2xl overflow-hidden bg-white">
      {/* 头部：首问 + 标签 + 展开按钮 */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
      >
        <div className="flex-1 pr-3">
          <div className="flex items-center gap-2 mb-1.5">
            <Badge
              variant="outline"
              className="text-[10px] font-black tracking-wider uppercase px-2 py-0.5 rounded-lg"
            >
              <span className="mr-1">{typeIcon(data.type)}</span>
              {data.type}
            </Badge>
            <Badge
              variant="outline"
              className={`text-[10px] font-black px-2 py-0.5 rounded-lg ${difficultyColor(
                data.difficulty
              )}`}
            >
              {data.difficulty}
            </Badge>
          </div>
          <p className="text-sm font-bold text-gray-800 leading-relaxed">
            {data.question}
          </p>
        </div>
        <div className="shrink-0 text-gray-400">
          {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </button>

      {/* 展开内容：考察点 + 追问列表 */}
      {expanded && (
        <div className="px-4 pb-4 pt-0 border-t border-gray-50">
          {/* 追问列表 */}
          {data.follow_ups.length > 0 && (
            <div className="mt-3 space-y-2">
              <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest">
                追问方向
              </div>
              {data.follow_ups.map((fu, idx) => (
                <div
                  key={idx}
                  className="bg-gray-50 rounded-xl p-3 text-xs text-gray-600 font-bold leading-relaxed"
                >
                  <span className="text-gray-800">{fu.question}</span>
                  <span className="text-gray-400 ml-2">// {fu.intent}</span>
                </div>
              ))}
            </div>
          )}

          {/* 来源 */}
          {data.source_jd_item && (
            <div className="mt-3 text-[10px] text-gray-400 font-bold">
              来源：基于 JD 要求「{data.source_jd_item}」
            </div>
          )}
        </div>
      )}
    </div>
  );
}
