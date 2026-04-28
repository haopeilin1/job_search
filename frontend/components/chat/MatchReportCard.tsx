"use client";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Target,
  Lightbulb,
} from "lucide-react";
import { QuestionItem, QuestionItemData } from "./QuestionItem";

// ──────────────────────────── 类型 ────────────────────────────

export interface MatchReportCardProps {
  score: number; // 0-100
  verdict: string; // 如"高度契合"
  dimensions?: Record<string, number>; // 5维度得分
  gapAnalysis?: {
    strong_matches: string[];
    weak_matches: string[];
    missing: string[];
  };
  questions?: QuestionItemData[];
}

// ──────────────────────────── 辅助函数 ────────────────────────────

function scoreColor(score: number): string {
  if (score >= 80) return "text-emerald-600";
  if (score >= 60) return "text-amber-500";
  return "text-red-500";
}

function scoreBg(score: number): string {
  if (score >= 80) return "bg-emerald-50";
  if (score >= 60) return "bg-amber-50";
  return "bg-red-50";
}

function scoreRingColor(score: number): string {
  if (score >= 80) return "#10b981"; // emerald-500
  if (score >= 60) return "#f59e0b"; // amber-500
  return "#ef4444"; // red-500
}

/**
 * 圆形进度条 SVG
 */
function ScoreRing({ score }: { score: number }) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color = scoreRingColor(score);

  return (
    <div className="relative w-28 h-28 flex items-center justify-center">
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="56"
          cy="56"
          r={radius}
          fill="none"
          stroke="#f1f5f9"
          strokeWidth="10"
        />
        <circle
          cx="56"
          cy="56"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s ease-out" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className={`text-3xl font-black ${scoreColor(score)}`}>
          {score}
        </span>
        <span className="text-[10px] font-black text-gray-400 uppercase tracking-wider">
          分
        </span>
      </div>
    </div>
  );
}

// ──────────────────────────── 组件 ────────────────────────────

export function MatchReportCard({
  score,
  verdict,
  dimensions,
  gapAnalysis,
  questions,
}: MatchReportCardProps) {
  return (
    <Card className="w-full border border-gray-100 shadow-sm rounded-[2.5rem] overflow-hidden bg-white">
      {/* 顶部：分数圆环 + 结论 */}
      <div className="p-6 flex items-center gap-5">
        <ScoreRing score={score} />
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <Target size={16} className="text-gray-500" />
            <span className="text-xs font-black text-gray-400 uppercase tracking-widest">
              单 JD 匹配
            </span>
          </div>
          <h3 className="text-xl font-black text-gray-900">{verdict}</h3>
          <p className="text-xs text-gray-500 font-bold mt-1">
            {score >= 80
              ? "你的背景与该岗位高度匹配，建议优先投递"
              : score >= 60
              ? "基本匹配，但有部分能力需要补充"
              : "匹配度较低，建议慎重考虑或针对性提升"}
          </p>
        </div>
      </div>

      {/* 五维得分雷达（简化展示） */}
      {dimensions && Object.keys(dimensions).length > 0 && (
        <div className="px-6 pb-4">
          <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-3">
            维度分析
          </div>
          <div className="grid grid-cols-5 gap-2">
            {Object.entries(dimensions).map(([key, val]) => (
              <div key={key} className="text-center">
                <div
                  className={`text-lg font-black ${scoreColor(val)}`}
                >
                  {val}
                </div>
                <div className="text-[10px] text-gray-400 font-bold mt-0.5">
                  {key}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gap 分析 */}
      {gapAnalysis && (
        <div className="px-6 pb-4">
          <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-3">
            能力 Gap 分析
          </div>
          <div className="space-y-2">
            {/* 强匹配 */}
            {gapAnalysis.strong_matches.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {gapAnalysis.strong_matches.map((item) => (
                  <Badge
                    key={item}
                    variant="outline"
                    className="bg-emerald-50 text-emerald-700 border-emerald-100 text-[10px] font-black px-2.5 py-1 rounded-full"
                  >
                    <CheckCircle2 size={10} className="mr-1" />
                    {item}
                  </Badge>
                ))}
              </div>
            )}
            {/* 弱匹配 */}
            {gapAnalysis.weak_matches.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {gapAnalysis.weak_matches.map((item) => (
                  <Badge
                    key={item}
                    variant="outline"
                    className="bg-amber-50 text-amber-700 border-amber-100 text-[10px] font-black px-2.5 py-1 rounded-full"
                  >
                    <AlertTriangle size={10} className="mr-1" />
                    {item}
                  </Badge>
                ))}
              </div>
            )}
            {/* 缺失 */}
            {gapAnalysis.missing.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {gapAnalysis.missing.map((item) => (
                  <Badge
                    key={item}
                    variant="outline"
                    className="bg-red-50 text-red-700 border-red-100 text-[10px] font-black px-2.5 py-1 rounded-full"
                  >
                    <XCircle size={10} className="mr-1" />
                    {item}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 面试题列表 */}
      {questions && questions.length > 0 && (
        <div className="px-6 pb-6 pt-2">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb size={14} className="text-gray-500" />
            <span className="text-xs font-black text-gray-500 uppercase tracking-widest">
              面试题预测
            </span>
          </div>
          <div className="space-y-3">
            {questions.map((q) => (
              <QuestionItem key={q.id} data={q} />
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
