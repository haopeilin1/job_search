"""
Agent 工具注册中心 —— BaseTool + ToolRegistry + PlaceholderResolver

职责：
1. 定义所有工具的 input/output schema（供 Planner 查询和参数校验）
2. 提供 BaseTool 抽象基类，统一工具接口
3. PlaceholderResolver：解析 Plan 中的 {{xxx}} 占位符为实际值
"""

import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# 1. 数据模型
# ═══════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """一次工具调用描述（Planner 生成）"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    task_id: int = 0
    depends_on: Optional[List[int]] = None
    # 新增：评测上下文透传
    session_id: Optional[str] = None
    turn_id: Optional[int] = None


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0
    cost_usd: float = 0.0


# ═══════════════════════════════════════════════════════
# 2. BaseTool 抽象基类
# ═══════════════════════════════════════════════════════

class BaseTool(ABC):
    """所有工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict:
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict:
        pass

    @property
    def cost_level(self) -> str:
        return "medium"

    @property
    def avg_latency_ms(self) -> int:
        return 1000

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """执行工具，返回结构化结果"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """基于 input_schema 校验参数，返回缺失的必填字段列表"""
        required = self.input_schema.get("required", [])
        missing = []
        for field_name in required:
            if field_name not in params or params[field_name] is None:
                missing.append(field_name)
        return missing

    def get_param_default(self, param_name: str) -> Any:
        """获取参数默认值"""
        props = self.input_schema.get("properties", {})
        if param_name in props:
            return props[param_name].get("default")
        return None


# ═══════════════════════════════════════════════════════
# 6. 工厂函数（延迟导入避免循环依赖）
# ═══════════════════════════════════════════════════════

def create_tool_registry():
    """创建并注册所有工具的 ToolRegistry 实例"""
    from app.core.tools import create_tool_registry as _factory
    return _factory()


# ═══════════════════════════════════════════════════════
# 3. 工具元信息注册表（schema 定义）
# ═══════════════════════════════════════════════════════

# ── Tool 1: kb_retrieve ──
KB_RETRIEVE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用于检索的query，优先使用search_keywords"
        },
        "company": {
            "type": ["string", "null"],
            "description": "公司名过滤，如'字节跳动'。为null时不过滤",
            "default": None
        },
        "position": {
            "type": ["string", "null"],
            "description": "岗位名过滤，如'算法工程师'。为null时不过滤",
            "default": None
        },
        "attributes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "属性词列表，用于BM25加权或向量增强",
            "default": []
        },
        "retrieval_mode": {
            "type": "string",
            "enum": ["full", "incremental", "reuse"],
            "description": "检索模式：full=全新检索；incremental=增量检索；reuse=直接复用缓存证据",
            "default": "full"
        },
        "evidence_cache": {
            "type": "array",
            "description": "上轮缓存的chunks，仅在incremental/reuse模式下传入",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "score": {"type": "number"},
                    "metadata": {"type": "object"}
                },
                "required": ["id", "content"]
            },
            "default": []
        },
        "top_k": {
            "type": "integer",
            "description": "最终返回的chunk数量",
            "default": 15,
            "minimum": 1,
            "maximum": 50
        }
    },
    "required": ["query"]
}

KB_RETRIEVE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "score": {"type": "number"},
                    "metadata": {"type": "object"}
                },
                "required": ["id", "content", "score", "metadata"]
            }
        },
        "query_used": {"type": "string"},
        "source": {
            "type": "string",
            "enum": ["vector", "bm25", "hybrid", "cache_reuse", "incremental_merge"]
        },
        "total_found": {"type": "integer"},
        "rerank_top_k": {"type": "integer"}
    },
    "required": ["success", "chunks", "query_used", "source"]
}

# ── Tool 2: match_analyze ──
MATCH_ANALYZE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "resume_text": {
            "type": "string",
            "description": "用户当前生效简历的完整文本或摘要"
        },
        "jd_text": {
            "type": "string",
            "description": "岗位描述文本（可从kb_retrieve结果或用户输入获取）"
        },
        "company": {
            "type": "string",
            "description": "公司名称（可选）"
        },
        "position": {
            "type": "string",
            "description": "岗位名称（可选）"
        }
    },
    "required": ["resume_text", "jd_text"]
}

MATCH_ANALYZE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "match_score": {"type": "number"},
        "dimensions": {"type": "array", "items": {"type": "object"}},
        "advantages": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "interview_focus": {"type": "array", "items": {"type": "string"}},
        "jd_summary": {"type": "string"}
    },
    "required": ["success", "match_score", "dimensions", "gaps", "jd_summary"]
}

# ── Tool 3: global_rank ──
GLOBAL_RANK_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "resume_text": {"type": "string", "description": "用户简历文本"},
        "candidate_jds": {
            "type": "array",
            "description": "来自kb_retrieve的多条JD",
            "items": {
                "type": "object",
                "properties": {
                    "jd_id": {"type": "string"},
                    "company": {"type": "string"},
                    "position": {"type": "string"},
                    "chunks": {"type": "array"},
                    "salary": {"type": ["string", "null"]}
                },
                "required": ["jd_id", "company", "position", "chunks"]
            }
        },
        "filters": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "location": {"type": ["string", "null"]},
                "experience_years": {"type": ["string", "null"]}
            },
            "default": {}
        },
        "sort_by": {
            "type": "string",
            "enum": ["match_score", "salary", "company_size", "growth_potential"],
            "default": "match_score"
        },
        "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
    },
    "required": ["resume_text", "candidate_jds"]
}

GLOBAL_RANK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "jd_id": {"type": "string"},
                    "company": {"type": "string"},
                    "position": {"type": "string"},
                    "match_score": {"type": "number"},
                    "recommend_reason": {"type": "string"},
                    "key_match": {"type": "array", "items": {"type": "string"}},
                    "key_gap": {"type": "array", "items": {"type": "string"}},
                    "apply_priority": {"type": "string", "enum": ["高", "中", "低"]}
                },
                "required": ["rank", "company", "position", "match_score", "apply_priority"]
            }
        },
        "strategy_advice": {"type": "string"}
    },
    "required": ["success", "rankings"]
}

# ── Tool 4: qa_synthesize ──
QA_SYNTHESIZE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "用户原始问题"},
        "rewritten_question": {"type": "string", "description": "改写后的标准问题"},
        "evidence_chunks": {
            "type": "array",
            "description": "来自kb_retrieve的证据chunks",
            "items": {"type": "object"}
        },
        "qa_type": {
            "type": "string",
            "enum": ["factual", "comparative", "temporal", "definition"]
        },
        "conversation_history": {"type": "string", "default": ""},
        "insufficient_behavior": {
            "type": "string",
            "enum": ["admit", "infer"],
            "default": "admit"
        }
    },
    "required": ["question", "evidence_chunks", "qa_type"]
}

QA_SYNTHESIZE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "object"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "insufficient_note": {"type": ["string", "null"]}
    },
    "required": ["success", "answer", "citations", "confidence"]
}

# ── Tool 5: interview_gen ──
INTERVIEW_GEN_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "match_result": {
            "type": "object",
            "description": "上游match_analyze的输出",
            "properties": {
                "gaps": {"type": "array", "items": {"type": "string"}},
                "interview_focus": {"type": "array", "items": {"type": "string"}},
                "jd_summary": {"type": "string"},
                "dimensions": {"type": "array"}
            },
            "required": ["gaps", "interview_focus", "jd_summary"]
        },
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard", "mixed"],
            "default": "mixed"
        },
        "count": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        "include_follow_up": {"type": "boolean", "default": True},
        "focus_area": {
            "type": "string",
            "enum": ["gap", "strength", "general"],
            "default": "gap"
        }
    },
    "required": ["match_result"]
}

INTERVIEW_GEN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "category": {"type": "string", "enum": ["技术", "项目", "场景", "压力", "价值观"]},
                    "question": {"type": "string"},
                    "target_gap": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "follow_ups": {"type": "array", "items": {"type": "string"}},
                    "reference_evidence": {"type": "string"}
                },
                "required": ["id", "category", "question", "difficulty"]
            }
        },
        "overall_assessment": {"type": "string"}
    },
    "required": ["success", "questions"]
}

# ── Tool 6: file_ops ──
FILE_OPS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["upload_jd", "upload_jd_image", "delete_jd", "update_resume", "list_jds", "list_resumes"]
        },
        "file_data": {
            "type": ["string", "null"],
            "description": "文件内容：base64编码的图片/PDF，或文件ID"
        },
        "text_data": {
            "type": ["string", "null"],
            "description": "用户粘贴的文本内容，与file_data二选一"
        },
        "target_id": {
            "type": ["string", "null"],
            "description": "删除/更新时的目标文件ID"
        },
        "parse_options": {
            "type": "object",
            "properties": {
                "ocr_backend": {"type": "string", "enum": ["vision_llm", "tesseract", "rapidocr"], "default": "vision_llm"},
                "extract_structured": {"type": "boolean", "default": True},
                "target_type": {"type": "string", "enum": ["resume", "jd"]}
            },
            "default": {"ocr_backend": "vision_llm", "extract_structured": True}
        }
    },
    "required": ["operation"]
}

FILE_OPS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "operation": {"type": "string"},
        "file_id": {"type": ["string", "null"]},
        "extracted_data": {
            "type": "object",
            "properties": {
                "structured": {"type": "object"},
                "raw_text": {"type": "string"}
            }
        },
        "message": {"type": "string"}
    },
    "required": ["success", "operation", "message"]
}

# ── Tool 7: general_chat ──
GENERAL_CHAT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_message": {"type": "string", "description": "用户原始消息"},
        "chat_type": {
            "type": "string",
            "enum": ["greeting", "career_advice", "industry", "how_to", "other"],
            "description": "对话类型"
        },
        "user_profile": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "experience": {"type": ["string", "null"]},
                "preferences": {"type": "object"}
            },
            "default": {}
        },
        "conversation_history": {"type": "string", "description": "最近3轮对话摘要", "default": ""}
    },
    "required": ["user_message"]
}

GENERAL_CHAT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "response": {"type": "string"},
        "suggested_topics": {"type": "array", "items": {"type": ["string", "null"]}}
    },
    "required": ["success", "response"]
}

# ── Tool 8: evidence_relevance_check ──
EVIDENCE_RELEVANCE_CHECK_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用户当前查询"
        },
        "evidence_chunks": {
            "type": "array",
            "description": "缓存的证据chunks",
            "items": {"type": "object"},
            "default": []
        }
    },
    "required": ["query", "evidence_chunks"]
}

EVIDENCE_RELEVANCE_CHECK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "relevant": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "chunks_checked": {"type": "integer"}
    },
    "required": ["success", "relevant", "confidence"]
}


# 工具元信息总表（供 Planner 查询）
TOOL_REGISTRY_META = {
    "kb_retrieve": {
        "description": "知识库混合检索：向量+BM25召回，CrossEncoder重排。支持多轮检索策略（全量/增量/复用）。",
        "input_schema": KB_RETRIEVE_INPUT_SCHEMA,
        "output_schema": KB_RETRIEVE_OUTPUT_SCHEMA,
        "cost_level": "medium",
        "avg_latency_ms": 800,
        "required_slots": ["query"],
        "optional_slots": ["company", "position", "attributes"]
    },
    "match_analyze": {
        "description": "单JD匹配分析：多维度评分、优势短板、面试关注点。支持知识库/附件/文本三种JD来源。",
        "input_schema": MATCH_ANALYZE_INPUT_SCHEMA,
        "output_schema": MATCH_ANALYZE_OUTPUT_SCHEMA,
        "cost_level": "high",
        "avg_latency_ms": 1500,
        "required_slots": ["resume_text", "jd_text"],
        "optional_slots": ["company", "position"]
    },
    "global_rank": {
        "description": "全局匹配排序：简历 vs 多JD批量对比，按匹配度排序并给出投递策略。",
        "input_schema": GLOBAL_RANK_INPUT_SCHEMA,
        "output_schema": GLOBAL_RANK_OUTPUT_SCHEMA,
        "cost_level": "high",
        "avg_latency_ms": 2000,
        "required_slots": ["resume_text"],
        "optional_slots": ["filters", "sort_by", "top_k"]
    },
    "qa_synthesize": {
        "description": "问答综合：基于检索证据回答用户问题，必须引用来源，证据不足时明确告知。",
        "input_schema": QA_SYNTHESIZE_INPUT_SCHEMA,
        "output_schema": QA_SYNTHESIZE_OUTPUT_SCHEMA,
        "cost_level": "medium",
        "avg_latency_ms": 1200,
        "required_slots": ["question", "evidence_chunks"],
        "optional_slots": ["qa_type"]
    },
    "interview_gen": {
        "description": "面试题生成：基于匹配短板和JD要求生成深度面试题，含追问方向。",
        "input_schema": INTERVIEW_GEN_INPUT_SCHEMA,
        "output_schema": INTERVIEW_GEN_OUTPUT_SCHEMA,
        "cost_level": "medium",
        "avg_latency_ms": 1000,
        "required_slots": ["match_result"],
        "optional_slots": ["count", "difficulty", "focus_area"]
    },
    "file_ops": {
        "description": "资料管理：简历/JD的上传、OCR解析、删除、更新、列表查询。",
        "input_schema": FILE_OPS_INPUT_SCHEMA,
        "output_schema": FILE_OPS_OUTPUT_SCHEMA,
        "cost_level": "low",
        "avg_latency_ms": 500,
        "required_slots": ["operation"],
        "optional_slots": ["file_data", "text_data", "target_id"]
    },
    "general_chat": {
        "description": "通用对话：问候、职业规划、行业咨询等无需检索的对话。",
        "input_schema": GENERAL_CHAT_INPUT_SCHEMA,
        "output_schema": GENERAL_CHAT_OUTPUT_SCHEMA,
        "cost_level": "low",
        "avg_latency_ms": 300,
        "required_slots": ["user_message"],
        "optional_slots": ["chat_type", "user_profile"]
    },
    "evidence_relevance_check": {
        "description": "证据相关性检查：判断缓存证据是否与用户查询相关。",
        "input_schema": EVIDENCE_RELEVANCE_CHECK_INPUT_SCHEMA,
        "output_schema": EVIDENCE_RELEVANCE_CHECK_OUTPUT_SCHEMA,
        "cost_level": "low",
        "avg_latency_ms": 300,
        "required_slots": ["query", "evidence_chunks"],
        "optional_slots": []
    }
}


# ═══════════════════════════════════════════════════════
# 4. ToolRegistry 注册中心
# ═══════════════════════════════════════════════════════

class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.info(f"[ToolRegistry] 注册工具: {tool.name}")

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_schema(self, name: str) -> Optional[Dict]:
        return TOOL_REGISTRY_META.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> Dict[str, Dict]:
        """返回所有工具的 schema，供 Planner 使用"""
        result = {}
        for name, tool in self._tools.items():
            meta = TOOL_REGISTRY_META.get(name, {})
            result[name] = {
                "name": name,
                "description": meta.get("description", ""),
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "cost_level": meta.get("cost_level", "medium"),
                "avg_latency_ms": meta.get("avg_latency_ms", 1000),
                "required_slots": meta.get("required_slots", []),
                "optional_slots": meta.get("optional_slots", []),
            }
        return result

    def validate_plan(self, plan: List[ToolCall]) -> List[str]:
        """校验整个 Plan 的工具名和参数合法性"""
        errors = []
        for task in plan:
            tool = self._tools.get(task.name)
            if not tool:
                errors.append(f"未知工具: {task.name}")
                continue
            missing = tool.validate_params(task.params)
            if missing:
                errors.append(f"工具 {task.name}(task_{task.task_id}) 缺失必填参数: {missing}")
        return errors

    async def execute(self, call: ToolCall) -> ToolResult:
        """执行单个工具调用（增加计时、成本估算、埋点）"""
        import time
        from app.core.cost_estimator import estimate_cost, estimate_tokens
        from app.core.telemetry import create_tracker

        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(success=False, error=f"未知工具: {call.name}")

        start_ts = time.time()
        input_text = ""
        try:
            input_text = json.dumps(call.params, ensure_ascii=False)
        except Exception:
            pass
        input_tokens = estimate_tokens(input_text)

        try:
            result = await tool.execute(call.params)

            # 成本估算
            output_text = ""
            try:
                output_text = json.dumps(result.data, ensure_ascii=False) if result.data else ""
            except Exception:
                pass
            output_tokens = estimate_tokens(output_text)
            cost = estimate_cost("default", input_tokens, output_tokens)

            latency_ms = int((time.time() - start_ts) * 1000)
            result.latency_ms = latency_ms
            result.cost_usd = cost

            # 提取粗筛数据（global_rank 工具）
            coarse_filter_meta = {}
            if call.name == "global_rank" and result.data:
                coarse_filter_meta = result.data.get("_coarse_filter_meta", {})

            # tool_executed 埋点
            if call.session_id:
                tracker = create_tracker(session_id=call.session_id, turn_id=call.turn_id)
                tracker.track("tool_executed", {
                    "session_id": call.session_id,
                    "turn_id": call.turn_id,
                    "plan_id": f"{call.session_id}#{call.turn_id}" if call.session_id and call.turn_id else None,
                    "tool_name": call.name,
                    "task_id": call.task_id,
                    "status": "success" if result.success else "failed",
                    "latency_ms": latency_ms,
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "has_evidence": call.name == "qa_synthesize" and bool(result.data.get("answer") if result.data else False),
                    "evidence_count": len(result.data.get("chunks", [])) if result.data and isinstance(result.data, dict) else 0,
                    "coarse_filter": coarse_filter_meta,
                })

            return result

        except Exception as e:
            import traceback
            latency_ms = int((time.time() - start_ts) * 1000)
            logger.error(f"[ToolRegistry] 工具 {call.name} 执行失败: {e}")
            logger.error(f"[ToolRegistry] 详细 traceback:\n{traceback.format_exc()}")

            # tool_executed 埋点（失败）
            if call.session_id:
                tracker = create_tracker(session_id=call.session_id, turn_id=call.turn_id)
                tracker.track("tool_executed", {
                    "session_id": call.session_id,
                    "turn_id": call.turn_id,
                    "tool_name": call.name,
                    "task_id": call.task_id,
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "cost_usd": 0.0,
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "error": str(e),
                })

            return ToolResult(success=False, error=str(e), latency_ms=latency_ms)


# ═══════════════════════════════════════════════════════
# 5. PlaceholderResolver 占位符解析器
# ═══════════════════════════════════════════════════════

PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\.?([\w.\[\]]*)\}\}")


class PlaceholderResolver:
    """
    解析 Plan 中工具参数里的占位符为实际值。

    支持的占位符语法：
    1. 系统变量（session 级别）：
       {{user_resume}}        → session.resume_text
       {{evidence_cache}}     → session.evidence_cache
       {{working_history}}    → session.working_memory.get_recent_context(3)
       {{compressed_history}} → session.compressed_memories
       {{long_term_profile}}  → session.long_term
       {{original_message}}   → session.original_message
       {{rewritten_query}}    → session.rewritten_query

    2. 任务引用（上游工具输出）：
       {{task_0.chunks}}      → observations[0]["chunks"]
       {{task_0.result.chunks}} → observations[0]["result"]["chunks"]（兼容写法）

    3. 附件变量：
       {{attachment_base64}}  → session.attachments[0].base64
       {{attachment_parsed}}  → session.attachments[0].parsed_data
    """

    def __init__(self, session, observations: Optional[Dict[int, Dict]] = None):
        self.session = session
        self.observations = observations or {}

    def resolve(self, value: Any) -> Any:
        """
        递归解析值中的占位符。
        支持字符串、字典、列表。
        """
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(item) for item in value]
        return value

    def _resolve_string(self, text: str) -> Any:
        """解析字符串中的占位符"""
        if not text or "{{" not in text:
            return text

        # 纯占位符（整个字符串就是一个占位符）→ 返回原始值（可能是对象/列表）
        match = PLACEHOLDER_PATTERN.fullmatch(text.strip())
        if match:
            return self._get_value(match.group(1), match.group(2))

        # 字符串中嵌入占位符 → 替换为字符串表示
        def replacer(m):
            val = self._get_value(m.group(1), m.group(2))
            if val is None:
                return ""
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return str(val)

        return PLACEHOLDER_PATTERN.sub(replacer, text)

    def _get_value(self, namespace: str, path: str) -> Any:
        """根据命名空间和路径获取值"""
        # 系统变量
        if namespace == "user_resume":
            return getattr(self.session, "resume_text", "")
        if namespace == "evidence_cache":
            return getattr(self.session, "evidence_cache", [])
        if namespace == "working_history":
            if hasattr(self.session, "working_memory") and self.session.working_memory.turns:
                return self.session.working_memory.get_recent_context(3)
            return ""
        if namespace == "compressed_history":
            if hasattr(self.session, "compressed_memories"):
                cms = self.session.compressed_memories[-5:]
                return "\n".join(f"{cm.summary[:100]}" for cm in cms)
            return ""
        if namespace == "long_term_profile":
            lt = getattr(self.session, "long_term", None)
            if lt:
                parts = []
                if lt.entities:
                    for k, v in lt.entities.items():
                        parts.append(f"{k}: {v}")
                if lt.preferences:
                    for k, v in lt.preferences.items():
                        parts.append(f"{k}: {v}")
                return "; ".join(parts)
            return ""
        if namespace == "original_message":
            return getattr(self.session, "original_message", "")
        if namespace == "rewritten_query":
            return getattr(self.session, "rewritten_query", "")
        if namespace == "search_keywords":
            return getattr(self.session, "search_keywords", "")

        # 附件变量
        if namespace == "attachment_base64":
            atts = getattr(self.session, "attachments", [])
            return atts[0].get("data", "") if atts else ""
        if namespace == "attachment_parsed":
            atts = getattr(self.session, "attachments", [])
            return atts[0].get("parsed_data", "") if atts else ""
        if namespace == "user_pasted_text":
            return getattr(self.session, "pasted_text", "")

        # 任务引用：task_0.chunks
        if namespace.startswith("task_"):
            task_id = int(namespace.replace("task_", ""))
            obs = self.observations.get(task_id, {})
            if not path:
                return obs
            # 处理 "result.chunks" 或 "chunks"
            keys = path.split(".")
            val = obs
            for key in keys:
                if isinstance(val, dict) and key in val:
                    val = val[key]
                else:
                    return None
            return val

        return None
