const BASE_URL = "http://localhost:8001";

async function fetchJson(url, options = {}) {
  // 当 body 为 FormData 时，不手动设置 headers，让浏览器自动添加 Content-Type（含 boundary）
  const fetchOptions = { ...options };
  if (!fetchOptions.headers) {
    delete fetchOptions.headers;
  }
  const res = await fetch(`${BASE_URL}${url}`, fetchOptions);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API Error ${res.status}: ${err}`);
  }
  return res.json();
}

// ---------- Knowledge Base ----------
export const apiKB = {
  list: () => fetchJson("/api/v1/knowledge-base/"),
  create: (data) => fetchJson("/api/v1/knowledge-base/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }),
  delete: (id) => fetchJson(`/api/v1/knowledge-base/${id}`, { method: "DELETE" }),
  /** 文本解析 JD */
  parseText: (rawText) => fetchJson("/api/v1/knowledge-base/parse-text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText }),
  }),
  /** 图片解析 JD */
  parseImage: (file) => {
    const form = new FormData();
    form.append("source_image", file);
    return fetchJson("/api/v1/knowledge-base/parse-image", {
      method: "POST",
      body: form,
    });
  },
};

// ---------- Chat ----------
export const apiChat = {
  /** 旧版接口（保留兼容） */
  sendMessage: (message, sessionId = null) => fetchJson("/api/v1/chat/message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  }),
  /** 新版统一对话入口（意图识别 + 路由分发） */
  send: ({ message, sessionId = null, type = "text", attachments = null, context = null }) => fetchJson("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId || "default", message, type, attachments, context }),
  }),
};

// ---------- Resumes (PRD v2) ----------
export const apiResume = {
  /** 获取当前生效简历 */
  getCurrent: () => fetchJson("/api/v1/resumes/current"),
  /** 获取简历列表 */
  list: () => fetchJson("/api/v1/resumes/"),
  /** 上传简历文件 */
  upload: (file, forceUpdate = false, parserMode = 'text') => {
    const form = new FormData();
    form.append("file", file);
    form.append("force_update", String(forceUpdate));
    form.append("parser_mode", parserMode);
    return fetchJson("/api/v1/resumes/upload", {
      method: "POST",
      body: form,
    });
  },
  /** 更新简历解析结果 */
  update: (resumeId, parsedSchema) => fetchJson(`/api/v1/resumes/${resumeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parsed_schema: parsedSchema }),
  }),
  /** 切换生效简历 */
  activate: (resumeId) => fetchJson(`/api/v1/resumes/${resumeId}/activate`, {
    method: "PUT",
  }),
  /** 基于原始文本重新解析简历 */
  reparse: (resumeId, rawText) => {
    const form = new FormData();
    form.append("raw_text", rawText);
    return fetchJson(`/api/v1/resumes/${resumeId}/reparse`, {
      method: "POST",
      body: form,
    });
  },
  /** 上传 JD 进行匹配 */
  match: (file, jdText = "") => {
    const form = new FormData();
    if (file) form.append("file", file);
    form.append("jd_text", jdText);
    return fetchJson("/api/v1/resumes/match", {
      method: "POST",
      body: form,
    });
  },
};

// ---------- Settings ----------
export const apiSettings = {
  getLLM: () => fetchJson("/api/v1/settings/llm"),
  updateLLM: (config) => fetchJson("/api/v1/settings/llm", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  }),
};
