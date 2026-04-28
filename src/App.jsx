import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { 
  BookOpen, 
  Plus, 
  Trash2, 
  Send, 
  FileText, 
  Image as ImageIcon, 
  ChevronLeft, 
  BrainCircuit, 
  Zap, 
  Sparkles, 
  FileSearch, 
  Sun,
  Database,
  X,
  MapPin,
  Briefcase,
  Loader2,
  ChevronRight,
  MessageSquare,
  AlertCircle,
  CheckCircle2,
  Settings,
  Upload
} from 'lucide-react';
import { apiKB, apiChat, apiResume, apiSettings } from './api/client';

const App = () => {
  // ---------------- 全局导航与视图 ----------------
  const [activeTab, setActiveTab] = useState('chat');
  const [currentView, setCurrentView] = useState('chat');

  useEffect(() => {
    if (activeTab === 'chat') setCurrentView('chat');
    if (activeTab === 'resume') setCurrentView('resume');
  }, [activeTab]);

  // ---------------- 简历状态 ----------------
  const [resumeText, setResumeText] = useState('');
  const [hasResume, setHasResume] = useState(false);

  // ---------------- 知识库状态 ----------------
  const [kbItems, setKbItems] = useState([]);
  const [selectedJD, setSelectedJD] = useState(null); 
  const [expandedCompany, setExpandedCompany] = useState(null); 
  const [isAddingJD, setIsAddingJD] = useState(false); 
  const [newJDText, setNewJDText] = useState(''); 
  const [isProcessing, setIsProcessing] = useState(false);
  // JD 添加流程状态
  const [jdAddStep, setJdAddStep] = useState('input'); // 'input' | 'parsing' | 'preview'
  const [jdAddMode, setJdAddMode] = useState('text'); // 'text' | 'image'
  const [jdImageFile, setJdImageFile] = useState(null);
  const [jdPreview, setJdPreview] = useState(null); // 解析后的结构化 JD
  // 简历页面状态（提升到 App 顶层，避免 ResumeView 组件重建导致输入丢失）
  const resumeFileRef = useRef(null);
  const resumeParserModeRef = useRef('text');
  const [resumeSchema, setResumeSchema] = useState(null);
  const [resumeId, setResumeId] = useState('');
  const [parseStatus, setParseStatus] = useState('idle'); // idle | parsing | parsed | error
  const [showRawText, setShowRawText] = useState(false);

  // ---------------- 对话状态 ----------------
  const [messages, setMessages] = useState([
    { role: 'bot', text: '嗨！我是你的 AI 助手小橘。\n\n你可以问我知识库里的职位信息，也可以直接上传 JD 截图或文本，我会立刻结合你的简历进行匹配分析！' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isMatching, setIsMatching] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ---------------- LLM 配置状态 ----------------
  const [showLLMConfig, setShowLLMConfig] = useState(false);
  const [llmConfig, setLlmConfig] = useState({
    chat: { baseUrl: '', apiKey: '', model: '' },
    core: { baseUrl: '', apiKey: '', model: '' },
    planner: { baseUrl: '', apiKey: '', model: '' },
    memory: { baseUrl: '', apiKey: '', model: '' },
    vision: { baseUrl: '', apiKey: '', model: '' }
  });

  // ---------------- 初始化：从后端加载数据 ----------------
  useEffect(() => {
    // 加载知识库
    apiKB.list().then(data => {
      setKbItems(data.map(item => ({ ...item, desc: item.description })));
    }).catch(err => console.error('KB load failed:', err));

    // 加载简历
    apiResume.getCurrent().then(data => {
      if (data.resume_id) {
        setHasResume(true);
        setResumeSchema(data.parsed_schema);
        setResumeId(data.resume_id);
        if (data.parsed_schema?.meta?.raw_text) {
          setResumeText(data.parsed_schema.meta.raw_text);
        }
      }
    }).catch(err => console.error('Resume load failed:', err));

    // 加载 LLM 配置
    apiSettings.getLLM().then(data => {
      setLlmConfig(data);
    }).catch(err => console.error('Settings load failed:', err));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ---------------- 核心逻辑 ----------------
  const groupedJDs = useMemo(() => {
    return kbItems.reduce((acc, item) => {
      if (!acc[item.company]) acc[item.company] = [];
      acc[item.company].push(item);
      return acc;
    }, {});
  }, [kbItems]);

  const handleSendMessage = async () => {
    const el = textareaRef.current;
    if (!el) return;
    const text = el.value.trim();
    if (!text) return;
    setMessages(prev => [...prev, { role: 'user', text }]);
    el.value = '';
    setInputValue('');

    try {
      const res = await apiChat.send({ message: text });
      const reply = res.reply;
      const routeMeta = res.route_meta;
      const botMsgBase = { role: 'bot', routeMeta };
      // 根据意图类型渲染不同内容
      if (reply.type === 'match_report') {
        setMessages(prev => [...prev, { ...botMsgBase, type: 'match_result', data: reply.data, text: reply.content }]);
      } else if (reply.type === 'global_ranking') {
        setMessages(prev => [...prev, { ...botMsgBase, type: 'global_ranking', data: reply.data, text: reply.content }]);
      } else if (reply.type === 'rag_answer') {
        setMessages(prev => [...prev, { ...botMsgBase, type: 'rag_answer', text: reply.content, sources: reply.sources }]);
      } else {
        setMessages(prev => [...prev, { ...botMsgBase, text: reply.content }]);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'bot', text: '❌ 网络错误，请稍后重试。' }]);
    }
  };

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  const handleJDUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    const isImage = file.type.startsWith('image/');
    const fileType = isImage ? 'image' : 'file';

    if (!hasResume) {
      setMessages(prev => [...prev, 
        { role: 'user', type: fileType, text: file.name },
        { role: 'bot', text: '⚠️ 你还没有上传简历哦！请先点击底部「我的简历」完善信息，我才能帮你做精准匹配。' }
      ]);
      return;
    }

    // 生成图片预览 URL
    const imagePreviewUrl = isImage ? URL.createObjectURL(file) : null;
    setMessages(prev => [...prev, { role: 'user', type: fileType, text: file.name, imageUrl: imagePreviewUrl }]);
    setIsMatching(true);
    setMessages(prev => [...prev, { role: 'bot', type: 'typing' }]);

    try {
      // 图片转 base64 传给后端 OCR
      let attachments = null;
      if (isImage) {
        const base64 = await fileToBase64(file);
        attachments = [{ filename: file.name, content_type: file.type, data: base64 }];
      }
      const res = await apiChat.send({
        message: '请分析这个JD与我的简历匹配度',
        type: fileType,
        attachments
      });
      setIsMatching(false);
      const reply = res.reply;
      const routeMeta = res.route_meta;
      const botMsgBase = { role: 'bot', routeMeta };
      setMessages(prev => {
        const filtered = prev.filter(m => m.type !== 'typing');
        if (reply.type === 'match_report') {
          return [...filtered, { ...botMsgBase, type: 'match_result', data: reply.data, text: reply.content }];
        }
        return [...filtered, { ...botMsgBase, text: reply.content }];
      });
    } catch (err) {
      console.error(err);
      setIsMatching(false);
      setMessages(prev => {
        const filtered = prev.filter(m => m.type !== 'typing');
        return [...filtered, { role: 'bot', text: '❌ 匹配分析失败，请检查网络后重试。' }];
      });
    }
  };

  const handleResumeUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';
    setParseStatus('parsing');

    try {
      const result = await apiResume.upload(file, false, resumeParserModeRef.current);
      const rawText = result.parsed_schema?.meta?.raw_text || '';
      setResumeText(rawText);
      setResumeSchema(result.parsed_schema);
      setResumeId(result.resume_id);
      setHasResume(true);
      setParseStatus('parsed');
    } catch (err) {
      console.error(err);
      setParseStatus('error');
      alert('简历上传失败，请重试');
    }
  };

  const triggerResumeUpload = (mode) => {
    resumeParserModeRef.current = mode;
    if (resumeFileRef.current) {
      resumeFileRef.current.accept = mode === 'vision' ? '.pdf' : '.txt,.pdf,.doc,.docx';
      resumeFileRef.current.click();
    }
  };

  // ---------- JD 添加流程（文本粘贴） ----------
  const handleParseJDText = async () => {
    if (!newJDText.trim()) return;
    setJdAddStep('parsing');
    try {
      const result = await apiKB.parseText(newJDText.trim());
      setJdPreview(result.parsed_schema || result);
      setJdAddStep('preview');
    } catch (err) {
      console.error(err);
      alert('JD 解析失败：' + (err.message || '请检查 LLM 配置'));
      setJdAddStep('input');
    }
  };

  // ---------- JD 添加流程（图片上传） ----------
  const handleParseJDImage = async () => {
    if (!jdImageFile) return;
    setJdAddStep('parsing');
    try {
      const result = await apiKB.parseImage(jdImageFile);
      setJdPreview(result.parsed_schema || result);
      setJdAddStep('preview');
    } catch (err) {
      console.error(err);
      alert('JD 图片解析失败：' + (err.message || '请检查多模态 LLM 配置'));
      setJdAddStep('input');
    }
  };

  // ---------- 确认存入知识库 ----------
  const handleConfirmAddJD = async () => {
    if (!jdPreview) return;
    setIsProcessing(true);
    try {
      const payload = {
        company: jdPreview.company || '未知公司',
        title: jdPreview.position || jdPreview.title || '未知岗位',
        position: jdPreview.position || jdPreview.title || '未知岗位',
        description: jdPreview.sections?.responsibilities || jdPreview.raw_text || '',
        location: jdPreview.location || '远程',
        salary: jdPreview.salary_range || '薪资面议',
        salary_range: jdPreview.salary_range || '薪资面议',
        color: 'bg-gray-100 text-gray-600',
        sections: jdPreview.sections || { responsibilities: jdPreview.raw_text || '', hard_requirements: [], soft_requirements: [] },
        keywords: jdPreview.keywords || [],
        raw_text: jdPreview.raw_text || '',
        meta: jdPreview.meta || { source_type: 'paste', created_at: new Date().toISOString(), updated_at: new Date().toISOString(), chunk_ids: [] },
      };
      const result = await apiKB.create(payload);
      setKbItems(prev => [{ ...result, desc: result.description }, ...prev]);
      setIsProcessing(false);
      closeJDAddFlow();
      const indexedMsg = result.vector_indexed
        ? `✅ 已存入知识库并向量化：「${payload.company}」· ${payload.title}`
        : `⚠️ 已存入知识库，但向量化失败：「${payload.company}」· ${payload.title}`;
      setMessages(prev => [...prev, { role: 'bot', text: indexedMsg }]);
    } catch (err) {
      console.error(err);
      setIsProcessing(false);
      alert('存入知识库失败，请重试');
    }
  };

  // ---------- 关闭添加流程并重置状态 ----------
  const closeJDAddFlow = () => {
    setIsAddingJD(false);
    setJdAddStep('input');
    setJdAddMode('text');
    setNewJDText('');
    setJdImageFile(null);
    setJdPreview(null);
    setIsProcessing(false);
  };

  // ---------------- 子组件：匹配结果卡片 ----------------
  const RouteBadge = ({ meta }) => {
    if (!meta) return null;
    const intentLabels = {
      match_single: '匹配分析',
      global_match: '全局对比',
      rag_qa: '知识问答',
      general: '通用对话'
    };
    const layerLabels = {
      rule: '规则层',
      llm: 'LLM层',
      llm_fallback: 'LLM降级'
    };
    const intent = meta.intent || 'general';
    const layer = meta.layer || 'rule';
    const confidence = meta.confidence !== undefined ? meta.confidence : 1.0;
    return (
      <div className="flex items-center gap-1.5 mb-1 ml-1">
        <span className="text-[9px] font-black px-2 py-0.5 rounded-full bg-gray-200 text-gray-600 uppercase tracking-wider">
          {intentLabels[intent] || intent}
        </span>
        <span className="text-[9px] font-bold text-gray-400">
          {layerLabels[layer] || layer} · {(confidence * 100).toFixed(0)}%
        </span>
      </div>
    );
  };

  const MatchResultCard = ({ data }) => (
    <div className="w-full space-y-5 animate-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
        <h3 className="text-gray-400 text-[10px] font-black tracking-widest uppercase mb-4 text-center">Match Score</h3>
        <div className="relative flex items-center justify-center mb-6">
          <svg className="w-44 h-44 transform -rotate-90">
            <circle cx="88" cy="88" r="72" fill="none" stroke="#F1F5F9" strokeWidth="14" />
            <circle cx="88" cy="88" r="72" fill="none" stroke="#475569" strokeWidth="14" strokeDasharray="452" strokeDashoffset="54" strokeLinecap="round" />
          </svg>
          <div className="absolute flex flex-col items-center">
            <span className="text-5xl font-black text-gray-800 tracking-tighter">{data.score}</span>
            <span className="text-xs font-black text-gray-500 mt-1">{data.label}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-50 p-4 rounded-3xl border border-gray-100">
            <div className="text-[10px] text-gray-500 font-black mb-1">关键优势</div>
            <div className="text-xs font-bold text-gray-800">{data.advantage}</div>
          </div>
          <div className="bg-gray-100 p-4 rounded-3xl border border-gray-200">
            <div className="text-[10px] text-gray-500 font-black mb-1">待补充</div>
            <div className="text-xs font-bold text-gray-800">{data.weakness}</div>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 p-6 rounded-[2.5rem] shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-gray-500/10 rounded-full blur-3xl"></div>
        <div className="flex items-center gap-3 mb-6 relative z-10">
          <div className="w-10 h-10 bg-gray-700 rounded-2xl flex items-center justify-center text-white">
            <Zap size={20} fill="currentColor" />
          </div>
          <h4 className="text-white text-lg font-black tracking-tight">专属面试题预测</h4>
        </div>
        <div className="space-y-5 relative z-10">
          {data.questions.map((q, idx) => (
            <div key={idx} className="space-y-2">
              <div className="text-[10px] font-black uppercase tracking-widest text-gray-400">{q.tag}</div>
              <p className="text-gray-300 text-sm font-bold leading-relaxed">「{q.text}」</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // ---------------- 视图组件 ----------------

  const chatViewJSX = (
    <div className="flex flex-col h-full bg-[#FAFAFA] relative overflow-hidden pb-24">
      {/* 顶部 Header */}
      <div className="px-6 pt-14 pb-3 flex items-center justify-between relative z-10">
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
          <button 
            onClick={() => setShowLLMConfig(true)}
            className="w-10 h-10 rounded-2xl bg-white shadow-sm flex items-center justify-center text-gray-500 border border-gray-100 hover:scale-105 transition-transform"
            title="LLM 配置"
          >
            <Settings size={18} />
          </button>
          <button 
            onClick={() => setCurrentView('kb')}
            className="w-10 h-10 rounded-2xl bg-white shadow-sm flex items-center justify-center text-gray-600 border border-gray-100 hover:scale-105 transition-transform"
            title="知识库"
          >
            <BookOpen size={18} />
          </button>
        </div>
      </div>

      {/* 消息区域 */}
      <div className="flex-1 overflow-y-auto px-6 pb-4 space-y-5 relative z-10">
        {messages.map((m, i) => {
          if (m.type === 'match_result') {
            return (
              <div key={i}>
                {m.routeMeta && <RouteBadge meta={m.routeMeta} />}
                <MatchResultCard data={m.data} />
              </div>
            );
          }
          if (m.type === 'global_ranking') {
            return (
              <div key={i} className="w-full space-y-3">
                {m.routeMeta && <RouteBadge meta={m.routeMeta} />}
                <div className="bg-white p-5 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4 text-center">Global Ranking</h3>
                  <div className="space-y-3">
                    {m.data?.rankings?.map((item, idx) => (
                      <div key={idx} className={`rounded-2xl p-4 ${idx === 0 ? 'bg-gray-800 text-white' : 'bg-gray-50'}`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-black ${idx === 0 ? 'text-white' : 'text-gray-800'}`}>{item.rank}. {item.company}</span>
                          <span className={`text-lg font-black ${idx === 0 ? 'text-white' : 'text-gray-800'}`}>{item.score}</span>
                        </div>
                        <div className={`text-xs font-bold ${idx === 0 ? 'text-gray-300' : 'text-gray-500'}`}>{item.title}</div>
                        <div className={`text-[10px] font-bold mt-1 ${idx === 0 ? 'text-gray-400' : 'text-gray-400'}`}>{item.reason}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          }
          if (m.type === 'rag_answer') {
            return (
              <div key={i} className="flex justify-start flex-col">
                {m.routeMeta && <RouteBadge meta={m.routeMeta} />}
                <div className="max-w-[85%] p-5 rounded-[2.2rem] bg-white text-gray-800 rounded-tl-none border border-gray-100 shadow-sm">
                  <p className="text-sm font-bold leading-relaxed whitespace-pre-line mb-3">{m.text}</p>
                  {m.sources?.length > 0 && (
                    <div className="pt-3 border-t border-gray-100 space-y-1">
                      <div className="text-[10px] font-black text-gray-400 uppercase tracking-widest">引用来源</div>
                      {m.sources.map((s, idx) => (
                        <div key={idx} className="text-[10px] text-gray-500 font-bold">• {s}</div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          }
          if (m.type === 'typing') {
            return (
              <div key={i} className="flex justify-start">
                <div className="bg-white text-gray-800 p-5 rounded-[2.2rem] rounded-tl-none border border-gray-100 shadow-sm inline-flex items-center gap-3">
                  <Loader2 size={16} className="animate-spin text-gray-500" />
                  <span className="text-sm font-bold text-gray-500">AI 正在深度解析简历与 JD 的契合度...</span>
                </div>
              </div>
            );
          }
          if (m.type === 'image') {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] p-4 rounded-[2.2rem] bg-gray-800 text-white rounded-tr-none shadow-sm space-y-2">
                  {m.imageUrl && (
                    <img 
                      src={m.imageUrl} 
                      alt="上传的 JD" 
                      className="max-w-full max-h-48 rounded-2xl object-cover"
                    />
                  )}
                  <div className="flex items-center gap-3">
                    <div className="bg-white/20 p-2 rounded-2xl">
                      <ImageIcon size={20} />
                    </div>
                    <div className="text-sm font-bold">{m.text}</div>
                  </div>
                </div>
              </div>
            );
          }
          if (m.type === 'file') {
            return (
              <div key={i} className="flex justify-end">
                <div className="max-w-[85%] p-4 rounded-[2.2rem] bg-gray-800 text-white rounded-tr-none shadow-sm flex items-center gap-3">
                  <div className="bg-white/20 p-2 rounded-2xl">
                    <FileText size={20} />
                  </div>
                  <div className="text-sm font-bold">{m.text}</div>
                </div>
              </div>
            );
          }
          return (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} flex-col`}>
              {m.role === 'bot' && m.routeMeta && <RouteBadge meta={m.routeMeta} />}
              <div className={`max-w-[85%] p-5 rounded-[2.2rem] text-sm font-bold leading-relaxed shadow-sm whitespace-pre-line ${
                m.role === 'user' 
                  ? 'bg-gray-800 text-white rounded-tr-none' 
                  : 'bg-white text-gray-800 rounded-tl-none border border-gray-100'
              }`}>
                {m.text}
              </div>
            </div>
          );
        })}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入栏 */}
      <div className="p-4 bg-white flex items-center gap-2 relative z-10 border-t border-gray-50">
        <label className="p-3 bg-gray-50 rounded-full text-gray-400 hover:text-gray-600 transition-colors cursor-pointer shrink-0">
          <input type="file" accept="image/*,.txt,.pdf,.doc,.docx" className="hidden" onChange={handleJDUpload} />
          <Upload size={20} />
        </label>
        <div className="flex-1 bg-gray-50 rounded-3xl px-5 py-3 border border-gray-100 focus-within:border-gray-300 focus-within:bg-white transition-all">
          <textarea 
            ref={textareaRef}
            defaultValue=""
            onInput={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              // 中文输入法编辑状态下不触发发送（isComposing 为 true）
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder="询问职位，或上传 JD 截图 / 文本..."
            className="w-full bg-transparent text-sm text-gray-600 outline-none font-bold placeholder:font-normal resize-none overflow-y-auto leading-relaxed"
            style={{ minHeight: '44px', maxHeight: '120px' }}
            rows={1}
          />
        </div>
        <button 
          onClick={handleSendMessage}
          disabled={!inputValue.trim()}
          className={`p-3.5 rounded-full shadow-lg transition-all ${inputValue.trim() ? 'bg-gray-800 text-white shadow-gray-200 hover:scale-105' : 'bg-gray-100 text-gray-300'}`}
        >
          <Send size={18} />
        </button>
      </div>

      {/* LLM 配置弹窗 */}
      {showLLMConfig && (
        <div className="absolute inset-0 z-[120] bg-black/20 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="absolute bottom-0 w-full bg-white rounded-t-[3.5rem] p-10 animate-in slide-in-from-bottom-full duration-500 max-h-[90%] flex flex-col">
            {/* 顶部关闭条 */}
            <button 
              onClick={() => setShowLLMConfig(false)} 
              className="absolute top-4 left-1/2 -translate-x-1/2 w-12 h-1.5 bg-gray-200 rounded-full"
            />
            <button 
              onClick={() => setShowLLMConfig(false)} 
              className="absolute top-3 right-6 p-2.5 bg-gray-100 rounded-full text-gray-500 hover:bg-gray-200 transition-colors"
            >
              <X size={20} />
            </button>

            <div className="flex items-center gap-3 mb-6 mt-2">
              <div className="w-10 h-10 bg-gray-100 rounded-2xl flex items-center justify-center text-gray-600">
                <Settings size={20} />
              </div>
              <h3 className="text-xl font-black text-gray-800">LLM 配置</h3>
            </div>

            <div className="flex-1 overflow-y-auto no-scrollbar pr-1">

            <div className="space-y-6">
              {/* 对话回复模型 */}
              <div className="bg-gray-50 rounded-3xl p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <MessageSquare size={14} className="text-gray-500" />
                  <span className="text-xs font-black text-gray-500 uppercase tracking-widest">对话回复模型</span>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">Base URL</label>
                  <input 
                    type="text" 
                    value={llmConfig.chat.baseUrl} 
                    onChange={e => setLlmConfig({...llmConfig, chat: {...llmConfig.chat, baseUrl: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">API Key</label>
                  <input 
                    type="password" 
                    value={llmConfig.chat.apiKey} 
                    onChange={e => setLlmConfig({...llmConfig, chat: {...llmConfig.chat, apiKey: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="sk-..."
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">模型名称</label>
                  <input 
                    type="text" 
                    value={llmConfig.chat.model} 
                    onChange={e => setLlmConfig({...llmConfig, chat: {...llmConfig.chat, model: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="gpt-4o"
                  />
                </div>
              </div>

              {/* 核心业务模型 */}
              <div className="bg-gray-50 rounded-3xl p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <BrainCircuit size={14} className="text-gray-500" />
                  <span className="text-xs font-black text-gray-500 uppercase tracking-widest">核心业务模型（匹配分析、面试题、简历解析）</span>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">Base URL</label>
                  <input 
                    type="text" 
                    value={llmConfig.core.baseUrl} 
                    onChange={e => setLlmConfig({...llmConfig, core: {...llmConfig.core, baseUrl: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">API Key</label>
                  <input 
                    type="password" 
                    value={llmConfig.core.apiKey} 
                    onChange={e => setLlmConfig({...llmConfig, core: {...llmConfig.core, apiKey: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">模型名称</label>
                  <input 
                    type="text" 
                    value={llmConfig.core.model} 
                    onChange={e => setLlmConfig({...llmConfig, core: {...llmConfig.core, model: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="gpt-4o"
                  />
                </div>
              </div>

              {/* 规划推理模型 */}
              <div className="bg-gray-50 rounded-3xl p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles size={14} className="text-gray-500" />
                  <span className="text-xs font-black text-gray-500 uppercase tracking-widest">规划推理模型（意图识别、Query改写、Plan生成）</span>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">Base URL</label>
                  <input 
                    type="text" 
                    value={llmConfig.planner.baseUrl} 
                    onChange={e => setLlmConfig({...llmConfig, planner: {...llmConfig.planner, baseUrl: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">API Key</label>
                  <input 
                    type="password" 
                    value={llmConfig.planner.apiKey} 
                    onChange={e => setLlmConfig({...llmConfig, planner: {...llmConfig.planner, apiKey: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">模型名称</label>
                  <input 
                    type="text" 
                    value={llmConfig.planner.model} 
                    onChange={e => setLlmConfig({...llmConfig, planner: {...llmConfig.planner, model: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="gpt-4o-mini"
                  />
                </div>
              </div>

              {/* 记忆管理模型 */}
              <div className="bg-gray-50 rounded-3xl p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <Database size={14} className="text-gray-500" />
                  <span className="text-xs font-black text-gray-500 uppercase tracking-widest">记忆管理模型（压缩、提取、话题检测）</span>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">Base URL</label>
                  <input 
                    type="text" 
                    value={llmConfig.memory.baseUrl} 
                    onChange={e => setLlmConfig({...llmConfig, memory: {...llmConfig.memory, baseUrl: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">API Key</label>
                  <input 
                    type="password" 
                    value={llmConfig.memory.apiKey} 
                    onChange={e => setLlmConfig({...llmConfig, memory: {...llmConfig.memory, apiKey: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="默认复用对话模型"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">模型名称</label>
                  <input 
                    type="text" 
                    value={llmConfig.memory.model} 
                    onChange={e => setLlmConfig({...llmConfig, memory: {...llmConfig.memory, model: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="gpt-3.5-turbo"
                  />
                </div>
              </div>

              {/* 多模态分析模型配置 */}
              <div className="bg-gray-50 rounded-3xl p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <ImageIcon size={14} className="text-gray-500" />
                  <span className="text-xs font-black text-gray-500 uppercase tracking-widest">多模态分析模型</span>
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">Base URL</label>
                  <input 
                    type="text" 
                    value={llmConfig.vision.baseUrl} 
                    onChange={e => setLlmConfig({...llmConfig, vision: {...llmConfig.vision, baseUrl: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">API Key</label>
                  <input 
                    type="password" 
                    value={llmConfig.vision.apiKey} 
                    onChange={e => setLlmConfig({...llmConfig, vision: {...llmConfig.vision, apiKey: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="sk-..."
                  />
                </div>
                <div>
                  <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">模型名称</label>
                  <input 
                    type="text" 
                    value={llmConfig.vision.model} 
                    onChange={e => setLlmConfig({...llmConfig, vision: {...llmConfig.vision, model: e.target.value}})}
                    className="w-full bg-white rounded-2xl px-5 py-3.5 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                    placeholder="gpt-4o-vision"
                  />
                </div>
              </div>
            </div>
            </div>

            <button 
              onClick={async () => {
                try {
                  await apiSettings.updateLLM(llmConfig);
                  setShowLLMConfig(false);
                  setMessages(prev => [...prev, { role: 'bot', text: '✅ LLM 配置已保存。' }]);
                } catch (err) {
                  console.error(err);
                  alert('配置保存失败，请重试');
                }
              }}
              className="w-full mt-6 py-5 bg-gray-800 text-white rounded-[2.5rem] font-black shadow-lg shadow-gray-200 shrink-0"
            >
              保存配置
            </button>
          </div>
        </div>
      )}
    </div>
  );

  const kbViewJSX = (
    <div className="flex flex-col h-full bg-[#FAFAFA] relative overflow-hidden">
      <div className="px-6 pt-14 pb-4 flex items-center justify-between relative z-10">
        <button 
          onClick={() => setCurrentView('chat')} 
          className="w-10 h-10 rounded-2xl bg-white shadow-sm flex items-center justify-center text-gray-600 border border-gray-100"
        >
          <ChevronLeft size={20} />
        </button>
        <h2 className="text-lg font-black text-gray-900">知识库管理</h2>
        <button 
          onClick={() => setIsAddingJD(true)}
          className="w-10 h-10 rounded-2xl bg-gray-800 text-white shadow-lg shadow-gray-200 flex items-center justify-center"
        >
          <Plus size={20} />
        </button>
      </div>

      <div className="flex-1 px-6 pt-2 overflow-y-auto space-y-4 pb-10">
         {Object.entries(groupedJDs).map(([company, jds]) => (
          <div key={company} className="space-y-2">
            <button 
              onClick={() => setExpandedCompany(expandedCompany === company ? null : company)}
              className={`w-full flex items-center p-5 rounded-[2rem] border transition-all ${expandedCompany === company ? 'bg-gray-800 border-gray-800 text-white shadow-lg' : 'bg-white border-gray-100 text-gray-800 shadow-sm'}`}
            >
              <div className={`w-10 h-10 rounded-2xl flex items-center justify-center mr-4 font-black text-lg ${expandedCompany === company ? 'bg-white/20' : 'bg-gray-50 text-gray-400'}`}>{company.charAt(0)}</div>
              <div className="text-left flex-1 font-black text-lg">{company}</div>
              {expandedCompany === company ? <ChevronLeft className="-rotate-90" size={20} /> : <ChevronRight size={20} />}
            </button>
            {expandedCompany === company && (
              <div className="pl-6 space-y-2 animate-in slide-in-from-top-4">
                {jds.map(jd => (
                  <div 
                    key={jd.id} 
                    onClick={() => setSelectedJD(jd)} 
                    className="bg-white p-5 rounded-[1.8rem] border border-gray-100 flex items-center justify-between shadow-sm hover:border-gray-300 cursor-pointer"
                  >
                    <div className="font-black text-gray-800 text-sm">{jd.title}</div>
                    <ChevronRight size={16} className="text-gray-200" />
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* JD 详情抽屉 */}
      {selectedJD && (
        <div className="absolute inset-0 z-[110] bg-black/20 backdrop-blur-sm animate-in fade-in duration-300">
          <div className="absolute bottom-0 w-full bg-white rounded-t-[3.5rem] p-8 pb-10 animate-in slide-in-from-bottom-full duration-500 max-h-[85%] flex flex-col">
            {/* 顶部关闭条 */}
            <button onClick={() => setSelectedJD(null)} className="absolute top-4 left-1/2 -translate-x-1/2 w-12 h-1.5 bg-gray-200 rounded-full" />
            <button onClick={() => setSelectedJD(null)} className="absolute top-3 right-6 p-2.5 bg-gray-100 rounded-full text-gray-500">
              <X size={20} />
            </button>

            <div className="flex-1 overflow-y-auto no-scrollbar pr-1 mt-4">
              {/* 头部信息 */}
              <div className="flex items-start gap-4 mb-6">
                <div className={`w-14 h-14 rounded-2xl ${selectedJD.color || 'bg-gray-100 text-gray-600'} flex items-center justify-center shrink-0 font-black text-xl`}>
                  {(selectedJD.company || '?').charAt(0)}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-xl font-black text-gray-800 leading-tight">{selectedJD.company}</h3>
                  <p className="text-gray-600 font-bold text-sm mt-1">{selectedJD.title}</p>
                  <div className="flex items-center gap-3 mt-2">
                    <span className="text-[10px] font-bold text-gray-400 flex items-center gap-1">
                      <MapPin size={10} /> {selectedJD.location || '远程'}
                    </span>
                    <span className="text-[10px] font-bold text-gray-400 flex items-center gap-1">
                      <Briefcase size={10} /> {selectedJD.salary || '面议'}
                    </span>
                  </div>
                </div>
              </div>

              {/* 关键词标签 */}
              {selectedJD.keywords && selectedJD.keywords.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-5">
                  {selectedJD.keywords.map((kw, idx) => (
                    <span key={idx} className="text-[10px] font-black px-2.5 py-1 bg-gray-100 text-gray-600 rounded-full">{kw}</span>
                  ))}
                </div>
              )}

              {/* 结构化内容 */}
              <div className="space-y-5">
                {/* 岗位职责 */}
                {(selectedJD.sections?.responsibilities || selectedJD.description) && (
                  <div>
                    <h4 className="text-gray-400 text-[10px] font-black uppercase tracking-widest mb-2">岗位职责</h4>
                    <div className="text-gray-700 text-sm font-bold leading-relaxed bg-gray-50 p-4 rounded-2xl whitespace-pre-line">
                      {selectedJD.sections?.responsibilities || selectedJD.description}
                    </div>
                  </div>
                )}

                {/* 硬性要求 */}
                {selectedJD.sections?.hard_requirements && selectedJD.sections.hard_requirements.length > 0 && (
                  <div>
                    <h4 className="text-gray-400 text-[10px] font-black uppercase tracking-widest mb-2">硬性要求</h4>
                    <div className="space-y-2">
                      {selectedJD.sections.hard_requirements.map((item, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5 shrink-0" />
                          <span className="text-sm font-bold text-gray-700">{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 软性要求 */}
                {selectedJD.sections?.soft_requirements && selectedJD.sections.soft_requirements.length > 0 && (
                  <div>
                    <h4 className="text-gray-400 text-[10px] font-black uppercase tracking-widest mb-2">软性要求 / 加分项</h4>
                    <div className="space-y-2">
                      {selectedJD.sections.soft_requirements.map((item, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-green-400 mt-1.5 shrink-0" />
                          <span className="text-sm font-bold text-gray-700">{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 原始文本折叠 */}
                {selectedJD.raw_text && (
                  <div>
                    <details className="group">
                      <summary className="text-[10px] font-black text-gray-400 uppercase tracking-widest cursor-pointer list-none flex items-center gap-1">
                        <ChevronRight size={12} className="group-open:rotate-90 transition-transform" /> 原始文本
                      </summary>
                      <div className="text-gray-500 text-xs font-bold leading-relaxed bg-gray-50 p-4 rounded-2xl mt-2 whitespace-pre-line">
                        {selectedJD.raw_text}
                      </div>
                    </details>
                  </div>
                )}
              </div>
            </div>

            {/* 底部操作按钮 */}
            <div className="flex gap-3 mt-6 pt-4 border-t border-gray-100 shrink-0">
              <button 
                onClick={async () => {
                  if (!confirm('确定要删除这条 JD 吗？')) return;
                  try {
                    const delRes = await apiKB.delete(selectedJD.id);
                    setKbItems(prev => prev.filter(i => i.id !== selectedJD.id));
                    setSelectedJD(null);
                    const cleaned = delRes?.vector_cleaned;
                    const delMsg = cleaned
                      ? `🗑️ 已删除 JD 并清理向量库：「${selectedJD.company}」· ${selectedJD.title}`
                      : `🗑️ 已删除 JD（向量库清理异常）：「${selectedJD.company}」· ${selectedJD.title}`;
                    setMessages(prev => [...prev, { role: 'bot', text: delMsg }]);
                  } catch (err) {
                    console.error(err);
                    alert('删除失败，请重试');
                  }
                }} 
                className="flex-1 py-4 bg-gray-100 text-gray-600 rounded-2xl font-black flex items-center justify-center gap-2"
              >
                <Trash2 size={18} /> 删除
              </button>
              <button 
                onClick={() => { setCurrentView('chat'); setSelectedJD(null); }} 
                className="flex-[2] py-4 bg-gray-800 text-white rounded-2xl font-black shadow-lg shadow-gray-200 flex items-center justify-center gap-2"
              >
                <MessageSquare size={18} /> 返回对话
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* 添加 JD 遮罩 */}
      {isAddingJD && (
        <div className="absolute inset-0 z-[70] bg-gray-50/90 backdrop-blur-md flex flex-col justify-end animate-in fade-in duration-300">
          <div className="bg-white rounded-t-[3.5rem] p-8 pb-12 shadow-2xl relative border-t border-gray-100 max-h-[90%] flex flex-col">
            {/* 关闭条 */}
            <button onClick={closeJDAddFlow} className="absolute top-4 left-1/2 -translate-x-1/2 w-12 h-1.5 bg-gray-200 rounded-full" />
            <button onClick={closeJDAddFlow} className="absolute top-3 right-6 p-2.5 bg-gray-100 rounded-full text-gray-500"><X size={20}/></button>

            {/* 标题 */}
            <div className="text-center mb-6 mt-4">
              <h3 className="text-xl font-black text-gray-800">
                {jdAddStep === 'preview' ? '确认 JD 信息' : '添加新 JD'}
              </h3>
              <p className="text-gray-400 text-xs font-bold mt-2">
                {jdAddStep === 'preview' ? '请核对 AI 识别的信息，支持编辑修改' : 'AI 将自动识别职位与公司信息'}
              </p>
            </div>

            {/* Tab 切换 —— 仅在输入步骤显示 */}
            {jdAddStep === 'input' && (
              <div className="flex bg-gray-100 rounded-2xl p-1 mb-6">
                <button 
                  onClick={() => setJdAddMode('text')}
                  className={`flex-1 py-2.5 rounded-xl text-xs font-black transition-all ${jdAddMode === 'text' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500'}`}
                >
                  文本粘贴
                </button>
                <button 
                  onClick={() => setJdAddMode('image')}
                  className={`flex-1 py-2.5 rounded-xl text-xs font-black transition-all ${jdAddMode === 'image' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500'}`}
                >
                  图片上传
                </button>
              </div>
            )}

            {/* 内容区域 */}
            <div className="flex-1 overflow-y-auto no-scrollbar">
              {/* --- 输入步骤：文本模式 --- */}
              {jdAddStep === 'input' && jdAddMode === 'text' && (
                <div className="space-y-4">
                  <textarea 
                    value={newJDText} 
                    onChange={e => setNewJDText(e.target.value)} 
                    className="w-full bg-gray-50 rounded-3xl p-6 h-64 border border-gray-100 outline-none text-sm font-bold resize-none leading-relaxed" 
                    placeholder="请粘贴完整的 JD 内容，包含岗位职责、任职要求等信息..."
                  />
                  <button 
                    onClick={handleParseJDText} 
                    disabled={!newJDText.trim()} 
                    className="w-full py-5 bg-gray-800 text-white rounded-[2rem] font-black shadow-lg shadow-gray-200 disabled:opacity-50"
                  >
                    开始识别
                  </button>
                </div>
              )}

              {/* --- 输入步骤：图片模式 --- */}
              {jdAddStep === 'input' && jdAddMode === 'image' && (
                <div className="space-y-4">
                  <label className="block w-full h-48 bg-gray-50 rounded-3xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center cursor-pointer hover:border-gray-400 transition-colors">
                    <input 
                      type="file" 
                      accept="image/*" 
                      className="hidden" 
                      onChange={e => { const f = e.target.files?.[0]; if (f) setJdImageFile(f); }} 
                    />
                    {jdImageFile ? (
                      <div className="text-center px-4">
                        <ImageIcon size={32} className="text-gray-400 mx-auto mb-2" />
                        <p className="text-sm font-bold text-gray-700">{jdImageFile.name}</p>
                        <p className="text-[10px] text-gray-400 font-bold mt-1">点击更换图片</p>
                      </div>
                    ) : (
                      <div className="text-center px-4">
                        <Upload size={32} className="text-gray-300 mx-auto mb-2" />
                        <p className="text-sm font-bold text-gray-500">点击上传 JD 截图</p>
                        <p className="text-[10px] text-gray-400 font-bold mt-1">支持 PNG / JPG / JPEG</p>
                      </div>
                    )}
                  </label>
                  <button 
                    onClick={handleParseJDImage} 
                    disabled={!jdImageFile} 
                    className="w-full py-5 bg-gray-800 text-white rounded-[2rem] font-black shadow-lg shadow-gray-200 disabled:opacity-50"
                  >
                    开始识别
                  </button>
                </div>
              )}

              {/* --- 解析中 --- */}
              {jdAddStep === 'parsing' && (
                <div className="flex flex-col items-center justify-center h-64">
                  <Loader2 size={40} className="animate-spin text-gray-400 mb-4" />
                  <p className="text-sm font-bold text-gray-500">AI 正在解析 JD 内容...</p>
                  <p className="text-[10px] text-gray-400 font-bold mt-2">提取岗位信息、要求、关键词</p>
                </div>
              )}

              {/* --- 预览确认 --- */}
              {jdAddStep === 'preview' && jdPreview && (
                <div className="space-y-4">
                  {/* 公司 & 岗位 */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">公司</label>
                      <input 
                        value={jdPreview.company || ''} 
                        onChange={e => setJdPreview({...jdPreview, company: e.target.value})}
                        className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">岗位</label>
                      <input 
                        value={jdPreview.position || jdPreview.title || ''} 
                        onChange={e => setJdPreview({...jdPreview, position: e.target.value})}
                        className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                      />
                    </div>
                  </div>
                  {/* 地点 & 薪资 */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">地点</label>
                      <input 
                        value={jdPreview.location || ''} 
                        onChange={e => setJdPreview({...jdPreview, location: e.target.value})}
                        className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                        placeholder="如：北京"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">薪资</label>
                      <input 
                        value={jdPreview.salary_range || ''} 
                        onChange={e => setJdPreview({...jdPreview, salary_range: e.target.value})}
                        className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                        placeholder="如：30k-60k"
                      />
                    </div>
                  </div>
                  {/* 岗位职责 */}
                  <div>
                    <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">岗位职责</label>
                    <textarea 
                      value={jdPreview.sections?.responsibilities || ''} 
                      onChange={e => setJdPreview({...jdPreview, sections: {...(jdPreview.sections || {}), responsibilities: e.target.value}})}
                      className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold resize-none h-24 leading-relaxed"
                    />
                  </div>
                  {/* 硬性要求 */}
                  <div>
                    <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">硬性要求</label>
                    <textarea 
                      value={Array.isArray(jdPreview.sections?.hard_requirements) ? jdPreview.sections.hard_requirements.join('\n') : (jdPreview.sections?.hard_requirements || '')} 
                      onChange={e => setJdPreview({...jdPreview, sections: {...(jdPreview.sections || {}), hard_requirements: e.target.value.split('\n').filter(s => s.trim())}})}
                      className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold resize-none h-20 leading-relaxed"
                      placeholder="逐行输入硬性要求..."
                    />
                  </div>
                  {/* 软性要求 */}
                  <div>
                    <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">软性要求 / 加分项</label>
                    <textarea 
                      value={Array.isArray(jdPreview.sections?.soft_requirements) ? jdPreview.sections.soft_requirements.join('\n') : (jdPreview.sections?.soft_requirements || '')} 
                      onChange={e => setJdPreview({...jdPreview, sections: {...(jdPreview.sections || {}), soft_requirements: e.target.value.split('\n').filter(s => s.trim())}})}
                      className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold resize-none h-20 leading-relaxed"
                      placeholder="逐行输入软性要求..."
                    />
                  </div>
                  {/* 关键词 */}
                  <div>
                    <label className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1.5 block">关键词</label>
                    <input 
                      value={Array.isArray(jdPreview.keywords) ? jdPreview.keywords.join(', ') : ''} 
                      onChange={e => setJdPreview({...jdPreview, keywords: e.target.value.split(',').map(s => s.trim()).filter(Boolean)})}
                      className="w-full bg-gray-50 rounded-2xl px-4 py-3 text-sm text-gray-700 border border-gray-100 outline-none focus:border-gray-300 font-bold"
                      placeholder="用逗号分隔，如：Python, 产品经理, 大模型"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* 底部按钮 */}
            {jdAddStep === 'preview' && (
              <div className="flex gap-3 mt-6 pt-4 border-t border-gray-100 shrink-0">
                <button 
                  onClick={() => setJdAddStep('input')} 
                  className="flex-1 py-4 bg-gray-100 text-gray-600 rounded-2xl font-black"
                >
                  返回修改
                </button>
                <button 
                  onClick={handleConfirmAddJD} 
                  disabled={isProcessing}
                  className="flex-[2] py-4 bg-gray-800 text-white rounded-2xl font-black shadow-lg shadow-gray-200 disabled:opacity-50"
                >
                  {isProcessing ? '存入中...' : '确认存入知识库'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const Tag = ({ text }) => (
    <span className="px-3 py-1.5 bg-gray-100 rounded-full text-xs font-black text-gray-700">{text}</span>
  );

  const resumeViewJSX = (
      <div className="flex flex-col h-full bg-[#F5F5F5] relative overflow-hidden pb-24">
        <div className="px-6 pt-14 pb-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-gray-200 rounded-2xl flex items-center justify-center text-gray-600">
              <FileText size={20} />
            </div>
            <div>
              <h2 className="text-2xl font-black text-gray-900">我的简历</h2>
              <p className="text-gray-400 text-xs font-bold">AI 结构化解析 · 支持手动修正</p>
            </div>
          </div>
        </div>

        <div className="flex-1 px-6 pb-8 overflow-y-auto space-y-4">
          {/* 未上传时提示 */}
          {!hasResume && (
            <div className="bg-gray-100 border border-gray-200 p-5 rounded-3xl flex items-start gap-3">
              <AlertCircle size={20} className="text-gray-500 shrink-0 mt-0.5" />
              <p className="text-sm font-bold text-gray-700 leading-relaxed">
                上传简历后，AI 将自动解析为结构化数据，用于后续的 JD 匹配与面试题生成。
              </p>
            </div>
          )}

          {/* 上传按钮区域 */}
          {!hasResume ? (
            <label className="block">
              <input type="file" accept=".txt,.pdf,.doc,.docx" className="hidden" onChange={handleResumeUpload} />
              <div className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm flex items-center gap-3 cursor-pointer hover:border-gray-300 transition-colors">
                <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center text-gray-500 shrink-0">
                  <Upload size={18} />
                </div>
                <div>
                  <div className="text-sm font-black text-gray-700">点击上传简历文件</div>
                  <div className="text-[10px] text-gray-400 font-bold">支持 PDF / TXT / DOC / DOCX</div>
                </div>
              </div>
            </label>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => triggerResumeUpload('text')}
                className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm flex items-center justify-center gap-2 hover:border-gray-300 transition-colors"
              >
                <div className="w-9 h-9 bg-gray-100 rounded-xl flex items-center justify-center text-gray-500 shrink-0">
                  <FileText size={16} />
                </div>
                <span className="text-sm font-black text-gray-700">函数提取</span>
              </button>
              <button
                onClick={() => triggerResumeUpload('vision')}
                className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm flex items-center justify-center gap-2 hover:border-gray-300 transition-colors"
              >
                <div className="w-9 h-9 bg-gray-100 rounded-xl flex items-center justify-center text-gray-500 shrink-0">
                  <ImageIcon size={16} />
                </div>
                <span className="text-sm font-black text-gray-700">多模态解析</span>
              </button>
              <input type="file" ref={resumeFileRef} className="hidden" onChange={handleResumeUpload} />
            </div>
          )}

          {/* 解析状态 */}
          {parseStatus === 'parsing' && (
            <div className="bg-white p-5 rounded-3xl border border-gray-100 shadow-sm flex items-center gap-3">
              <Loader2 size={20} className="animate-spin text-gray-500" />
              <span className="text-sm font-bold text-gray-600">AI 正在解析简历内容，请稍候...</span>
            </div>
          )}

          {/* 结构化展示 */}
          {hasResume && resumeSchema && (
            <div className="space-y-4">
              {/* 基础信息 */}
              <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-xl font-black text-gray-900">{resumeSchema.basic_info.name || '未识别姓名'}</h3>
                    <p className="text-sm text-gray-500 font-bold mt-1">
                      {resumeSchema.basic_info.current_title || ''}
                      {resumeSchema.basic_info.current_title && resumeSchema.basic_info.current_company ? ' · ' : ''}
                      {resumeSchema.basic_info.current_company || ''}
                    </p>
                    <p className="text-xs text-gray-400 font-bold mt-0.5">
                      {resumeSchema.basic_info.years_exp !== null ? `${resumeSchema.basic_info.years_exp} 年经验` : ''}
                      {resumeSchema.basic_info.location ? ` · ${resumeSchema.basic_info.location}` : ''}
                    </p>
                  </div>
                  <div className="w-12 h-12 bg-gray-100 rounded-2xl flex items-center justify-center text-gray-600">
                    <CheckCircle2 size={24} />
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 text-xs text-gray-500 font-bold">
                  {resumeSchema.basic_info.phone && <span>📱 {resumeSchema.basic_info.phone}</span>}
                  {resumeSchema.basic_info.email && <span>✉️ {resumeSchema.basic_info.email}</span>}
                  {resumeSchema.basic_info.availability && <span>🚀 {resumeSchema.basic_info.availability}</span>}
                </div>
                {(resumeSchema.basic_info.target_locations?.length > 0 || resumeSchema.basic_info.target_salary_min || resumeSchema.basic_info.target_salary_max) && (
                  <div className="mt-3 pt-3 border-t border-gray-50 flex flex-wrap gap-2 text-xs text-gray-500 font-bold">
                    {resumeSchema.basic_info.target_locations?.length > 0 && <span>📍 期望：{resumeSchema.basic_info.target_locations.join(' / ')}</span>}
                    {(resumeSchema.basic_info.target_salary_min || resumeSchema.basic_info.target_salary_max) && (
                      <span>💰 薪资：{resumeSchema.basic_info.target_salary_min || '?'}k - {resumeSchema.basic_info.target_salary_max || '?'}k</span>
                    )}
                  </div>
                )}
              </div>

              {/* 工作经历 */}
              {resumeSchema.work_experience?.length > 0 && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">工作经历</h4>
                  <div className="space-y-4">
                    {resumeSchema.work_experience.map((we, i) => (
                      <div key={i} className="bg-gray-50 rounded-2xl p-4">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-black text-gray-800">{we.company}</span>
                          <span className="text-[10px] text-gray-400 font-bold">{we.start_date} ~ {we.end_date || '至今'}</span>
                        </div>
                        <div className="text-xs text-gray-500 font-bold mb-1">{we.title} {we.department ? `· ${we.department}` : ''}</div>
                        {we.team_size && <div className="text-[10px] text-gray-400 font-bold mb-2">👥 团队规模：{we.team_size} 人</div>}
                        <p className="text-xs text-gray-600 font-bold leading-relaxed mb-2">{we.description}</p>
                        {we.achievements?.length > 0 && (
                          <div className="space-y-1 mb-2">
                            {we.achievements.map((a, idx) => (
                              <div key={idx} className="text-[10px] text-gray-500 font-bold">🏆 {a}</div>
                            ))}
                          </div>
                        )}
                        {we.keywords?.length > 0 && (
                          <div className="flex flex-wrap gap-1.5">
                            {we.keywords.map(k => <span key={k} className="px-2 py-1 bg-white rounded-lg text-[10px] font-black text-gray-500 border border-gray-100">{k}</span>)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 教育背景 */}
              {resumeSchema.education?.length > 0 && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">教育背景</h4>
                  <div className="space-y-3">
                    {resumeSchema.education.map((edu, i) => (
                      <div key={i} className="flex items-start gap-3">
                        <div className="w-8 h-8 bg-gray-100 rounded-xl flex items-center justify-center text-gray-500 shrink-0 text-xs font-black">{edu.degree?.charAt(0)}</div>
                        <div>
                          <div className="text-sm font-black text-gray-800">{edu.school} {edu.school_level ? <span className="text-[10px] text-gray-400 font-bold ml-1">{edu.school_level}</span> : ''}</div>
                          <div className="text-xs text-gray-500 font-bold">{edu.degree} · {edu.major} {edu.major_category ? `(${edu.major_category})` : ''} {edu.graduation_year ? `· ${edu.graduation_year}` : ''}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 项目经历 */}
              {resumeSchema.projects?.length > 0 && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">项目经历</h4>
                  <div className="space-y-4">
                    {resumeSchema.projects.map((proj, i) => (
                      <div key={i} className={`rounded-2xl p-4 ${proj.is_key_project ? 'bg-gray-800 text-white' : 'bg-gray-50'}`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-black ${proj.is_key_project ? 'text-white' : 'text-gray-800'}`}>{proj.name}</span>
                          <span className={`text-[10px] font-bold ${proj.is_key_project ? 'text-gray-300' : 'text-gray-400'}`}>{proj.duration}</span>
                        </div>
                        <div className={`text-xs font-bold mb-1 ${proj.is_key_project ? 'text-gray-300' : 'text-gray-500'}`}>{proj.role} {proj.company ? `· ${proj.company}` : ''}</div>
                        <p className={`text-xs font-bold leading-relaxed mb-2 ${proj.is_key_project ? 'text-gray-200' : 'text-gray-600'}`}>{proj.description}</p>
                        {proj.metrics?.length > 0 && (
                          <div className="space-y-1 mb-2">
                            {proj.metrics.map((m, idx) => (
                              <div key={idx} className={`text-[10px] font-bold ${proj.is_key_project ? 'text-gray-300' : 'text-gray-500'}`}>📈 {m}</div>
                            ))}
                          </div>
                        )}
                        <div className="flex flex-wrap gap-1.5">
                          {proj.tech_keywords?.map(k => <span key={k} className={`px-2 py-1 rounded-lg text-[10px] font-black ${proj.is_key_project ? 'bg-gray-700 text-gray-200' : 'bg-white text-gray-500 border border-gray-100'}`}>{k}</span>)}
                          {proj.business_keywords?.map(k => <span key={k} className={`px-2 py-1 rounded-lg text-[10px] font-black ${proj.is_key_project ? 'bg-gray-600 text-gray-100' : 'bg-gray-100 text-gray-600'}`}>{k}</span>)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 技能栈 */}
              {(resumeSchema.skills?.technical?.length > 0 || resumeSchema.skills?.business?.length > 0 || resumeSchema.skills?.soft?.length > 0) && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">技能栈</h4>
                  {resumeSchema.skills.technical?.length > 0 && (
                    <div className="mb-3">
                      <div className="text-[10px] text-gray-400 font-black mb-1.5">技术</div>
                      <div className="flex flex-wrap gap-2">
                        {resumeSchema.skills.technical.map(s => {
                          const level = resumeSchema.skills.proficiency_map?.[s];
                          return (
                            <span key={s} className={`px-3 py-1.5 rounded-full text-xs font-black ${level === '精通' ? 'bg-gray-800 text-white' : level === '熟练' ? 'bg-gray-200 text-gray-800' : 'bg-gray-100 text-gray-700'}`}>
                              {s} {level ? `· ${level}` : ''}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  {resumeSchema.skills.business?.length > 0 && (
                    <div className="mb-3">
                      <div className="text-[10px] text-gray-400 font-black mb-1.5">业务</div>
                      <div className="flex flex-wrap gap-2">
                        {resumeSchema.skills.business.map(s => <Tag key={s} text={s} />)}
                      </div>
                    </div>
                  )}
                  {resumeSchema.skills.soft?.length > 0 && (
                    <div>
                      <div className="text-[10px] text-gray-400 font-black mb-1.5">软技能</div>
                      <div className="flex flex-wrap gap-2">
                        {resumeSchema.skills.soft.map(s => <Tag key={s} text={s} />)}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* 证书 */}
              {resumeSchema.certifications?.length > 0 && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">证书与加分项</h4>
                  <div className="flex flex-wrap gap-2">
                    {resumeSchema.certifications.map((cert, i) => (
                      <span key={i} className="px-3 py-1.5 bg-gray-100 rounded-full text-xs font-black text-gray-700">
                        {cert.name} {cert.issuer ? `(${cert.issuer})` : ''}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* 个人优势 */}
              {resumeSchema.advantages?.length > 0 && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-4">个人优势</h4>
                  <div className="space-y-2">
                    {resumeSchema.advantages.map((a, i) => (
                      <div key={i} className="flex items-start gap-2">
                        <span className="text-gray-400 mt-0.5">•</span>
                        <span className="text-sm font-bold text-gray-700">{a}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 原始文本编辑（可折叠） */}
              <div className="flex items-center justify-between px-2">
                <span className="text-[10px] font-bold text-gray-400">
                  {resumeSchema.meta?.parser_version ? `解析来源：${resumeSchema.meta.parser_version}` : ''}
                </span>
                <button 
                  onClick={() => setShowRawText(!showRawText)}
                  className="py-3 text-xs font-black text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showRawText ? '收起原始文本 ▲' : '编辑原始文本 ▼'}
                </button>
              </div>

              {showRawText && (
                <div className="bg-white p-6 rounded-[2.5rem] border border-gray-100 shadow-sm">
                  <textarea 
                    className="w-full bg-gray-50 rounded-3xl p-6 h-64 border border-gray-100 outline-none text-sm text-gray-600 resize-none font-bold leading-relaxed focus:bg-white focus:border-gray-300 transition-all"
                    value={resumeText}
                    onChange={e => setResumeText(e.target.value)}
                    placeholder="原始简历文本..."
                  />
                  <button 
                    onClick={async () => {
                      if (!resumeText.trim()) return;
                      setParseStatus('parsing');
                      try {
                        const result = await apiResume.reparse(resumeId, resumeText.trim());
                        setResumeSchema(result.parsed_schema);
                        setResumeText(result.parsed_schema.meta?.raw_text || '');
                        setParseStatus('parsed');
                        setMessages(prev => [...prev, { role: 'bot', text: '✅ 简历已重新解析并更新。' }]);
                      } catch (err) {
                        console.error(err);
                        setParseStatus('error');
                        alert('重新解析失败：' + (err.message || '请检查 LLM 配置'));
                      }
                    }}
                    disabled={!resumeText.trim() || parseStatus === 'parsing'}
                    className={`w-full mt-4 py-4 rounded-[2rem] font-black text-lg shadow-xl transition-all ${
                      resumeText.trim() && parseStatus !== 'parsing' ? 'bg-gray-800 text-white shadow-gray-200 active:scale-95' : 'bg-gray-200 text-gray-400'
                    }`}
                  >
                    {parseStatus === 'parsing' ? 'AI 正在重新解析...' : '保存修改'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
  );

  // ---------------- 底部导航 ----------------
  const BottomNav = () => (
    <div className="absolute bottom-0 left-0 w-full bg-white/90 backdrop-blur-md border-t border-gray-100 z-50 flex justify-around items-center px-6 pt-3 pb-7 rounded-b-[4.3rem]">
      <button 
        onClick={() => setActiveTab('chat')} 
        className={`flex flex-col items-center gap-1 transition-colors ${activeTab === 'chat' ? 'text-gray-900' : 'text-gray-300 hover:text-gray-500'}`}
      >
        <MessageSquare size={24} strokeWidth={activeTab === 'chat' ? 2.5 : 2} />
        <span className="text-[10px] font-black tracking-wide">对话</span>
      </button>
      <button 
        onClick={() => setActiveTab('resume')} 
        className={`flex flex-col items-center gap-1 transition-colors ${activeTab === 'resume' ? 'text-gray-900' : 'text-gray-300 hover:text-gray-500'}`}
      >
        <FileText size={24} strokeWidth={activeTab === 'resume' ? 2.5 : 2} />
        <span className="text-[10px] font-black tracking-wide">我的简历</span>
      </button>
    </div>
  );

  return (
    <div className="flex items-center justify-center min-h-screen bg-neutral-100 p-4">
      <div className="w-full max-w-[390px] h-[844px] bg-white rounded-[4.5rem] shadow-[0_50px_120px_-30px_rgba(0,0,0,0.15)] overflow-hidden border-[12px] border-slate-900 relative">
        {/* 刘海屏 */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-9 bg-slate-900 rounded-b-[2rem] z-[150] flex items-center justify-center pointer-events-none">
           <div className="w-10 h-1 bg-slate-800 rounded-full"></div>
        </div>
        
        {/* 主内容区 */}
        <div className="h-full relative">
          {currentView === 'chat' && chatViewJSX}
          {currentView === 'kb' && kbViewJSX}
          {currentView === 'resume' && resumeViewJSX}
        </div>

        {/* 底部导航（知识库页面隐藏） */}
        {currentView !== 'kb' && <BottomNav />}

        {/* Home Indicator */}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-32 h-1.5 bg-gray-200 rounded-full z-[60] pointer-events-none"></div>
      </div>
    </div>
  );
};

export default App;
