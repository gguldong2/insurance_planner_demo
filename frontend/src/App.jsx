import { useState, useRef, useEffect, useMemo } from 'react';
import axios from 'axios';
import './App.css';
import {
  Send,
  Bot,
  User,
  Terminal,
  Database,
  ShieldAlert,
  Cpu,
  ListChecks,
  Hash,
  TimerReset,
  Sparkles,
  ChevronRight,
} from 'lucide-react';

function renderInline(text) {
  const parts = String(text || '')
    .split(/(\*\*.*?\*\*|`.*?`)/g)
    .filter(Boolean);

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={index} className="font-semibold text-gray-900">
          {part.slice(2, -2)}
        </strong>
      );
    }

    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code
          key={index}
          className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.9em] text-slate-800"
        >
          {part.slice(1, -1)}
        </code>
      );
    }

    return <span key={index}>{part}</span>;
  });
}

function renderMarkdown(text) {
  if (!text) return null;

  const normalized = String(text)
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^Thinking Process:[\s\S]*?(?=\n\n|## |### |#### |# )/i, '')
    .trim();

  const lines = normalized.replace(/\r\n/g, '\n').split('\n');
  const elements = [];
  let i = 0;
  let key = 0;

  const isTableLine = (value) => /\|/.test(value);
  const isTableDivider = (value) =>
    /^\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?$/.test(value.trim());

  const renderTableCell = (value, cellKey, tag = 'td') => {
    const Tag = tag;
    return <Tag key={cellKey}>{renderInline(value.trim())}</Tag>;
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith('```')) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;

      elements.push(
        <pre key={key++} className="markdown-pre">
          <code>{codeLines.join('\n')}</code>
        </pre>
      );
      continue;
    }

    if (trimmed.startsWith('#### ')) {
      elements.push(
        <h4 key={key++} className="markdown-h4">
          {renderInline(trimmed.slice(5))}
        </h4>
      );
      i += 1;
      continue;
    }

    if (trimmed.startsWith('### ')) {
      elements.push(
        <h3 key={key++} className="markdown-h3">
          {renderInline(trimmed.slice(4))}
        </h3>
      );
      i += 1;
      continue;
    }

    if (trimmed.startsWith('## ')) {
      elements.push(
        <h2 key={key++} className="markdown-h2">
          {renderInline(trimmed.slice(3))}
        </h2>
      );
      i += 1;
      continue;
    }

    if (trimmed.startsWith('# ')) {
      elements.push(
        <h1 key={key++} className="markdown-h1">
          {renderInline(trimmed.slice(2))}
        </h1>
      );
      i += 1;
      continue;
    }

    if (/^-{3,}$/.test(trimmed)) {
      elements.push(<hr key={key++} className="markdown-hr" />);
      i += 1;
      continue;
    }

    if (trimmed.startsWith('>')) {
      const quoteLines = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i += 1;
      }

      elements.push(
        <blockquote key={key++} className="markdown-blockquote">
          {quoteLines.map((quoteLine, idx) => (
            <p key={idx}>{renderInline(quoteLine)}</p>
          ))}
        </blockquote>
      );
      continue;
    }

    if (isTableLine(trimmed) && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const headerCells = trimmed
        .replace(/^\||\|$/g, '')
        .split('|')
        .map((cell) => cell.trim());

      i += 2;

      const rows = [];
      while (i < lines.length && lines[i].trim() && isTableLine(lines[i].trim())) {
        rows.push(
          lines[i]
            .trim()
            .replace(/^\||\|$/g, '')
            .split('|')
            .map((cell) => cell.trim())
        );
        i += 1;
      }

      elements.push(
        <div key={key++} className="markdown-table-wrap">
          <table className="markdown-table">
            <thead>
              <tr>
                {headerCells.map((cell, idx) =>
                  renderTableCell(cell, `h-${idx}`, 'th')
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIdx) => (
                <tr key={`r-${rowIdx}`}>
                  {headerCells.map((_, cellIdx) =>
                    renderTableCell(row[cellIdx] || '', `c-${rowIdx}-${cellIdx}`)
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ''));
        i += 1;
      }

      elements.push(
        <ul key={key++} className="markdown-ul">
          {items.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ul>
      );
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i += 1;
      }

      elements.push(
        <ol key={key++} className="markdown-ol">
          {items.map((item, idx) => (
            <li key={idx}>{renderInline(item)}</li>
          ))}
        </ol>
      );
      continue;
    }

    const paragraphLines = [trimmed];
    i += 1;

    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^#{1,4}\s/.test(lines[i].trim()) &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !/^>/.test(lines[i].trim()) &&
      !/^```/.test(lines[i].trim()) &&
      !/^-{3,}$/.test(lines[i].trim()) &&
      !(isTableLine(lines[i].trim()) && i + 1 < lines.length && isTableDivider(lines[i + 1]))
    ) {
      paragraphLines.push(lines[i].trim());
      i += 1;
    }

    elements.push(
      <p key={key++} className="markdown-p">
        {renderInline(paragraphLines.join(' '))}
      </p>
    );
  }

  return <div className="markdown-body">{elements}</div>;
}

function parseLog(log, index) {
  if (typeof log === 'object' && log !== null) {
    return { id: index, raw: log, type: 'json', ...log };
  }

  const text = String(log || '');
  try {
    const parsed = JSON.parse(text);
    return { id: index, raw: text, type: 'json', ...parsed };
  } catch {
    return { id: index, type: 'text', raw: text, message: text };
  }
}

function summarizeLogs(items) {
  const jsonItems = items.filter((item) => item.type === 'json');
  const latest = jsonItems[jsonItems.length - 1] || null;
  const analyzer = jsonItems.find((item) => item.message === 'analyzer finished');
  const planner = jsonItems.find((item) => item.message === 'planner finished');
  const grounder = jsonItems.find((item) => item.message === 'grounder finished');
  const generator = jsonItems.find((item) => item.message === 'generator started');
  const failed = jsonItems.find((item) => /failed/i.test(item.message || '') || item.level === 'ERROR');

  const selectedTasks = [
    ...(Array.isArray(analyzer?.tasks) ? analyzer.tasks : []),
    ...(Array.isArray(planner?.task_plan)
      ? planner.task_plan.map((task) => task.task_type || task.title).filter(Boolean)
      : []),
  ];

  return {
    requestId: latest?.request_id || analyzer?.request_id || planner?.request_id || '-',
    selectedTasks: [...new Set(selectedTasks)],
    mode: generator?.mode || '-',
    keywordCount:
      typeof grounder?.keyword_count === 'number'
        ? grounder.keyword_count
        : Array.isArray(analyzer?.concept_keywords)
          ? analyzer.concept_keywords.length
          : 0,
    resolvedCount:
      typeof grounder?.resolved_count === 'number'
        ? grounder.resolved_count
        : Array.isArray(grounder?.resolved_concept_ids)
          ? grounder.resolved_concept_ids.length
          : 0,
    taskCount:
      typeof planner?.task_count === 'number'
        ? planner.task_count
        : Array.isArray(planner?.task_plan)
          ? planner.task_plan.length
          : 0,
    latestStatus: failed ? 'ERROR' : latest?.level || 'INFO',
    latestMessage: failed?.message || latest?.message || '대기 중',
    latestDuration:
      typeof latest?.duration_ms === 'number' ? `${latest.duration_ms} ms` : '-',
  };
}

function metricCard(icon, label, value, tone = 'default') {
  const toneClass = {
    default: 'bg-white border-gray-200 text-gray-900',
    softBlue: 'bg-blue-50 border-blue-200 text-blue-900',
    softPurple: 'bg-violet-50 border-violet-200 text-violet-900',
    softGreen: 'bg-emerald-50 border-emerald-200 text-emerald-900',
    softAmber: 'bg-amber-50 border-amber-200 text-amber-900',
    softRed: 'bg-rose-50 border-rose-200 text-rose-900',
  }[tone];

  return (
    <div className={`trace-metric-card ${toneClass}`} key={label}>
      <div className="trace-metric-icon">{icon}</div>
      <div>
        <div className="trace-metric-label">{label}</div>
        <div className="trace-metric-value">{value}</div>
      </div>
    </div>
  );
}

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [logs, setLogs] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const parsedLogs = useMemo(() => logs.map(parseLog), [logs]);
  const traceSummary = useMemo(() => summarizeLogs(parsedLogs), [parsedLogs]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setLogs([]);

    try {
      const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080';
      const response = await axios.post(`${API_BASE}/chat`, {
        query: userMsg.text,
      });

      const data = response.data;
      setLogs(data.logs || []);

      const botMsg = { role: 'bot', text: data.answer };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: '❌ Error: 서버와 연결할 수 없습니다.' },
      ]);
      setLogs((prev) => [...prev, `[System Error] ${error.message}`]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderLogItem = (log, index) => {
    const message = log.message || log.raw || 'No message';
    const isError = log.level === 'ERROR' || /failed|error/i.test(message);
    let icon = <Terminal size={14} />;
    let color = 'text-gray-500';
    let pill = 'bg-gray-100 text-gray-700 border-gray-200';

    if (/analyzer/i.test(message)) {
      icon = <Cpu size={14} />;
      color = 'text-amber-600';
      pill = 'bg-amber-50 text-amber-700 border-amber-200';
    } else if (/grounder|Graph|SQL/i.test(message)) {
      icon = <Database size={14} />;
      color = 'text-violet-600';
      pill = 'bg-violet-50 text-violet-700 border-violet-200';
    } else if (/planner|task_plan/i.test(message)) {
      icon = <ListChecks size={14} />;
      color = 'text-sky-600';
      pill = 'bg-sky-50 text-sky-700 border-sky-200';
    } else if (/generator|Final/i.test(message)) {
      icon = <Sparkles size={14} />;
      color = 'text-emerald-600';
      pill = 'bg-emerald-50 text-emerald-700 border-emerald-200';
    }

    if (isError) {
      icon = <ShieldAlert size={14} />;
      color = 'text-rose-600';
      pill = 'bg-rose-50 text-rose-700 border-rose-200';
    }

    const details = [];
    if (Array.isArray(log.tasks) && log.tasks.length) details.push(`tasks ${log.tasks.join(', ')}`);
    if (typeof log.task_count === 'number') details.push(`task_count ${log.task_count}`);
    if (typeof log.keyword_count === 'number') details.push(`keywords ${log.keyword_count}`);
    if (typeof log.resolved_count === 'number') details.push(`resolved ${log.resolved_count}`);
    if (typeof log.duration_ms === 'number') details.push(`${log.duration_ms} ms`);
    if (log.mode) details.push(`mode ${log.mode}`);

    return (
      <div key={index} className="trace-log-card">
        <div className="trace-log-card-top">
          <div className={`trace-log-icon ${color}`}>{icon}</div>
          <div className="min-w-0 flex-1">
            <div className="trace-log-title-row">
              <div className="trace-log-title">{message}</div>
              <span className={`trace-log-pill ${pill}`}>{log.level || 'LOG'}</span>
            </div>
            <div className="trace-log-subtitle">
              {log.taskName || 'task -'}
              {log.logger ? ` · ${log.logger}` : ''}
            </div>
          </div>
        </div>

        {details.length > 0 && (
          <div className="trace-log-details">
            {details.map((item, idx) => (
              <span key={idx} className="trace-chip">
                {item}
              </span>
            ))}
          </div>
        )}

        {Array.isArray(log.task_plan) && log.task_plan.length > 0 && (
          <div className="trace-task-plan">
            {log.task_plan.map((task, idx) => (
              <div key={idx} className="trace-task-item">
                <ChevronRight size={14} className="text-sky-500 shrink-0 mt-0.5" />
                <div>
                  <div className="trace-task-item-title">
                    {task.task_type || task.title || `task_${idx + 1}`}
                  </div>
                  <div className="trace-task-item-meta">
                    {task.title || '제목 없음'}
                    {Array.isArray(task.depends_on) && task.depends_on.length
                      ? ` · depends ${task.depends_on.join(', ')}`
                      : ''}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {log.exception && (
          <details className="trace-exception">
            <summary>exception 보기</summary>
            <pre>{log.exception}</pre>
          </details>
        )}
      </div>
    );
  };

  return (
    <div className="app-shell">
      <div className="chat-shell">
        <header className="chat-header">
          <div className="chat-header-avatar">
            <Bot size={22} />
          </div>
          <div>
            <h1 className="chat-header-title">AgensGraph Agent</h1>
            <p className="chat-header-subtitle">Powered by LangGraph & vLLM</p>
          </div>
        </header>

        <div className="chat-body">
          {messages.length === 0 && (
            <div className="empty-state">
              <Bot size={52} className="mx-auto mb-4 opacity-50" />
              <p className="empty-state-title">무엇이든 물어보세요</p>
              <p className="empty-state-desc">
                Agent가 워크플로우를 실행하고, 오른쪽 패널에 trace를 구조적으로 보여줍니다.
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-3 max-w-[82%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-900 text-white'
                  }`}
                >
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm break-words ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-none'
                      : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'
                  }`}
                >
                  {msg.role === 'user' ? msg.text : renderMarkdown(msg.text)}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="flex gap-3">
                <div className="w-9 h-9 bg-slate-900 rounded-full flex items-center justify-center text-white">
                  <Bot size={16} />
                </div>
                <div className="bg-white border border-gray-200 p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-75" />
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-150" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-wrap">
          <div className="chat-input-box">
            <input
              type="text"
              className="chat-input"
              placeholder="질문을 입력하세요..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className="chat-send-btn"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      <aside className="trace-shell">
        <header className="trace-header">
          <div className="trace-header-title">
            <Terminal size={16} />
            Agent Workflow Trace
          </div>
          <div className="trace-header-subtitle">실행 흐름과 선택된 task를 한눈에 확인</div>
        </header>

        <div className="trace-summary-grid">
          {metricCard(<Hash size={16} />, 'Request ID', traceSummary.requestId, 'softBlue')}
          {metricCard(<Cpu size={16} />, 'Mode', traceSummary.mode, 'softPurple')}
          {metricCard(<ListChecks size={16} />, 'Task Count', traceSummary.taskCount, 'softGreen')}
          {metricCard(
            <TimerReset size={16} />,
            'Latest',
            traceSummary.latestDuration,
            traceSummary.latestStatus === 'ERROR' ? 'softRed' : 'softAmber'
          )}
        </div>

        <div className="trace-section">
          <div className="trace-section-title">Selected Tasks</div>
          {traceSummary.selectedTasks.length > 0 ? (
            <div className="trace-chip-wrap">
              {traceSummary.selectedTasks.map((task) => (
                <span key={task} className="trace-chip trace-chip-strong">
                  {task}
                </span>
              ))}
            </div>
          ) : (
            <div className="trace-empty-box">아직 추출된 task가 없습니다.</div>
          )}
        </div>

        <div className="trace-section">
          <div className="trace-section-title">Quick Stats</div>
          <div className="trace-chip-wrap">
            <span className="trace-chip">keywords {traceSummary.keywordCount}</span>
            <span className="trace-chip">resolved {traceSummary.resolvedCount}</span>
            <span className="trace-chip">status {traceSummary.latestStatus}</span>
            <span className="trace-chip">message {traceSummary.latestMessage}</span>
          </div>
        </div>

        <div className="trace-log-list">
          {parsedLogs.length === 0 ? (
            <div className="trace-empty-state">
              대기 중...
              <br />
              워크플로우 실행 기록이 여기에 표시됩니다.
            </div>
          ) : (
            parsedLogs.map((log, idx) => renderLogItem(log, idx))
          )}
          <div ref={logsEndRef} />
        </div>
      </aside>
    </div>
  );
}

export default App;