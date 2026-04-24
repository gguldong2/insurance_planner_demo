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
        <code key={index} className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.9em] text-slate-800">
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
      elements.push(<h4 key={key++} className="markdown-h4">{renderInline(trimmed.slice(5))}</h4>);
      i += 1;
      continue;
    }
    if (trimmed.startsWith('### ')) {
      elements.push(<h3 key={key++} className="markdown-h3">{renderInline(trimmed.slice(4))}</h3>);
      i += 1;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      elements.push(<h2 key={key++} className="markdown-h2">{renderInline(trimmed.slice(3))}</h2>);
      i += 1;
      continue;
    }
    if (trimmed.startsWith('# ')) {
      elements.push(<h1 key={key++} className="markdown-h1">{renderInline(trimmed.slice(2))}</h1>);
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
                {headerCells.map((cell, idx) => renderTableCell(cell, `h-${idx}`, 'th'))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIdx) => (
                <tr key={`r-${rowIdx}`}>
                  {headerCells.map((_, cellIdx) => renderTableCell(row[cellIdx] || '', `c-${rowIdx}-${cellIdx}`))}
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
  const latestText = items[items.length - 1];
  return {
    latestStatus: 'INFO',
    latestMessage: latestText?.message || latestText?.raw || '대기 중',
  };
}

function metricCard(icon, label, value) {
  return (
    <div className="trace-metric-card">
      <div className="trace-metric-icon">{icon}</div>
      <div>
        <div className="trace-metric-label">{label}</div>
        <div className="trace-metric-value">{value ?? '-'}</div>
      </div>
    </div>
  );
}

function chipList(items, strong = false) {
  if (!items || items.length === 0) return <div className="trace-empty-box">데이터 없음</div>;
  return (
    <div className="trace-chip-wrap">
      {items.map((item) => (
        <span key={String(item)} className={`trace-chip ${strong ? 'trace-chip-strong' : ''}`}>
          {String(item)}
        </span>
      ))}
    </div>
  );
}

function JsonPreview({ data }) {
  if (data == null) return <div className="trace-empty-box">데이터 없음</div>;
  return <pre className="trace-json-preview">{JSON.stringify(data, null, 2)}</pre>;
}

function PlanCandidateCard({ candidate, index }) {
  const breakdown = candidate?.score_breakdown || {};
  return (
    <div className="trace-candidate-card">
      <div className="trace-candidate-top">
        <div>
          <div className="trace-candidate-title">
            {index + 1}. {candidate?.company || '-'} / {candidate?.product_name || '-'} / {candidate?.rider_name || '-'}
          </div>
          <div className="trace-candidate-subtitle">
            eligible: {String(candidate?.is_eligible)}
            {candidate?.ineligible_reason ? ` · ${candidate.ineligible_reason}` : ''}
          </div>
        </div>
        <div className="trace-score-badge">{breakdown.final_score ?? '-'}</div>
      </div>
      <div className="trace-chip-wrap">
        <span className="trace-chip">benefit {breakdown.benefit_match_score ?? '-'}</span>
        <span className="trace-chip">condition {breakdown.condition_clarity_score ?? '-'}</span>
        <span className="trace-chip">exclusion {breakdown.exclusion_penalty ?? '-'}</span>
        <span className="trace-chip">coverage {breakdown.coverage_breadth_score ?? '-'}</span>
        <span className="trace-chip">user-fit {breakdown.user_filter_match_score ?? '-'}</span>
      </div>
    </div>
  );
}

function SelectedCandidateCard({ candidate, index }) {
  const scoringBasis = candidate?.scoring_basis || [];
  return (
    <div className="trace-log-card">
      <div className="trace-log-title-row">
        <div>
          <div className="trace-log-title">#{index + 1} {candidate.company} / {candidate.product_name}</div>
          <div className="trace-log-subtitle">{candidate.rider_name || '-'}{candidate.renewal_type ? ` · ${candidate.renewal_type}` : ''}</div>
        </div>
        <span className="trace-log-pill">selected</span>
      </div>
      {scoringBasis.length > 0 ? (
        <div className="trace-chip-wrap mt-2">
          {scoringBasis.map((item, basisIndex) => (
            <span key={`${item.category}-${basisIndex}`} className="trace-chip">{item.category}</span>
          ))}
        </div>
      ) : null}
      <JsonPreview data={candidate} />
    </div>
  );
}

function TaskResultCard({ item }) {
  const evidence = item?.evidence || [];
  return (
    <div className="trace-log-card">
      <div className="trace-log-title-row">
        <div>
          <div className="trace-log-title">{item?.title || item?.task_type}</div>
          <div className="trace-log-subtitle">
            status {item?.status || '-'} · evidence {item?.evidence_count ?? evidence.length} · {item?.duration_ms ?? '-'}ms
          </div>
        </div>
        <span className="trace-log-pill">{item?.task_type}</span>
      </div>
      <div className="trace-task-item-meta">{item?.summary || '-'}</div>
      {evidence.length > 0 && (
        <details className="trace-details-block">
          <summary>evidence 보기</summary>
          <JsonPreview data={evidence} />
        </details>
      )}
    </div>
  );
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [logs, setLogs] = useState([]);
  const [debugData, setDebugData] = useState(null);

  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const parsedLogs = useMemo(() => logs.map((log, index) => parseLog(log, index)), [logs]);
  const traceSummary = useMemo(() => summarizeLogs(parsedLogs), [parsedLogs]);
  const selectedCandidates = useMemo(() => {
    const grouped = debugData?.answer_skeleton?.grouped_candidates || [];
    if (grouped.length > 0) {
      return grouped.flatMap((group) => group.candidates || []);
    }
    return debugData?.answer_skeleton?.candidates || [];
  }, [debugData]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading || isStreaming) return;
    const userMessage = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setLogs([]);
    setDebugData(null);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assistantMsgAdded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          let event;
          try { event = JSON.parse(jsonStr); } catch { continue; }

          if (event.type === 'chunk') {
            if (!assistantMsgAdded) {
              assistantMsgAdded = true;
              setIsLoading(false);
              setIsStreaming(true);
              setMessages((prev) => [...prev, { role: 'assistant', text: event.text }]);
            } else {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { role: 'assistant', text: next[next.length - 1].text + event.text };
                return next;
              });
            }
          } else if (event.type === 'done') {
            if (!assistantMsgAdded) {
              setMessages((prev) => [...prev, { role: 'assistant', text: event.answer || '응답이 없습니다.' }]);
            } else {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { role: 'assistant', text: event.answer || '응답이 없습니다.' };
                return next;
              });
            }
            setLogs(event.logs || []);
            setDebugData(event);
            setIsLoading(false);
            setIsStreaming(false);
          } else if (event.type === 'error') {
            const errText = '요청 처리 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요.';
            if (!assistantMsgAdded) {
              setMessages((prev) => [...prev, { role: 'assistant', text: errText }]);
            } else {
              setMessages((prev) => {
                const next = [...prev];
                next[next.length - 1] = { role: 'assistant', text: errText };
                return next;
              });
            }
            setIsLoading(false);
            setIsStreaming(false);
          }
        }
      }
    } catch (error) {
      setMessages((prev) => {
        const next = [...prev];
        if (next[next.length - 1]?.role === 'assistant') {
          next[next.length - 1] = { role: 'assistant', text: '요청 처리 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요.' };
        } else {
          next.push({ role: 'assistant', text: '요청 처리 중 오류가 발생했습니다. 백엔드 로그를 확인해 주세요.' });
        }
        return next;
      });
      setLogs((prev) => [...prev, `[System Error] ${error.message}`]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const renderLogItem = (log, index) => {
    if (log.type !== 'json') {
      return (
        <div key={index} className="trace-log-card">
          <div className="trace-log-title">{log.message}</div>
        </div>
      );
    }
    return (
      <div key={index} className="trace-log-card">
        <div className="trace-log-title-row">
          <div>
            <div className="trace-log-title">{log.message || `trace-${index}`}</div>
            <div className="trace-log-subtitle">structured log</div>
          </div>
          <span className="trace-log-pill">json</span>
        </div>
        <JsonPreview data={log} />
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
            <p className="chat-header-subtitle">Grounded insurance QA</p>
          </div>
        </header>

        <div className="chat-body">
          {messages.length === 0 && (
            <div className="empty-state">
              <Bot size={52} className="mx-auto mb-4 opacity-50" />
              <p className="empty-state-title">무엇이든 물어보세요</p>
              <p className="empty-state-desc">
                추천·비교는 검색 근거 안의 회사/상품/특약만 사용하고, 오른쪽 패널에 분석·검색·점수화 결과를 함께 표시합니다.
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-3 max-w-[82%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-gray-100 text-gray-800 border border-gray-200' : 'bg-slate-900 text-white'}`}>
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm break-words ${msg.role === 'user' ? 'bg-gray-100 text-gray-800 border border-gray-200 rounded-tr-none' : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'}`}>
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
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-75" />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-150" />
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
            <button onClick={sendMessage} disabled={isLoading || isStreaming || !input.trim()} className="chat-send-btn">
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
          <div className="trace-header-subtitle">의도 분석 · 검색 결과 · 점수화 · 근거 사용 위치</div>
        </header>

        <div className="trace-summary-grid">
          {metricCard(<Hash size={16} />, 'Request ID', debugData?.request_id || '-')}
          {metricCard(<Cpu size={16} />, 'Intent', debugData?.intent || '-')}
          {metricCard(<ListChecks size={16} />, 'Tasks', (debugData?.tasks || []).length)}
          {metricCard(<TimerReset size={16} />, 'Latest', traceSummary.latestMessage)}
        </div>

        <div className="trace-log-list">
          <div className="trace-section">
            <div className="trace-section-title">Analyzer</div>
            {chipList(debugData?.task_candidates || [], true)}
            <div className="trace-chip-wrap mt-2">
              {(debugData?.required_tasks || []).map((task) => (
                <span key={task} className="trace-chip">required {task}</span>
              ))}
            </div>
            <div className="trace-chip-wrap mt-2">
              {(debugData?.concept_keywords || []).map((item) => (
                <span key={item} className="trace-chip">concept {item}</span>
              ))}
              {(debugData?.product_keywords || []).map((item) => (
                <span key={item} className="trace-chip">product {item}</span>
              ))}
            </div>
            <details className="trace-details-block">
              <summary>user_filters 보기</summary>
              <JsonPreview data={debugData?.user_filters || {}} />
            </details>
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Grounder</div>
            {(debugData?.resolved_concepts || []).length > 0 ? (
              (debugData?.resolved_concepts || []).map((item, idx) => (
                <div key={idx} className="trace-log-card">
                  <div className="trace-log-title">{item.keyword} → {item.label_ko || item.concept_id}</div>
                  <div className="trace-log-subtitle">score {item.score} · {item.category || '-'}</div>
                </div>
              ))
            ) : (
              <div className="trace-empty-box">resolved concept 없음</div>
            )}
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Planner</div>
            {(debugData?.task_plan || []).length > 0 ? (
              (debugData?.task_plan || []).map((task) => (
                <div key={task.task_id} className="trace-log-card">
                  <div className="trace-log-title">{task.task_id} · {task.task_type}</div>
                  <div className="trace-log-subtitle">depends_on {Array.isArray(task.depends_on) ? task.depends_on.join(', ') || '-' : '-'}</div>
                  <JsonPreview data={task.inputs || {}} />
                </div>
              ))
            ) : (
              <div className="trace-empty-box">task plan 없음</div>
            )}
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Plan Scoring</div>
            {(debugData?.plan_candidates || []).length > 0 ? (
              (debugData?.plan_candidates || []).map((candidate, index) => (
                <PlanCandidateCard key={`${candidate.product_name}-${candidate.rider_name}-${index}`} candidate={candidate} index={index} />
              ))
            ) : (
              <div className="trace-empty-box">점수화된 후보 없음</div>
            )}
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Executor Results</div>
            {(debugData?.task_results || []).length > 0 ? (
              (debugData?.task_results || []).map((item) => <TaskResultCard key={item.task_id} item={item} />)
            ) : (
              <div className="trace-empty-box">task 결과 없음</div>
            )}
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Generator Input</div>
            {selectedCandidates.length > 0 ? (
              selectedCandidates.map((candidate, index) => (
                <SelectedCandidateCard key={`${candidate.product_name}-${candidate.rider_name}-${index}`} candidate={candidate} index={index} />
              ))
            ) : (
              <div className="trace-empty-box">generator에 전달된 후보 없음</div>
            )}
            <details className="trace-details-block" open>
              <summary>answer_skeleton 보기</summary>
              <JsonPreview data={debugData?.answer_skeleton || {}} />
            </details>
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Guard & Allowed Entities</div>
            <details className="trace-details-block" open>
              <summary>guarded_sections 보기</summary>
              <JsonPreview data={debugData?.guarded_sections || []} />
            </details>
            <details className="trace-details-block">
              <summary>allowed_entities 보기</summary>
              <JsonPreview data={debugData?.allowed_entities || {}} />
            </details>
          </div>

          <div className="trace-section">
            <div className="trace-section-title">Raw Trace Logs</div>
            {parsedLogs.length === 0 ? (
              <div className="trace-empty-state">대기 중...\n워크플로우 실행 기록이 여기에 표시됩니다.</div>
            ) : (
              parsedLogs.map((log, idx) => renderLogItem(log, idx))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </aside>
    </div>
  );
}

export default App;
