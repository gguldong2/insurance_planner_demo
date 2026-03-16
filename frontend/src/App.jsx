import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Send, Bot, User, Terminal, Database, ShieldAlert, Cpu } from 'lucide-react';

function renderInline(text) {
  const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g).filter(Boolean);

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index} className="font-semibold text-gray-900">{part.slice(2, -2)}</strong>;
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

  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const elements = [];
  let i = 0;
  let key = 0;

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
      i += 1;
      elements.push(
        <pre key={key++} className="markdown-pre">
          <code>{codeLines.join('\n')}</code>
        </pre>
      );
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

    if (trimmed.startsWith('---')) {
      elements.push(<hr key={key++} className="my-4 border-gray-200" />);
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
      !/^#{1,3}\s/.test(lines[i].trim()) &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !/^>/.test(lines[i].trim()) &&
      !/^```/.test(lines[i].trim()) &&
      !/^---/.test(lines[i].trim())
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

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [logs, setLogs] = useState([]); 
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = { role: "user", text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setLogs([]);

    try {
      const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";
      const response = await axios.post(`${API_BASE}/chat`, {
        query: userMsg.text
      });

      const data = response.data;
      setLogs(data.logs || []);

      const botMsg = { role: "bot", text: data.answer };
      setMessages(prev => [...prev, botMsg]);

    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: "bot", text: "❌ Error: 서버와 연결할 수 없습니다." }]);
      setLogs(prev => [...prev, `[System Error] ${error.message}`]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderLogItem = (log, index) => {
    let icon = <Terminal size={14} />;
    let color = "text-gray-400";

    if (log.includes("[Router]")) { icon = <Cpu size={14} />; color = "text-yellow-400"; }
    else if (log.includes("Graph")) { icon = <Database size={14} />; color = "text-purple-400"; }
    else if (log.includes("SQL")) { icon = <Database size={14} />; color = "text-blue-400"; }
    else if (log.includes("[Critic]")) { icon = <ShieldAlert size={14} />; color = "text-red-400"; }
    else if (log.includes("Final")) { icon = <Bot size={14} />; color = "text-green-400"; }

    return (
      <div key={index} className={`flex items-start gap-2 mb-2 font-mono text-xs ${color} break-words`}>
        <span className="mt-0.5">{icon}</span>
        <span>{log}</span>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans overflow-hidden">
      <div className="flex-1 flex flex-col bg-white shadow-xl z-10">
        <header className="bg-slate-800 text-white p-4 flex items-center gap-3 shadow-md">
          <div className="p-2 bg-blue-600 rounded-full">
            <Bot size={24} />
          </div>
          <div>
            <h1 className="font-bold text-lg">AgensGraph Agent</h1>
            <p className="text-xs text-slate-400">Powered by LangGraph & vLLM</p>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-slate-50">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
              <Bot size={48} className="mx-auto mb-4 opacity-50" />
              <p>무엇이든 물어보세요. Agent가 DB를 분석하여 답변합니다.</p>
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 
                  ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-white'}`}>
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className={`p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap break-words
                  ${msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'}`}>
                  {msg.role === 'user' ? msg.text : renderMarkdown(msg.text)}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
               <div className="flex gap-3">
                <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center text-white">
                  <Bot size={16} />
                </div>
                <div className="bg-white border p-4 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-75" />
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce delay-150" />
                </div>
               </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-white border-t">
          <div className="flex gap-2 items-center bg-gray-100 p-2 rounded-full border border-gray-300 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-100 transition-all">
            <input 
              type="text"
              className="flex-1 bg-transparent border-none outline-none px-4 text-gray-700"
              placeholder="질문을 입력하세요..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button 
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
              className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
      
      <div className="w-[400px] bg-[#1e1e1e] flex flex-col border-l border-gray-700">
        <header className="p-3 border-b border-gray-700 bg-[#252526] text-gray-300 text-sm font-bold flex items-center gap-2">
          <Terminal size={16} />
          Agent Workflow Trace
        </header>
        <div className="flex-1 p-4 overflow-y-auto font-mono text-sm space-y-1">
          {logs.length === 0 ? (
            <div className="text-gray-600 text-center mt-10 text-xs">
              대기 중...<br/>워크플로우 실행 기록이 여기에 표시됩니다.
            </div>
          ) : (
            logs.map((log, idx) => renderLogItem(log, idx))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

    </div>
  );
}

export default App;
