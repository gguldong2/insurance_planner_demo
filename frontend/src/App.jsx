import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Bot, User, Terminal, Database, ShieldAlert, Cpu } from 'lucide-react';

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [logs, setLogs] = useState([]); 
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const logsEndRef = useRef(null);

  // 스크롤 자동 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;   // 빈 입력 방지

    
    // 1. 사용자 메시지를 화면에 먼저 표시 (즉시 반응)
    const userMsg = { role: "user", text: input };
    setMessages(prev => [...prev, userMsg]);   // 리스트에 append 하는 방식
    setInput("");      // 입력창 비우기
    setIsLoading(true);    // 로딩 시작
    setLogs([]); // 새 질문 시 로그 초기화

    try {
      // Backend API 호출 (FastAPI 포트 확인: 8080)
      // [Python 비유] response = requests.post("...", json={"query": ...})
      const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";
      // [Python 비유] response = requests.post("...", json={"query": ...})
      const response = await axios.post(`${API_BASE}/chat`, {
        query: userMsg.text
      });

      const data = response.data;   // 백엔드에서 온 JSON 데이터
      
      // 백엔드에서 받은 logs와 answer로 상태 업데이트 -> 화면이 바뀜
      setLogs(data.logs || []);
      
      // 답변 업데이트
      const botMsg = { role: "bot", text: data.answer };
      setMessages(prev => [...prev, botMsg]);

    } catch (error) {
      // 에러 처리
      console.error(error);
      setMessages(prev => [...prev, { role: "bot", text: "❌ Error: 서버와 연결할 수 없습니다." }]);
      setLogs(prev => [...prev, `[System Error] ${error.message}`]);
    } finally {
      setIsLoading(false);   // 로딩 끝
    }
  };

  // 로그 키워드에 따른 아이콘/색상 렌더링
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
    // HTML처럼 보이지만 사실은 자바스크립트 코드(JSX)
    <div className="flex h-screen bg-gray-100 font-sans overflow-hidden">
      
      {/* 1. 채팅 영역 */}
      {/* messages 리스트를 순회하며(map) 화면에 그림 */}
      {/* 왼쪽: 채팅 영역 (65%) */}
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
              {/* Python의 f-string처럼 {변수명}을 쓰면 값이 들어갑니다 */}
              {/* {msg.text}  */}
              <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 
                  ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-white'}`}>
                  {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div className={`p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm
                  ${msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-800 border border-gray-200 rounded-tl-none'}`}>
                  {msg.text}
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
              value={input}    // input 변수와 연결
              onChange={(e) => setInput(e.target.value)}      // 타이핑할 때마다 변수 업데이트
              onKeyDown={(e) => e.key === 'Enter' && sendMessage()}    // 엔터키 처리
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
      
      {/* 3. 우측 로그 영역 */}
      {/* {logs.map((log, idx) => renderLogItem(log, idx))} */}
      {/* 오른쪽: 워크플로우 로그 패널 (35%) */}
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