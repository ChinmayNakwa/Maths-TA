'use client';
import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
    Send, Mic, Image as ImageIcon, StopCircle, Sparkles, 
    X, Plus, Menu, MessageSquare, Trash2 
} from 'lucide-react';
import MathRenderer from '@/src/components/MathRenderer';

// --- Types ---
type Message = {
    role: 'user' | 'ta';
    content: string;
    sources?: Array<{ location: string; url: string }>;
};

type ChatSession = {
    id: string;
    preview: string;
    timestamp: number;
};

export default function ChatPage() {
    // --- State ---
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [sessions, setSessions] = useState<ChatSession[]>([]);
    
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);

    // --- Initialization ---
    useEffect(() => {
        const savedSessionsRaw = localStorage.getItem('my_chat_sessions');
        let parsedSessions: ChatSession[] = savedSessionsRaw ? JSON.parse(savedSessionsRaw) : [];
        setSessions(parsedSessions);

        let currentId = sessionStorage.getItem('chatSessionId');
        
        if (!currentId) {
            currentId = createNewSessionId();
            sessionStorage.setItem('chatSessionId', currentId);
            
            if (!parsedSessions.find(s => s.id === currentId)) {
                const newSession = { id: currentId, preview: "New Conversation", timestamp: Date.now() };
                parsedSessions = [newSession, ...parsedSessions];
                setSessions(parsedSessions);
                localStorage.setItem('my_chat_sessions', JSON.stringify(parsedSessions));
            }
        }
        
        setSessionId(currentId);
        fetchChatHistory(currentId);
    }, []);

    // --- Helpers ---
    const createNewSessionId = () => {
        if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
            return `session-${crypto.randomUUID()}`;
        }
        return `session-${Date.now().toString(36)}`;
    };

    const fetchChatHistory = async (id: string) => {
        setIsLoading(true);
        setMessages([]); // Clear previous messages while loading
        try {
            const res = await fetch(`/api/py/history/${id}`);
            if (!res.ok) throw new Error("Failed to load history");
            
            const data = await res.json();
            
            if (data.history && data.history.length > 0) {
                const formattedMessages: Message[] = data.history.map((msg: any) => ({
                    role: msg.role === 'assistant' ? 'ta' : msg.role,
                    content: msg.content,
                    sources: msg.sources || []
                }));
                setMessages(formattedMessages);
            } else {
                setMessages([{ role: 'ta', content: "Hello! I'm **Cerebrix**, your AI tutor for CS109. How can I help you today?" }]);
            }
        } catch (error) {
            console.error(error);
            setMessages([{ role: 'ta', content: "Hello! I'm Cerebrix. (Could not load history, starting fresh)." }]);
        } finally {
            setIsLoading(false);
        }
    };

    // --- Actions ---

    const handleNewChat = () => {
        const newId = createNewSessionId();
        const newSession = { id: newId, preview: "New Conversation", timestamp: Date.now() };
        
        const updatedSessions = [newSession, ...sessions];
        setSessions(updatedSessions);
        localStorage.setItem('my_chat_sessions', JSON.stringify(updatedSessions));
        
        sessionStorage.setItem('chatSessionId', newId);
        setSessionId(newId);
        setMessages([{ role: 'ta', content: "Hello! I'm **Cerebrix**. New session started. How can I help?" }]);
        
        if (window.innerWidth < 768) setIsSidebarOpen(false);
    };

    const handleSelectSession = (id: string) => {
        if (id === sessionId) return;
        sessionStorage.setItem('chatSessionId', id);
        setSessionId(id);
        fetchChatHistory(id);
        if (window.innerWidth < 768) setIsSidebarOpen(false);
    };

    const handleDeleteSession = async (e: React.MouseEvent, idToDelete: string) => {
        e.stopPropagation(); // Prevent triggering handleSelectSession
        
        if (!confirm("Are you sure you want to delete this chat history?")) return;

        // 1. Update Frontend State immediately
        const updatedSessions = sessions.filter(s => s.id !== idToDelete);
        setSessions(updatedSessions);
        localStorage.setItem('my_chat_sessions', JSON.stringify(updatedSessions));

        // 2. Handle active session logic
        if (idToDelete === sessionId) {
            if (updatedSessions.length > 0) {
                // Switch to the first available session
                handleSelectSession(updatedSessions[0].id);
            } else {
                // No sessions left, create a fresh one
                handleNewChat();
            }
        }

        // 3. Call Backend to delete from DB
        try {
            await fetch(`/api/py/history/${idToDelete}`, { method: 'DELETE' });
        } catch (err) {
            console.error("Failed to delete from backend:", err);
            // Optional: Show a toast notification here
        }
    };

    const updateSessionPreview = (id: string, text: string) => {
        const updated = sessions.map(s => 
            s.id === id ? { ...s, preview: text.substring(0, 30) + (text.length > 30 ? '...' : '') } : s
        );
        const current = updated.find(s => s.id === id);
        const others = updated.filter(s => s.id !== id);
        const finalSort = current ? [current, ...others] : others;
        
        setSessions(finalSort);
        localStorage.setItem('my_chat_sessions', JSON.stringify(finalSort));
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e?: React.FormEvent) => {
        e?.preventDefault();
        if ((!input.trim() && !selectedImage) || isLoading) return;

        const userMsg = input;
        const currentImage = selectedImage;

        setInput('');
        setSelectedImage(null);
        setMessages(prev => [...prev, { role: 'user', content: userMsg || "[Image Uploaded]" }]);
        updateSessionPreview(sessionId, userMsg || "Image Query");

        setIsLoading(true);

        try {
            let imageData = null;
            if (currentImage) imageData = await toBase64(currentImage);

            const res = await fetch('/api/py/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: userMsg, session_id: sessionId, image_data: imageData })
            });

            const data = await res.json();
            if(!res.ok) throw new Error(data.detail || "Error");

            setMessages(prev => [...prev, { role: 'ta', content: data.answer, sources: data.sources }]);
        } catch (error) {
            console.error(error);
            setMessages(prev => [...prev, { role: 'ta', content: "I'm having trouble connecting to my knowledge base." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const toggleRecording = async () => {
        if (isRecording) {
            mediaRecorderRef.current?.stop();
            setIsRecording(false);
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const recorder = new MediaRecorder(stream);
                const chunks: BlobPart[] = [];
                recorder.ondataavailable = e => chunks.push(e.data);
                recorder.onstop = async () => {
                    const blob = new Blob(chunks, { type: 'audio/webm' });
                    const formData = new FormData();
                    formData.append("audio_file", blob, "recording.webm");
                    setIsLoading(true);
                    try {
                        const res = await fetch('/api/py/transcribe', { method: 'POST', body: formData });
                        const data = await res.json();
                        setInput(data.transcription);
                    } catch (e) { console.error(e); } 
                    finally { setIsLoading(false); }
                    stream.getTracks().forEach(track => track.stop());
                };
                recorder.start();
                mediaRecorderRef.current = recorder;
                setIsRecording(true);
            } catch (err) { alert("Microphone access denied."); }
        }
    };

    const toBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve((reader.result as string).split(',')[1]);
            reader.onerror = error => reject(error);
        });
    };

    return (
        <div className="flex h-[calc(100vh-80px)] overflow-hidden">
            
            {/* --- LEFT SIDEBAR --- */}
            <AnimatePresence mode='wait'>
                {isSidebarOpen && (
                    <motion.aside 
                        initial={{ width: 0, opacity: 0 }} 
                        animate={{ width: 280, opacity: 1 }} 
                        exit={{ width: 0, opacity: 0 }}
                        className="bg-black/20 border-r border-white/5 backdrop-blur-md flex flex-col h-full z-30 absolute md:relative"
                    >
                        <div className="p-4">
                            <button 
                                onClick={handleNewChat}
                                className="w-full flex items-center justify-center gap-2 bg-indigo-600/20 hover:bg-indigo-600/40 border border-indigo-500/30 text-indigo-100 p-3 rounded-xl transition-all shadow-sm"
                            >
                                <Plus size={18} />
                                <span className="text-sm font-medium">New Chat</span>
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto px-2 space-y-1 scrollbar-thin scrollbar-thumb-white/10">
                            <div className="px-2 py-2 text-xs font-semibold text-zinc-500 uppercase tracking-wider">Recent History</div>
                            {sessions.map((session) => (
                                <div
                                    key={session.id}
                                    onClick={() => handleSelectSession(session.id)}
                                    className={`group relative w-full text-left p-3 rounded-lg text-sm transition-colors flex items-center gap-3 cursor-pointer ${
                                        sessionId === session.id 
                                        ? 'bg-white/10 text-white shadow-inner' 
                                        : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
                                    }`}
                                >
                                    <MessageSquare size={16} className="shrink-0" />
                                    <div className="truncate w-full pr-6">
                                        {session.preview}
                                    </div>

                                    {/* Delete Button (Visible on Hover) */}
                                    <button 
                                        onClick={(e) => handleDeleteSession(e, session.id)}
                                        className="absolute right-2 p-1.5 text-zinc-500 hover:text-red-400 hover:bg-red-400/10 rounded-md opacity-0 group-hover:opacity-100 transition-opacity"
                                        title="Delete chat"
                                    >
                                        <Trash2 size={14} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </motion.aside>
                )}
            </AnimatePresence>

            {/* --- MAIN CHAT AREA --- */}
            <main className="flex-1 flex flex-col relative min-w-0">
                
                {/* Mobile/Sidebar Toggle Header */}
                <div className="absolute top-4 left-4 z-20">
                     <button 
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
                        className="p-2 bg-black/40 backdrop-blur-sm border border-white/10 rounded-lg text-zinc-400 hover:text-white transition"
                    >
                        {isSidebarOpen ? <X size={18} /> : <Menu size={18} />}
                    </button>
                </div>

                {/* Messages Feed */}
                <div className="flex-grow overflow-y-auto p-4 md:p-6 space-y-6 pt-14 md:pt-6">
                    {messages.map((msg, idx) => (
                        <motion.div key={idx} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`max-w-[90%] md:max-w-[75%] p-5 rounded-3xl backdrop-blur-sm shadow-sm ${
                                msg.role === 'user' 
                                ? 'bg-indigo-600/20 border border-indigo-500/30 text-white rounded-br-sm' 
                                : 'glass text-zinc-100 rounded-bl-sm'
                            }`}>
                                <div className="flex items-center gap-2 mb-2 opacity-50 text-xs font-bold uppercase tracking-wider">
                                    {msg.role === 'user' ? 'You' : <><Sparkles size={12} /> Cerebrix</>}
                                </div>
                                <MathRenderer content={msg.content} />
                                
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-4 pt-3 border-t border-white/5 flex flex-wrap gap-2">
                                        {msg.sources.map((src, i) => (
                                            <div key={i} className="text-[10px] uppercase tracking-wide bg-black/40 px-3 py-1.5 rounded-full border border-white/10 text-zinc-400">
                                                {src.location}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    ))}
                    {isLoading && (
                        <div className="flex items-center gap-2 text-zinc-500 text-sm ml-4 bg-surface/50 px-4 py-2 rounded-full w-fit">
                            <Sparkles size={14} className="animate-pulse text-indigo-400"/>
                            <span>Thinking...</span>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 md:p-6 z-20 relative">
                     <div className="max-w-4xl mx-auto">
                        {selectedImage && (
                            <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} className="absolute bottom-full left-4 md:left-6 mb-3 p-3 glass rounded-xl flex items-center gap-3">
                                <div className="w-8 h-8 bg-white/10 rounded flex items-center justify-center"><ImageIcon size={14}/></div>
                                <span className="text-xs text-zinc-300 truncate max-w-[200px]">{selectedImage.name}</span>
                                <button onClick={() => setSelectedImage(null)} className="hover:text-red-400"><X size={14}/></button>
                            </motion.div>
                        )}
                        
                        <form onSubmit={handleSubmit} className="glass rounded-2xl p-2 flex items-center gap-2 shadow-2xl shadow-indigo-500/5 transition-all focus-within:border-indigo-500/50">
                            <button type="button" onClick={() => fileInputRef.current?.click()} className="p-3 hover:bg-white/10 rounded-xl text-zinc-400 hover:text-white transition">
                                <ImageIcon size={20} />
                            </button>
                            <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => { if(e.target.files?.[0]) setSelectedImage(e.target.files[0]); }} />
                            
                            <input 
                                type="text" 
                                value={input} 
                                onChange={(e) => setInput(e.target.value)} 
                                placeholder="Ask a question..." 
                                className="flex-grow bg-transparent border-none outline-none text-white placeholder-zinc-500 px-2 py-2" 
                            />
                            
                            <button type="button" onClick={toggleRecording} className={`p-3 rounded-xl transition ${isRecording ? 'text-red-500 animate-pulse bg-red-500/10' : 'text-zinc-400 hover:text-white hover:bg-white/10'}`}>
                                {isRecording ? <StopCircle size={20} /> : <Mic size={20} />}
                            </button>
                            
                            <button type="submit" disabled={isLoading || (!input && !selectedImage)} className="bg-white text-black p-3 rounded-xl hover:scale-105 active:scale-95 transition disabled:opacity-50 disabled:scale-100">
                                <Send size={20} />
                            </button>
                        </form>
                        <div className="text-center mt-2 text-xs text-zinc-600">
                            Cerebrix can make mistakes. Check important info.
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}