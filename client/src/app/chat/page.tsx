'use client';
import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Mic, Image as ImageIcon, StopCircle, Sparkles, X } from 'lucide-react';
import MathRenderer from '@/src/components/MathRenderer';

type Message = {
    role: 'user' | 'ta';
    content: string;
    sources?: Array<{ location: string; url: string }>;
};

export default function ChatPage() {
    // Updated Greeting
    const [messages, setMessages] = useState<Message[]>([
        { role: 'ta', content: "Hello! I'm **Cerebrix**, your AI tutor for CS109. \n\nI can help you understand concepts from the lectures, guide you through homework problems socratically, or clarify textbook definitions. What are we working on today?" }
    ]);
    const [input, setInput] = useState('');
    const [sessionId, setSessionId] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [selectedImage, setSelectedImage] = useState<File | null>(null);
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);

    useEffect(() => {
        let storedSession = sessionStorage.getItem('chatSessionId');
        if (!storedSession) {
             if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
                storedSession = `session-${crypto.randomUUID()}`;
            } else {
                storedSession = `session-${Date.now().toString(36)}`;
            }
            sessionStorage.setItem('chatSessionId', storedSession);
        }
        setSessionId(storedSession);
    }, []);

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
            setMessages(prev => [...prev, { role: 'ta', content: "I'm having trouble connecting to my knowledge base. Is the backend server running?" }]);
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
        <div className="flex flex-col h-[calc(100vh-80px)] max-w-4xl mx-auto p-4 md:p-6">
            <div className="flex-grow overflow-y-auto mb-6 space-y-6 pr-2 scrollbar-hide">
                {messages.map((msg, idx) => (
                    <motion.div key={idx} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div className={`max-w-[85%] p-5 rounded-3xl backdrop-blur-sm shadow-sm ${
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

            <div className="relative z-20">
                {selectedImage && (
                    <motion.div initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} className="absolute bottom-full left-0 mb-3 p-3 glass rounded-xl flex items-center gap-3">
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
                        placeholder="Ask a question or upload a problem..." 
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
    );
}