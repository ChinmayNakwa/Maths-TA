'use client';
import { motion } from 'framer-motion';

export default function AboutPage() {
    const stack = [
        { name: "LangGraph", desc: "Stateful Agent Architecture" },
        { name: "Google Gemini 2.5", desc: "Reasoning & Vision Model" },
        { name: "AstraDB", desc: "Vector Database" },
        { name: "FastAPI", desc: "Backend Server" },
        { name: "Next.js 16", desc: "Frontend Framework" },
        { name: "Cloudinary", desc: "Image & Media Hosting" },
    ];

  return (
    <div className="min-h-screen pt-24 px-6 max-w-4xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl md:text-5xl font-bold mb-6">Inside <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">Cerebrix</span></h1>
        
        <div className="glass p-8 rounded-3xl mb-8 space-y-6 text-lg text-secondary leading-relaxed">
          <p>
            While studying Stanford's CS109 on YouTube, I encountered a common frustration: 
            there was no way to discuss problems or get guidance when attempting exercises. 
            <strong>Cerebrix</strong> emerged from that need.
          </p>
          <p>
            It is a course-agnostic architecture that transforms one-way video lectures into 
            an interactive, two-way educational dialogue.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <div className="space-y-4">
                <h3 className="text-2xl font-bold text-white">Core Capabilities</h3>
                <ul className="space-y-3 text-zinc-400">
                    <li className="flex items-start gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2.5"/>
                        <span><strong>Multi-Modal RAG:</strong> Ingests PDFs and Video content to create a comprehensive knowledge base.</span>
                    </li>
                    <li className="flex items-start gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-2.5"/>
                        <span><strong>Socratic Engine:</strong> Analyzes user inputs and images to provide hints, not just answers.</span>
                    </li>
                    <li className="flex items-start gap-3">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2.5"/>
                        <span><strong>Visual Problem Solving:</strong> Upload handwritten math or screenshots for instant AI feedback.</span>
                    </li>
                </ul>
            </div>

            <div className="space-y-4">
                <h3 className="text-2xl font-bold text-white">Tech Stack</h3>
                <div className="grid grid-cols-1 gap-2">
                    {stack.map((tech) => (
                        <div key={tech.name} className="flex items-center justify-between p-3 glass rounded-lg border-none bg-white/5">
                            <span className="font-semibold text-zinc-200">{tech.name}</span>
                            <span className="text-xs text-zinc-500 uppercase tracking-wider">{tech.desc}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
      </motion.div>
    </div>
  );
}