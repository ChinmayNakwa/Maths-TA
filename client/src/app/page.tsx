'use client';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight, BrainCircuit, Youtube, BookOpen, MessageSquareText, Layers } from 'lucide-react';

export default function LandingPage() {
  const features = [
    {
      icon: <Youtube className="text-red-500" />,
      title: "Video Intelligence",
      desc: "Ingests full YouTube playlists and connects concepts across lectures."
    },
    {
      icon: <BookOpen className="text-blue-500" />,
      title: "Textbook RAG",
      desc: "Retrieves precise context from PDF course materials to ground answers."
    },
    {
      icon: <BrainCircuit className="text-purple-500" />,
      title: "Socratic Tutoring",
      desc: "Doesn't just give answers. It guides your thinking with strategic hints."
    },
    {
      icon: <MessageSquareText className="text-green-500" />,
      title: "Conversational Memory",
      desc: "Remembers your previous questions for a natural, flowing dialogue."
    }
  ];

  return (
    <div className="flex flex-col items-center min-h-screen pt-20 px-4 relative overflow-hidden">
        
        {/* Hero Section */}
        <section className="text-center max-w-4xl mx-auto mt-16 mb-24 relative z-10">
            <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-accent mb-6"
            >
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
                </span>
                Now Live: CS109 Probability
            </motion.div>

            <motion.h1 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-5xl md:text-7xl font-bold tracking-tight mb-6 leading-[1.1]"
            >
                Your AI Teaching Assistant for <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400 animate-gradient">
                    Active Learning
                </span>
            </motion.h1>
            
            <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-secondary text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed"
            >
                Cerebrix transforms passive video lectures into interactive conversations. 
                Upload problems, ask questions, and get Socratic guidance 24/7.
            </motion.p>
            
            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="flex flex-col sm:flex-row gap-4 justify-center"
            >
                <Link href="/chat">
                    <button className="group relative px-8 py-4 bg-white text-black rounded-full font-bold text-lg overflow-hidden transition-all hover:scale-105">
                        <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-500 opacity-0 group-hover:opacity-10 transition-opacity" />
                        <span className="flex items-center gap-2">
                            Start Tutoring <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform"/>
                        </span>
                    </button>
                </Link>
                <Link href="/about">
                    <button className="px-8 py-4 rounded-full font-bold text-lg glass text-white hover:bg-white/10 transition-all">
                        View Architecture
                    </button>
                </Link>
            </motion.div>
        </section>

        {/* Features Grid */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-5xl w-full mb-24">
            {features.map((f, i) => (
                <motion.div 
                    key={i}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="glass glass-hover p-6 rounded-2xl transition-all duration-300 group"
                >
                    <div className="bg-surface-highlight w-12 h-12 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                        {f.icon}
                    </div>
                    <h3 className="text-xl font-bold mb-2">{f.title}</h3>
                    <p className="text-secondary leading-relaxed">{f.desc}</p>
                </motion.div>
            ))}
        </section>
    </div>
  );
}