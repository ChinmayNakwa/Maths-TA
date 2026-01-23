'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/src/lib/utils';
import { Sparkles } from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();
  const navItems = [
    { name: 'Mission', href: '/' },
    { name: 'Tutor', href: '/chat' },
    { name: 'Architecture', href: '/about' },
  ];

  return (
    <nav className="fixed top-0 w-full z-50 flex items-center justify-between px-6 py-4 glass border-b-0">
      <Link href="/" className="flex items-center gap-2 group">
        <div className="bg-gradient-to-tr from-indigo-500 to-purple-500 p-1.5 rounded-lg group-hover:scale-110 transition-transform duration-300">
            <Sparkles size={18} className="text-white" fill="white" />
        </div>
        <span className="font-bold text-xl tracking-tight">
          Cerebrix
        </span>
      </Link>
      
      <div className="flex gap-1 bg-surface-highlight/50 p-1 rounded-full border border-white/5">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-300",
              pathname === item.href 
                ? "bg-white/10 text-white shadow-lg shadow-purple-500/10" 
                : "text-zinc-400 hover:text-white hover:bg-white/5"
            )}
          >
            {item.name}
          </Link>
        ))}
      </div>
    </nav>
  );
}