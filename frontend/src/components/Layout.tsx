import { useState } from 'react';
import { Outlet, useNavigate, Link, useLocation } from 'react-router-dom';
import { Search } from 'lucide-react';

export const Layout = () => {
    const [query, setQuery] = useState("");
    const navigate = useNavigate();
    const location = useLocation();

    // Helper to style active vs inactive links
    const getLinkClass = (path: string) => {
        const isActive = location.pathname === path;
        return isActive
            ? "bg-white text-black px-5 py-3 rounded-full font-bold text-sm transition"
            : "text-white hover:bg-white/10 px-5 py-3 rounded-full font-bold text-sm transition";
    };

    const handleSearch = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            window.location.href = `/?q=${query}`;
        }
    };

    return (
        <div className="min-h-screen bg-black text-white font-sans">
            {/* Persistent Header */}
            <nav className="sticky top-0 z-50 bg-black py-4 px-6 border-b border-white/10">
                <div className="flex items-center gap-4 max-w-[1600px] mx-auto">
                    <Link to="/" className="p-3 rounded-full hover:bg-white/10 transition flex-shrink-0">
                        <div className="w-6 h-6 bg-red-600 rounded-full flex items-center justify-center font-bold text-xs text-white">P</div>
                    </Link>

                    {/* Navigation Links */}
                    <div className="hidden md:flex items-center gap-2">
                        <Link to="/" className={getLinkClass('/')}>
                            Home
                        </Link>
                        <Link to="/people" className={getLinkClass('/people')}>
                            People
                        </Link>
                    </div>

                    <div className="flex-1 relative">
                        <Search className="absolute left-4 top-3.5 w-5 h-5 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleSearch}
                            className="w-full bg-[#1e1e1e] text-white rounded-full py-3 pl-12 pr-4 hover:bg-[#2a2a2a] focus:bg-[#2a2a2a] focus:outline-none focus:ring-2 focus:ring-white/20 transition-all"
                        />
                    </div>
                </div>
            </nav>

            <Outlet />
        </div>
    );
};