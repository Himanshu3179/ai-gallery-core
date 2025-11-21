import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Pencil, Check } from 'lucide-react';
import { MasonryGrid } from '../components/MasonryGrid';
import type { ImageItem } from '../types';

const API_BASE = "http://localhost:8000";

export const PersonDetail = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    
    const [images, setImages] = useState<ImageItem[]>([]);
    const [personName, setPersonName] = useState("");
    const [isEditing, setIsEditing] = useState(false);
    const [newName, setNewName] = useState("");

    useEffect(() => {
        // Fetch the person's list to get their current name (optional, or store in location state)
        // For simplicity, we'll just default to "Person" until we implement a specific GET /people/{id} metadata endpoint
        // But we can infer it if we passed it via state, or just fetch the list again.
        // Let's fetch the list of people to find THIS person's name for now.
        fetch(`${API_BASE}/people`)
            .then(res => res.json())
            .then((data: any[]) => {
                const p = data.find(p => p.id === Number(id));
                if (p) {
                    setPersonName(p.name);
                    setNewName(p.name);
                }
            });

        // Fetch images for this person
        fetch(`${API_BASE}/people/${id}`)
            .then(res => res.json())
            .then(data => setImages(data));
    }, [id]);

    const handleRename = async () => {
        if (!newName.trim()) return;
        try {
            const res = await fetch(`${API_BASE}/people/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: newName })
            });
            if (res.ok) {
                setPersonName(newName);
                setIsEditing(false);
            }
        } catch (e) {
            console.error("Rename failed", e);
        }
    };

    return (
        <main className="min-h-screen bg-black">
            {/* Header Section */}
            <div className="sticky top-[73px] z-40 bg-black/95 backdrop-blur py-6 px-4 border-b border-white/10 mb-6">
                <div className="max-w-[1600px] mx-auto flex items-center gap-4">
                    <button 
                        onClick={() => navigate(-1)} 
                        className="p-2 hover:bg-white/10 rounded-full transition"
                    >
                        <ArrowLeft className="text-white" size={24} />
                    </button>

                    <div className="flex items-center gap-3">
                        {isEditing ? (
                            <div className="flex items-center gap-2">
                                <input 
                                    type="text" 
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    className="bg-[#333] text-white text-2xl font-bold px-3 py-1 rounded-lg focus:outline-none focus:ring-2 focus:ring-white/20"
                                    autoFocus
                                    onKeyDown={(e) => e.key === 'Enter' && handleRename()}
                                />
                                <button 
                                    onClick={handleRename}
                                    className="p-2 bg-green-600 hover:bg-green-700 rounded-full text-white"
                                >
                                    <Check size={20} />
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-3 group cursor-pointer" onClick={() => setIsEditing(true)}>
                                <h1 className="text-3xl font-bold text-white">{personName || "Loading..."}</h1>
                                <Pencil size={18} className="text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                        )}
                        <span className="text-gray-500 text-sm mt-2 ml-2">{images.length} photos</span>
                    </div>
                </div>
            </div>

            {/* Grid Section */}
            <div className="max-w-[1600px] mx-auto px-4">
                {images.length > 0 ? (
                    <MasonryGrid images={images} />
                ) : (
                    <div className="text-gray-500 text-center py-20">No photos found for this person.</div>
                )}
            </div>
        </main>
    );
};