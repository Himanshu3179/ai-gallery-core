import { useEffect, useState } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Pencil, Check } from 'lucide-react';
import { MasonryGrid } from '../components/MasonryGrid';
import type { ImageItem, Person } from '../types';

const API_BASE = "http://localhost:8000";

export const PersonDetail = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const location = useLocation();

    const [images, setImages] = useState<ImageItem[]>([]);
    const [personName, setPersonName] = useState("Person");
    const [isEditing, setIsEditing] = useState(false);
    const [newName, setNewName] = useState("");

    // Force scroll to top on mount
    useEffect(() => {
        window.scrollTo(0, 0);
    }, [id]);

    useEffect(() => {
        // 1. Get Name Instantly (No Fetch)
        if (location.state?.person) {
            // Priority A: Passed from previous page
            setPersonName(location.state.person.name);
            setNewName(location.state.person.name);
        } else {
            // Priority B: Fallback to cache if page refreshed
            const cachedPeople = sessionStorage.getItem("people_data_cache");
            if (cachedPeople) {
                const list: Person[] = JSON.parse(cachedPeople);
                const p = list.find((p) => p.id === Number(id));
                if (p) {
                    setPersonName(p.name);
                    setNewName(p.name);
                }
            }
        }

        // 2. Fetch Images
        fetch(`${API_BASE}/people/${id}`)
            .then(res => res.json())
            .then(data => setImages(data));
    }, [id, location.state]);

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

                // Update Cache
                const cachedPeople = sessionStorage.getItem("people_data_cache");
                if (cachedPeople) {
                    const list = JSON.parse(cachedPeople);
                    const idx = list.findIndex((p: any) => p.id === Number(id));
                    if (idx !== -1) {
                        list[idx].name = newName;
                        sessionStorage.setItem("people_data_cache", JSON.stringify(list));
                    }
                }
            }
        } catch (e) {
            console.error("Rename failed", e);
        }
    };

    return (
        <main className="min-h-screen bg-black">
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
                                <h1 className="text-3xl font-bold text-white">{personName}</h1>
                                <Pencil size={18} className="text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                        )}
                        <span className="text-gray-500 text-sm mt-2 ml-2">{images.length} photos</span>
                    </div>
                </div>
            </div>

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