import { useEffect, useState } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, X, Send, Link as LinkIcon, MoreHorizontal } from 'lucide-react';
import { Pin } from '../components/Pin';
import type { ImageItem } from '../types';

const API_BASE = "http://localhost:8000";
const MAX_CAPTION_LENGTH = 100; // Shorter limit for cleaner look

export const ImageDetail = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const location = useLocation();

    // Check if this is an overlay
    const isModal = !!location.state?.background;

    const [mainImageDetails, setMainImageDetails] = useState<ImageItem | null>(null);
    const [similarImages, setSimilarImages] = useState<ImageItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [showFullCaption, setShowFullCaption] = useState(false);

    // Lock scrolling when modal is open
    useEffect(() => {
        if (isModal) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
            window.scrollTo(0, 0);
        }
        return () => { document.body.style.overflow = 'unset'; };
    }, [isModal]);

    useEffect(() => {
        const fetchData = async () => {
            if (!id) return;
            setLoading(true);
            try {
                // Fetch Main Image
                const mainRes = await fetch(`${API_BASE}/image_details/${id}`);
                if (mainRes.ok) setMainImageDetails(await mainRes.json());

                // Fetch Suggestions
                const similarRes = await fetch(`${API_BASE}/similar/${id}?limit=20`);
                if (similarRes.ok) setSimilarImages(await similarRes.json());
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [id]);

    if (loading) return null;
    if (!mainImageDetails) return null;

    // Calculate orientation
    let mainMeta: any = mainImageDetails.meta_data;
    if (typeof mainMeta === 'string') {
        try { mainMeta = JSON.parse(mainMeta); } catch (e) { mainMeta = {}; }
    }
    // If height > width, it's vertical. 
    const isVertical = (mainMeta?.height && mainMeta?.width) ? mainMeta.height > mainMeta.width : false;

    const filteredSimilarImages = similarImages.filter(img => img.id !== mainImageDetails.id);

    // Caption Logic
    const captionText = mainImageDetails.caption || "";
    const isLongCaption = captionText.length > MAX_CAPTION_LENGTH;
    const displayedCaption = !showFullCaption && isLongCaption
        ? captionText.slice(0, MAX_CAPTION_LENGTH).trim() + "... "
        : captionText;

    return (
        <div
            className="fixed inset-0 z-50 flex justify-center bg-black/90 backdrop-blur-sm overflow-y-auto pt-10 pb-10 px-4 animate-in fade-in duration-200"
            onClick={() => isModal && navigate(-1)}
        >
            {/* Close Button */}
            <button
                onClick={(e) => { e.stopPropagation(); navigate(-1); }}
                className="fixed top-6 right-8 z-[60] text-white/80 hover:text-white hover:bg-white/10 p-2 rounded-full transition"
            >
                <X size={32} />
            </button>

            {/* Card Container */}
            <div
                className="relative w-full max-w-[1000px] bg-[#111] rounded-[32px] shadow-2xl flex flex-col overflow-hidden min-h-[600px] h-fit"
                onClick={(e) => e.stopPropagation()}
            >

                <div className="flex flex-col md:flex-row">

                    {/* LEFT: Image Area */}
                    {/* Dynamic width: If vertical, make image area smaller (50%), if horizontal, make it wider (65%) */}
                    <div className={`flex-1 flex justify-center bg-black p-4 
                             ${isVertical ? 'md:w-1/2' : 'md:w-[65%]'}`}>
                        <img
                            src={`${API_BASE}/image/${id}`}
                            className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-md"
                            alt="Memory"
                        />
                    </div>

                    {/* RIGHT: Details Sidebar */}
                    <div className={`w-full bg-[#111] p-6 flex flex-col border-l border-white/5 
                             ${isVertical ? 'md:w-1/2' : 'md:w-[35%]'}`}>

                        {/* Actions */}
                        <div className="flex justify-between items-center mb-6">
                            <div className="flex gap-1 text-white">
                                <button onClick={() => navigate(-1)} className="p-2 hover:bg-white/10 rounded-full"><ArrowLeft size={20} /></button>
                                <button className="p-2 hover:bg-white/10 rounded-full"><MoreHorizontal size={20} /></button>
                                <button className="p-2 hover:bg-white/10 rounded-full"><Send size={20} /></button>
                            </div>
                            <button className="bg-red-600 hover:bg-red-700 text-white px-5 py-3 rounded-full font-bold text-sm transition">
                                Save
                            </button>
                        </div>

                        {/* Scrollable Content in Sidebar */}
                        <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">

                            {/* Caption Section */}
                            <div className="mb-6">
                                <h1 className="text-sm text-white font-medium leading-relaxed">
                                    {displayedCaption}
                                    {isLongCaption && (
                                        <button
                                            onClick={() => setShowFullCaption(!showFullCaption)}
                                            className="text-gray-400 hover:text-white font-bold ml-1"
                                        >
                                            {showFullCaption ? "less" : "more"}
                                        </button>
                                    )}
                                </h1>
                            </div>

                            <div className="h-[1px] bg-white/10 w-full mb-6" />

                            {/* Comments */}
                            <div className="mb-6">
                                <h3 className="text-sm font-bold text-white mb-2">Comments</h3>
                                <p className="text-gray-500 text-sm italic">No comments yet. Add one to start the conversation.</p>
                            </div>
                        </div>

                        {/* Bottom Sticky Input */}
                        <div className="pt-4 border-t border-white/10 bg-[#111]">
                            <div className="flex gap-3 items-center">
                                <div className="w-8 h-8 bg-gray-600 rounded-full flex-shrink-0"></div>
                                <input
                                    type="text"
                                    placeholder="Add a comment"
                                    className="w-full bg-[#222] text-white rounded-full py-3 px-4 text-sm focus:outline-none focus:bg-[#333] transition"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* BOTTOM SECTION: More Like This Grid */}
                <div className="p-6 md:p-8 bg-[#111] border-t border-white/5">
                    <h3 className="text-lg font-bold text-white mb-4 text-center">More like this</h3>
                    <div className="columns-2 sm:columns-3 md:columns-4 gap-4 space-y-4">
                        {filteredSimilarImages.map((img) => (
                            <Pin key={img.id} data={img} />
                            // REMOVED: onClick={...} 
                            // Pin component already handles the modal navigation internally!
                        ))}
                    </div>
                </div>

            </div>
        </div>
    );
};