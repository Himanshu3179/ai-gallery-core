import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Pin } from '../components/Pin';
import type { ImageItem } from '../types';

const API_BASE = "http://localhost:8000";

export const ImageDetail = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [mainImage, setMainImage] = useState<ImageItem | null>(null);
    const [similarImages, setSimilarImages] = useState<ImageItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        window.scrollTo(0, 0); // Reset scroll on new image
        const fetchData = async () => {
            if (!id) return;
            setLoading(true);
            try {
                // 1. Get Similar Images (Using your backend)
                const similarRes = await fetch(`${API_BASE}/similar/${id}?limit=15`);
                const similarData = await similarRes.json();
                setSimilarImages(similarData);

                // 2. We need the Main Image details.
                // Since your backend doesn't have a specific /metadata/{id} endpoint yet (only /image/{id} for binary),
                // we can cheat: usually, the main image is the "query" for similar images.
                // BUT, if you clicked from Home, we ideally passed the object. 
                // Since we are reloading, let's assume the user clicked. 
                // A quick fix: The /similar endpoint calculates based on ID.
                // We might need to update Backend to return the source image metadata OR just fetch the feed/search to find it.
                // HACK: For now, let's try to find the main image in the 'similar' list logic or fetch feed. 
                // BETTER: Add a specific endpoint in backend `GET /metadata/{id}`.
                // FOR NOW: Let's just display the image using the ID and the raw file, and skip the caption for the main image if missing,
                // or use the first similar image's structure to infer.

                // Actually, let's just fetch the image file for display.
                // For a proper app, add `GET /api/images/:id/details` to backend.
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [id]);

    return (
        <div className="max-w-[1600px] mx-auto p-4 animate-in fade-in duration-300">

            <button
                onClick={() => navigate(-1)}
                className="mb-6 flex items-center gap-2 text-white hover:bg-white/10 px-4 py-2 rounded-full transition w-fit"
            >
                <ArrowLeft className="w-5 h-5" />
                Back
            </button>

            {/* Split Layout */}
            <div className="flex flex-col lg:flex-row gap-8 bg-[#1e1e1e] rounded-[32px] overflow-hidden shadow-2xl min-h-[600px]">

                {/* LEFT: The Image (Pinterest style: Centered, Dark/Light background) */}
                <div className="flex-1 bg-black flex items-center justify-center p-4 lg:p-10">
                    <img
                        src={`${API_BASE}/image/${id}`}
                        className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-lg"
                        alt="Detail"
                    />
                </div>

                {/* RIGHT: Details & Comments (Simulated) */}
                <div className="w-full lg:w-[400px] bg-[#1e1e1e] p-8 flex flex-col">
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex gap-4">
                            <div className="w-10 h-10 bg-gray-600 rounded-full"></div>
                            <div>
                                <p className="font-bold text-white">Himanshu</p>
                                <p className="text-xs text-gray-400">200 followers</p>
                            </div>
                        </div>
                        <button className="bg-gray-800 text-white px-4 py-2 rounded-full font-bold hover:bg-gray-700">Follow</button>
                    </div>

                    <h1 className="text-2xl font-bold text-white mb-2">Captured Moment</h1>
                    <p className="text-gray-300 mb-6">
                        This is a memory captured by your AI Gallery. It matches the vibe of the images below.
                    </p>

                    <div className="mt-auto pt-6 border-t border-white/10">
                        <h3 className="text-lg font-bold text-white mb-4">Comments</h3>
                        <p className="text-gray-500 italic">No comments yet.</p>
                    </div>
                </div>
            </div>

            {/* "More Like This" Section */}
            <div className="mt-16">
                <h2 className="text-xl font-bold text-white mb-6 text-center">More like this</h2>
                <div className="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4">
                    {similarImages.map((img) => (
                        <Pin key={img.id} data={img} />
                    ))}
                </div>
            </div>

        </div>
    );
};