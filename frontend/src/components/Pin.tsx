import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom'; // Import useLocation
import type { ImageItem } from '../types';

const API_BASE = "http://localhost:8000";

interface Props {
    data: ImageItem;
}

export const Pin = ({ data }: Props) => {
    const [loaded, setLoaded] = useState(false);
    const location = useLocation(); // Get current location

    let meta: any = data.meta_data;
    if (typeof meta === 'string') {
        try { meta = JSON.parse(meta); } catch (e) { meta = { width: 1, height: 1 }; }
    }

    const aspectRatio = (meta?.height && meta?.width) ? meta.height / meta.width : 1;

    return (
        <Link
            to={`/image/${data.id}`}
            // THIS IS THE MAGIC: Pass the current location as "background"
            state={{ background: location }}
            className="block mb-6 break-inside-avoid group"
        >
            <div className="relative rounded-2xl overflow-hidden bg-[#1e1e1e]" style={{ paddingBottom: `${aspectRatio * 100}%` }}>
                <img
                    src={`${API_BASE}/image/${data.id}`}
                    alt={data.caption}
                    className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
                    onLoad={() => setLoaded(true)}
                    loading="lazy"
                />
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-end p-4">
                    <button className="bg-red-600 text-white px-4 py-2 rounded-full font-bold text-sm ml-auto hover:bg-red-700">
                        View
                    </button>
                </div>
            </div>
            <p className="mt-2 text-sm text-white/90 font-medium truncate pr-2">
                {data.caption}
            </p>
        </Link>
    );
};