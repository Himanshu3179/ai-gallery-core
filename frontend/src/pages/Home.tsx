import { useEffect, useState, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { MasonryGrid } from '../components/MasonryGrid'; // Import the new component
import type { ImageItem } from '../types';

const API_BASE = "http://localhost:8000";

export const Home = () => {
    const [images, setImages] = useState<ImageItem[]>([]);
    const [loading, setLoading] = useState(false);
    const [hasMore, setHasMore] = useState(true);

    // Refs for stable state across renders
    const observerTarget = useRef(null);
    const seedRef = useRef(Math.random().toString(36).substring(7));
    const offsetRef = useRef(0);

    const [searchParams] = useSearchParams();
    const searchQuery = searchParams.get("q");

    const fetchImages = useCallback(async (isNewSearch = false) => {
        // Prevent fetching if already loading or no more data (unless it's a new search)
        if (loading || (!hasMore && !isNewSearch)) return;

        setLoading(true);

        try {
            const currentOffset = isNewSearch ? 0 : offsetRef.current;

            // Construct URL
            let url = "";
            if (searchQuery) {
                url = `${API_BASE}/search?q=${searchQuery}&limit=20`;
            } else {
                // Use stable seed to prevent backend randomization shuffling
                url = `${API_BASE}/feed?limit=20&offset=${currentOffset}&seed=${seedRef.current}`;
            }

            const res = await fetch(url);
            if (!res.ok) throw new Error("Failed to fetch");
            const data = await res.json();

            if (data.length === 0) {
                setHasMore(false);
                setLoading(false);
                return;
            }

            setImages(prev => {
                if (isNewSearch) return data;

                // ROBUST DEDUPLICATION:
                // Filter out any incoming image that is ALREADY in 'prev'
                const existingIds = new Set(prev.map(p => p.id));
                const uniqueNew = data.filter((d: ImageItem) => !existingIds.has(d.id));

                // STRICT APPEND: Old images + New images.
                // The MasonryGrid component will handle the layout, keeping order intact.
                return [...prev, ...uniqueNew];
            });

            // Update offset
            offsetRef.current = isNewSearch ? data.length : offsetRef.current + data.length;

        } catch (e) {
            console.error("Fetch error:", e);
        } finally {
            setLoading(false);
        }
    }, [loading, hasMore, searchQuery]);

    // 1. Initial Load / Search Change
    useEffect(() => {
        // Reset everything
        setImages([]);
        offsetRef.current = 0;
        setHasMore(true);
        // New seed for new search context
        if (!searchQuery) seedRef.current = Math.random().toString(36).substring(7);

        fetchImages(true);
    }, [searchQuery]);

    // 2. Infinite Scroll Observer
    useEffect(() => {
        const observer = new IntersectionObserver(
            entries => {
                if (entries[0].isIntersecting && hasMore && !loading) {
                    fetchImages();
                }
            },
            { threshold: 0.1, rootMargin: '100px' } // Trigger 100px before bottom
        );

        if (observerTarget.current) {
            observer.observe(observerTarget.current);
        }

        return () => {
            if (observerTarget.current) observer.unobserve(observerTarget.current);
        };
    }, [fetchImages, hasMore, loading]);

    return (
        <main className="p-4 max-w-[1600px] mx-auto min-h-screen">

            {/* THE FIX: Use the deterministic MasonryGrid */}
            <MasonryGrid images={images} />

            {/* Loading / Trigger Area */}
            <div ref={observerTarget} className="h-20 w-full flex justify-center items-center mt-10">
                {loading && (
                    <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                )}
                {!hasMore && images.length > 0 && (
                    <p className="text-gray-500 text-sm">End of memories.</p>
                )}
            </div>
        </main>
    );
};