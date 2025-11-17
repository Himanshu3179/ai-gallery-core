import { useState, useEffect, useMemo } from 'react';
import { Pin } from './Pin';
import type { ImageItem } from '../types';

interface Props {
    images: ImageItem[];
}

export const MasonryGrid = ({ images }: Props) => {
    const [columns, setColumns] = useState(2);

    // 1. Responsive Column Logic
    useEffect(() => {
        const updateColumns = () => {
            const width = window.innerWidth;
            if (width >= 1536) setColumns(5); // 2xl
            else if (width >= 1280) setColumns(4); // xl
            else if (width >= 1024) setColumns(3); // lg
            else if (width >= 640) setColumns(2); // sm
            else setColumns(2); // mobile
        };

        updateColumns();
        window.addEventListener('resize', updateColumns);
        return () => window.removeEventListener('resize', updateColumns);
    }, []);

    // 2. "Shortest Column First" Distribution Algorithm
    // We use useMemo so this calculation only runs when images or column count changes
    const columnWrapper = useMemo(() => {
        // Initialize columns
        const cols: ImageItem[][] = Array.from({ length: columns }, () => []);
        const colHeights = new Array(columns).fill(0);

        images.forEach((img) => {
            // Parse metadata to get aspect ratio
            // Handle case where meta_data might be a string or object
            let meta: any = img.meta_data;
            if (typeof meta === 'string') {
                try {
                    meta = JSON.parse(meta);
                } catch (e) {
                    meta = { width: 1, height: 1 };
                }
            }

            // Calculate Aspect Ratio (Height / Width)
            // If dimensions missing, assume square (1)
            const aspectRatio = (meta?.height && meta?.width)
                ? meta.height / meta.width
                : 1;

            // FIND THE SHORTEST COLUMN
            let minHeight = colHeights[0];
            let minColIndex = 0;

            for (let i = 1; i < columns; i++) {
                if (colHeights[i] < minHeight) {
                    minHeight = colHeights[i];
                    minColIndex = i;
                }
            }

            // Add image to the shortest column
            cols[minColIndex].push(img);

            // Update that column's height
            // We add the aspect ratio because width is fixed in a grid
            colHeights[minColIndex] += aspectRatio;
        });

        return cols;
    }, [images, columns]);

    return (
        <div className="flex gap-4 w-full justify-center items-start">
            {columnWrapper.map((colImages, colIndex) => (
                <div
                    key={colIndex}
                    className="flex flex-col gap-4 flex-1 min-w-0"
                >
                    {colImages.map((img) => (
                        <Pin key={img.id} data={img} />
                    ))}
                </div>
            ))}
        </div>
    );
};