export interface ImageMeta {
    width: number;
    height: number;
    size_bytes?: number;
}

export interface ImageItem {
    id: number;
    image_path: string; // Added this based on backend response
    caption: string;
    meta_data: ImageMeta;
}