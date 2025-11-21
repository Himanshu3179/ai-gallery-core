export interface ImageMeta {
    width: number;
    height: number;
    size_bytes?: number;
}

export interface ImageItem {
    id: number;
    image_path: string;
    caption: string;
    meta_data: ImageMeta;
}

export interface Person {
    id: number;
    name: string;
    face_count: number;
    // cover_image_id is removed as we now use the /thumbnail endpoint
}