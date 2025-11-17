interface Props {
  onSearch: (q: string) => void;
}

export const SearchBar = ({ onSearch }: Props) => {
  return (
    <div className="sticky top-0 z-50 bg-black/90 backdrop-blur-md py-4 px-6 border-b border-white/10">
      <div className="max-w-3xl mx-auto">
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <input
            type="text"
            placeholder="Search your memories... (e.g., 'birthday party', 'sad rain', 'mountain trip')"
            className="w-full bg-gray-800 text-white rounded-full py-3 pl-12 pr-4 focus:outline-none focus:ring-2 focus:ring-white/20 transition-all"
            onKeyDown={(e) => e.key === 'Enter' && onSearch(e.currentTarget.value)}
          />
        </div>
      </div>
    </div>
  );
};