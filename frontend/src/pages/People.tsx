import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Users } from 'lucide-react';
import type { Person } from '../types';

const API_BASE = "http://localhost:8000";
const CACHE_KEY = "people_data_cache";

export const People = () => {
    // Initialize from Cache (Instant Load)
    const [people, setPeople] = useState<Person[]>(() => {
        const cached = sessionStorage.getItem(CACHE_KEY);
        return cached ? JSON.parse(cached) : [];
    });

    const [loading, setLoading] = useState(people.length === 0);

    useEffect(() => {
        if (people.length > 0) {
            setLoading(false);
            return;
        }

        fetch(`${API_BASE}/people`)
            .then(res => res.json())
            .then(data => {
                setPeople(data);
                sessionStorage.setItem(CACHE_KEY, JSON.stringify(data));
            })
            .catch(err => console.error("Failed to load people", err))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return <div className="w-full h-96 flex items-center justify-center text-gray-500">Loading identities...</div>;
    }

    return (
        <main className="p-6 max-w-[1600px] mx-auto min-h-screen">
            <h1 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
                <Users className="text-white" /> People & Faces
            </h1>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                {people.map((person) => (
                    <Link
                        to={`/people/${person.id}`}
                        key={person.id}
                        state={{ person }}
                        className="group flex flex-col gap-3"
                    >
                        <div className="aspect-square w-full overflow-hidden rounded-3xl bg-[#1e1e1e] relative border border-white/5 group-hover:border-white/20 transition-colors">
                            <img
                                src={`${API_BASE}/people/${person.id}/thumbnail`}
                                alt={person.name}
                                className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                                loading="lazy"
                            />
                        </div>

                        <div className="px-1 text-center sm:text-left">
                            <h3 className="text-white font-bold text-lg truncate">{person.name}</h3>
                            <p className="text-gray-500 text-sm">{person.face_count} photos</p>
                        </div>
                    </Link>
                ))}
            </div>
        </main>
    );
};