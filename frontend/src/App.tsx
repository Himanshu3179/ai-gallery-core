import { Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { ImageDetail } from './pages/ImageDetail';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="image/:id" element={<ImageDetail />} />
      </Route>
    </Routes>
  );
}

export default App;