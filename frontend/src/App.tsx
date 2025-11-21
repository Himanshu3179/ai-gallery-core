import { Routes, Route, useLocation } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { ImageDetail } from './pages/ImageDetail';
import { PersonDetail } from './pages/PersonDetail';
import { People } from './pages/People';

function App() {
  const location = useLocation();
  const state = location.state as { background?: Location };
  const background = state?.background;

  return (
    <>
      <Routes location={background || location}>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="people" element={<People />} />
          <Route path="people/:id" element={<PersonDetail />} />
          <Route path="image/:id" element={<ImageDetail />} />
        </Route>
      </Routes>

      {background && (
        <Routes>
          <Route path="/image/:id" element={<ImageDetail />} />
        </Routes>
      )}
    </>
  );
}

export default App;