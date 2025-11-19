import { Routes, Route, useLocation } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Home } from './pages/Home';
import { ImageDetail } from './pages/ImageDetail';

function App() {
  const location = useLocation();

  // Check if we have a background location state
  // This exists if we clicked a Link with state={{ background: location }}
  const state = location.state as { background?: Location };
  const background = state?.background;

  return (
    <>
      {/* Main Routes:
        If 'background' exists, we force this Switch to render the 'background' (Home),
        even though the URL says '/image/123'.
      */}
      <Routes location={background || location}>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          {/* We still keep this route here so that if someone refreshes 
             the page on /image/123, it renders normally (without modal).
          */}
          <Route path="image/:id" element={<ImageDetail />} />
        </Route>
      </Routes>

      {/* Modal Routes:
        If 'background' exists, we ALSO render this. 
        This puts the ImageDetail ON TOP of the Home page.
      */}
      {background && (
        <Routes>
          <Route path="/image/:id" element={<ImageDetail />} />
        </Routes>
      )}
    </>
  );
}

export default App;