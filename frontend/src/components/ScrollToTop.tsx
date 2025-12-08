import { useEffect } from "react";
import { useLocation, useNavigationType } from "react-router-dom";

export const ScrollToTop = () => {
  const { pathname } = useLocation();
  const action = useNavigationType();

  useEffect(() => {
    // If it's a new navigation (PUSH), scroll to the top.
    // If it's a back button (POP), let the browser handle the restoration.
    if (action !== "POP") {
      window.scrollTo(0, 0);
    }
  }, [action, pathname]);

  return null;
};