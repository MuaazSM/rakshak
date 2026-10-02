import { Navigate, Route, Routes } from "react-router-dom";
import { Check } from "./screens/Check";
import { Home } from "./screens/Home";
import { Install } from "./screens/Install";
import { Settings } from "./screens/Settings";
import { Talk } from "./screens/Talk";
import { VerdictScreen } from "./screens/Verdict";
import { isIOS, isStandalone } from "./lib/install";
import { installSeen } from "./lib/storage";

/** B1 is shown once, in the browser, before the app is on the Home screen. */
function Root() {
  const showInstall = !isStandalone() && !installSeen() && !isIOS();
  return showInstall ? <Install /> : <Home />;
}

export function App() {
  return (
    <Routes>
      <Route path="/" element={<Root />} />
      <Route path="/home" element={<Home />} />
      <Route path="/check" element={<Check />} />
      <Route path="/v/:id" element={<VerdictScreen />} />
      <Route path="/talk" element={<Talk />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
