import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/tokens.css";
import "./styles/tracer.css";
import "./styles/setup.css";
import { TracerApp } from "./TracerApp";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <TracerApp />
  </React.StrictMode>
);
