import { useEffect, useState } from "react";
import type { ReactElement } from "react";

import { DemoRecord, fetchDemos } from "./api";
import "./styles.css";

const fallbackDemos: Record<string, DemoRecord> = {
  ambiguous_interpretation: {
    passed: true,
    detail: "Local fallback data is active until the API is started."
  }
};

export function App(): ReactElement {
  const [demos, setDemos] = useState<Record<string, DemoRecord>>(fallbackDemos);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    void fetchDemos()
      .then((payload) => {
        if (!cancelled) {
          setDemos(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("API not running, showing fallback demo state.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Olympus Forge</p>
        <h1>Cognitive behaviors, traced and testable.</h1>
        <p className="summary">
          A local research dashboard for interpretive branching, verification, retrodiction,
          learned transforms, and behavior-graph execution.
        </p>
        {error ? <p className="notice">{error}</p> : null}
      </section>
      <section className="grid">
        {Object.entries(demos).map(([name, record]) => (
          <article className="card" key={name}>
            <div className="card-header">
              <h2>{name.replace(/_/g, " ")}</h2>
              <span data-status={record.passed === false ? "failed" : "passed"}>
                {record.passed === false ? "Needs work" : "Passing"}
              </span>
            </div>
            <p>{record.detail ?? record.response ?? "No detail available."}</p>
            {record.category ? <small>Category: {record.category}</small> : null}
          </article>
        ))}
      </section>
    </main>
  );
}

export default App;
