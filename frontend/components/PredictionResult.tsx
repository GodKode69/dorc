"use client";

interface Prediction {
  class: string;
  confidence: number;
  top5: { class: string; confidence: number }[];
}

interface PredictionResultProps {
  prediction: Prediction;
}

function barColor(c: number): string {
  if (c >= 80) return "var(--green)";
  if (c >= 50) return "var(--yellow)";
  return "var(--red)";
}

function textColor(c: number): string {
  if (c >= 80) return "var(--green)";
  if (c >= 50) return "var(--yellow)";
  return "var(--red)";
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  return (
    <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
      {/* Primary */}
      <div>
        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "0.6rem",
          color: "var(--text-dim)",
          textTransform: "uppercase",
          letterSpacing: "0.15em",
          marginBottom: "0.5rem",
        }}>
          Detected
        </div>
        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "clamp(1.8rem, 4vw, 3rem)",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "-0.03em",
          lineHeight: 1,
          marginBottom: "0.75rem",
        }}>
          {prediction.class.replace(/_/g, " ")}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span style={{
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: "0.85rem",
            fontWeight: 600,
            color: textColor(prediction.confidence),
          }}>
            {prediction.confidence.toFixed(1)}%
          </span>
          <span style={{
            fontFamily: '"Inter", sans-serif',
            fontSize: "0.75rem",
            color: "var(--text-muted)",
          }}>
            confidence
          </span>
        </div>
      </div>

      {/* Divider */}
      <div style={{ height: "1px", background: "var(--border)" }} />

      {/* Top 5 */}
      <div>
        <div style={{
          fontFamily: '"JetBrains Mono", monospace',
          fontSize: "0.6rem",
          color: "var(--text-dim)",
          textTransform: "uppercase",
          letterSpacing: "0.15em",
          marginBottom: "0.75rem",
        }}>
          Top 5
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {prediction.top5.map((item, i) => (
            <div key={item.class} className="fade-in" style={{
              display: "flex",
              alignItems: "center",
              gap: "0.75rem",
              animationDelay: `${i * 0.05}s`,
            }}>
              <span style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.72rem",
                color: "var(--text-dim)",
                width: "1.5rem",
                textAlign: "right",
                flexShrink: 0,
              }}>
                {i + 1}
              </span>
              <span style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.78rem",
                fontWeight: 500,
                textTransform: "capitalize",
                width: "110px",
                flexShrink: 0,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}>
                {item.class.replace(/_/g, " ")}
              </span>
              <div style={{
                flex: 1,
                height: "6px",
                background: "rgba(255,255,255,0.04)",
                borderRadius: "999px",
                overflow: "hidden",
              }}>
                <div className="bar-fill" style={{
                  height: "100%",
                  background: barColor(item.confidence),
                  borderRadius: "999px",
                  width: `${item.confidence}%`,
                  animationDelay: `${0.3 + i * 0.08}s`,
                }} />
              </div>
              <span style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "0.72rem",
                color: "var(--text-muted)",
                width: "42px",
                textAlign: "right",
                flexShrink: 0,
              }}>
                {item.confidence.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
