"use client";

interface Prediction {
  class: string;
  confidence: number;
  top5: { class: string; confidence: number }[];
}

interface PredictionResultProps {
  prediction: Prediction;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 80) return "bg-green-500";
  if (confidence >= 50) return "bg-yellow-500";
  return "bg-red-500";
}

function getConfidenceText(confidence: number): string {
  if (confidence >= 80) return "High confidence";
  if (confidence >= 50) return "Medium confidence";
  return "Low confidence";
}

export default function PredictionResult({ prediction }: PredictionResultProps) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
        Prediction Result
      </h2>

      <div className="text-center mb-8">
        <div className="inline-block">
          <span className="text-5xl font-bold text-primary-600 capitalize">
            {prediction.class}
          </span>
          <div className="mt-2">
            <span
              className={`inline-block px-3 py-1 rounded-full text-sm font-medium text-white ${getConfidenceColor(prediction.confidence)}`}
            >
              {prediction.confidence.toFixed(1)}% -{" "}
              {getConfidenceText(prediction.confidence)}
            </span>
          </div>
        </div>
      </div>

      <div className="border-t pt-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Top 5 Predictions
        </h3>

        <div className="space-y-3">
          {prediction.top5.map((item, index) => (
            <div key={item.class} className="flex items-center">
              <span className="w-8 text-sm font-medium text-gray-500">
                {index + 1}.
              </span>
              <span className="w-32 text-sm font-medium text-gray-700 capitalize">
                {item.class}
              </span>
              <div className="flex-1 mx-3">
                <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getConfidenceColor(item.confidence)} transition-all duration-500`}
                    style={{ width: `${item.confidence}%` }}
                  />
                </div>
              </div>
              <span className="w-16 text-sm text-gray-600 text-right">
                {item.confidence.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
