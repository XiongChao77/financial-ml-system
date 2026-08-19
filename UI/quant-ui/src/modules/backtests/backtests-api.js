import { apiRequest } from "../../shared/api-client.js";

export function loadLatestBacktest(signal) {
  return apiRequest("/api/backtests/latest", { signal });
}

export function loadExperimentBacktest(datasetId, recordId, period, signal) {
  const dataset = encodeURIComponent(datasetId);
  const record = encodeURIComponent(recordId);
  if (period === "all") {
    return apiRequest(`/api/backtests/experiment/${dataset}/${record}/complete`, { signal });
  }
  const query = new URLSearchParams({ period });
  return apiRequest(`/api/backtests/experiment/${dataset}/${record}?${query}`, { signal });
}
