import { apiRequest } from "../../shared/api-client.js";

export function loadLiveStrategies(options = {}) {
  return apiRequest("/api/live/strategies", options);
}

export function loadLiveStrategy(strategyId, options = {}) {
  return apiRequest(
    `/api/live/strategies/${encodeURIComponent(strategyId)}`,
    options,
  );
}

