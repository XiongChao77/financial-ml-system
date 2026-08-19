import { apiRequest, postJson } from "../../shared/api-client.js";

export function loadLabelSchema(signal) {
  return apiRequest("/api/labels/schema", { signal });
}

export function loadCurrentLabels(signal) {
  return apiRequest("/api/labels/current", { signal });
}

export function generateLabels(config, signal) {
  return postJson("/api/labels/generate", config, { signal });
}
