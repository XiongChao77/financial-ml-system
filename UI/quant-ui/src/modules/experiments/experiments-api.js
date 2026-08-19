import { apiRequest, postJson } from "../../shared/api-client.js";

const ROOT = "/api/experiments";

export function experimentGet(path, signal) {
  return apiRequest(`${ROOT}${path}`, { signal });
}

export function experimentPost(path, payload, signal) {
  return postJson(`${ROOT}${path}`, payload, { signal });
}
