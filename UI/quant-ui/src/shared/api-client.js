export async function apiRequest(path, options = {}) {
  if (!path.startsWith("/api/")) {
    throw new Error(`API path must be same-origin and start with /api/: ${path}`);
  }

  const headers = new Headers(options.headers || {});
  if (options.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(path, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (response.status !== 204) {
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const text = await response.text();
      payload = text ? { detail: text } : null;
    }
  }

  if (!response.ok) {
    const detail = payload?.detail || payload?.error;
    throw new Error(formatApiError(detail, response));
  }
  return payload;
}

export function postJson(path, payload, options = {}) {
  return apiRequest(path, {
    ...options,
    method: "POST",
    body: JSON.stringify(payload),
  });
}

function formatApiError(detail, response) {
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const location = Array.isArray(item.loc)
        ? item.loc.filter((part) => part !== "body").join(".")
        : "";
      return `${location ? `${location}: ` : ""}${item.msg || JSON.stringify(item)}`;
    }).join("\n");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return String(detail || `${response.status} ${response.statusText}`);
}
