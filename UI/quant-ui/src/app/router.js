export function createRouter({ outlet, navItems, routes, defaultRoute }) {
  let disposeCurrent = null;
  let navigationVersion = 0;

  function parseLocation() {
    const raw = window.location.hash.startsWith("#/")
      ? window.location.hash.slice(2)
      : "";
    const [routeName, queryString = ""] = raw.split("?", 2);
    const route = routes[routeName] ? routeName : defaultRoute;
    return { route, params: new URLSearchParams(queryString) };
  }

  function navigate(route, params = {}) {
    const search = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined && value !== "") {
        search.set(key, String(value));
      }
    });
    const suffix = search.size ? `?${search.toString()}` : "";
    window.location.hash = `#/${route}${suffix}`;
  }

  async function render() {
    const location = parseLocation();
    const expectedHash = `#/${location.route}${location.params.size ? `?${location.params}` : ""}`;
    if (!window.location.hash.startsWith(`#/${location.route}`)) {
      window.history.replaceState(null, "", expectedHash);
    }

    navigationVersion += 1;
    const version = navigationVersion;
    if (disposeCurrent) {
      disposeCurrent();
      disposeCurrent = null;
    }

    navItems.forEach((element, name) => {
      element.classList.toggle("active", name === location.route);
      if (name === location.route) element.setAttribute("aria-current", "page");
      else element.removeAttribute("aria-current");
    });

    outlet.innerHTML = '<div class="route-loading"><span class="spinner"></span>Loading module...</div>';
    try {
      const dispose = await routes[location.route](outlet, {
        params: location.params,
        navigate,
      });
      if (version !== navigationVersion) {
        if (typeof dispose === "function") dispose();
        return;
      }
      disposeCurrent = typeof dispose === "function" ? dispose : null;
    } catch (error) {
      if (version !== navigationVersion) return;
      console.error(error);
      outlet.innerHTML = `
        <div class="page-state error-state">
          <strong>Unable to open this module</strong>
          <span>${escapeHtml(error.message || String(error))}</span>
        </div>
      `;
    }
  }

  function start() {
    if (!window.location.hash || window.location.hash === "#/") {
      window.history.replaceState(null, "", `#/${defaultRoute}`);
    }
    window.addEventListener("hashchange", render);
    render();
  }

  return { start, navigate };
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}
