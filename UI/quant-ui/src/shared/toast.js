export function createToast(host = document.body) {
  const element = document.createElement("div");
  element.className = "toast";
  host.appendChild(element);
  let timer = null;

  function show(message, error = false) {
    element.textContent = message;
    element.className = `toast visible${error ? " error" : ""}`;
    clearTimeout(timer);
    timer = setTimeout(() => {
      element.className = "toast";
    }, 3800);
  }

  return {
    show,
    destroy() {
      clearTimeout(timer);
      element.remove();
    },
  };
}
