let ankiDeckRequest = 0;

async function refreshAnkiDecks(data, action = "/anki-connect/decks") {
  const requestId = ++ankiDeckRequest;
  const panel = document.querySelector('[data-export-panel="connect"]');
  const status = panel.querySelector("[data-connect-status]");
  const select = panel.querySelector("[data-connect-decks]");
  const importButton = panel.querySelector("[data-connect-import]");
  select.disabled = importButton.disabled = true;
  status.textContent = "Подключаемся к Anki…";
  try {
    const response = await fetch(action, { method: "POST", body: data });
    const result = await response.json();
    if (requestId !== ankiDeckRequest) return;
    if (result.url) panel.querySelector('[name="url"]').value = result.url;
    if (!response.ok) throw new Error(result.error || "Не удалось подключиться к Anki.");
    const previous = select.value;
    select.replaceChildren(new Option("Выберите колоду", ""));
    for (const name of result.decks) select.add(new Option(name, name));
    if (result.decks.includes(previous)) select.value = previous;
    select.disabled = false;
    importButton.disabled = result.decks.length === 0
      || Number(panel.querySelector("[data-connect-summary]").dataset.pending) === 0;
    panel.querySelector("[data-connect-destination]").textContent = select.value || "не выбрана";
    status.textContent = result.decks.length
      ? "Связь с Anki установлена. Выберите колоду."
      : "В Anki нет колод. Создайте колоду в Anki и обновите список.";
  } catch (error) {
    if (requestId !== ankiDeckRequest) return;
    status.textContent = error.message;
    panel.querySelector("details").open = true;
  }
}

document.addEventListener("click", (event) => {
  const mode = event.target.closest("[data-export-mode]");
  if (mode) {
    document.querySelectorAll("[data-export-mode]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button === mode));
    });
    document.querySelectorAll("[data-export-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.exportPanel !== mode.dataset.exportMode;
    });
    if (mode.dataset.exportMode !== "connect") return;
  }
  if (mode || event.target.closest("[data-connect-refresh]")) {
    const data = new FormData();
    data.set("csrf", document.querySelector('[data-connect-settings] [name="csrf"]').value);
    refreshAnkiDecks(data);
  }
});

document.addEventListener("change", (event) => {
  if (!event.target.matches("[data-connect-decks]")) return;
  document.querySelector("[data-connect-destination]").textContent = event.target.value || "не выбрана";
});

document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-connect-settings]");
  if (!form) return;
  event.preventDefault();
  const button = form.querySelector("button");
  if (button.disabled) return;
  button.disabled = true;
  await refreshAnkiDecks(new FormData(form), form.action);
  form.querySelector('[name="key"]').value = "";
  button.disabled = false;
});

document.addEventListener("input", (event) => {
  const form = event.target.closest(".review-card form:not(.order)");
  if (form) form.dataset.unsaved = "1";
});
