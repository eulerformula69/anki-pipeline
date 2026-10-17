document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-async-page] form");
  if (form?.matches("[data-connect-settings]")) return;
  if (form?.matches("[data-export-form]") && document.querySelector('[data-unsaved="1"]')) {
    event.preventDefault();
    alert("Сначала сохраните изменённые карточки, затем повторите экспорт.");
    return;
  }
  if (!form) return;
  event.preventDefault();
  if (form.dataset.submitting === "1") return;

  const submitter = event.submitter;
  const cardId = form.closest(".review-card")?.id;
  const data = new FormData(form);
  if (submitter?.name) data.append(submitter.name, submitter.value);
  form.dataset.submitting = "1";
  form.setAttribute("aria-busy", "true");
  form.classList.toggle("uploading", form.dataset.audioUpload === "1");
  document.body.classList.add("page-busy");

  const originalLabel = submitter?.textContent;
  if (submitter?.dataset.busyLabel) {
    submitter.textContent = submitter.dataset.busyLabel;
  }

  try {
    const response = await fetch(form.getAttribute("action"), {
      method: form.method,
      body: data,
    });
    const nextDocument = new DOMParser().parseFromString(await response.text(), "text/html");
    const nextMain = nextDocument.querySelector("main");
    if (!nextMain) throw new Error("Response does not contain the page content");
    document.querySelector("main").replaceWith(nextMain);
    document.title = nextDocument.title;
    history.replaceState(null, "", response.url);
    if (cardId) document.getElementById(cardId)?.scrollIntoView({ block: "nearest" });
  } catch (error) {
    if (submitter?.dataset.busyLabel) submitter.textContent = originalLabel;
    form.dataset.submitting = "0";
    form.removeAttribute("aria-busy");
    alert("Не удалось получить ответ. Проверьте результат перед повторной отправкой.");
  } finally {
    document.body.classList.remove("page-busy");
  }
});
