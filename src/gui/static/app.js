document.addEventListener("submit", async (event) => {
  const form = event.target.closest("[data-async-page] form");
  if (form?.matches("[data-connect-settings]")) return;
  if (!form) return;
  event.preventDefault();
  if (document.body.classList.contains("page-busy")) return;
  const editor = form.matches(".editor");
  if (!editor && document.querySelector('[data-unsaved="1"]')) {
    alert("Сначала сохраните изменённые карточки, затем повторите действие.");
    return;
  }
  if (form.dataset.submitting === "1") return;
  if (form.dataset.confirm && !confirm(form.dataset.confirm)) return;

  const submitter = event.submitter;
  const cardId = form.closest(".review-card")?.id;
  const data = new FormData(form);
  if (submitter?.name) data.append(submitter.name, submitter.value);
  form.dataset.submitting = "1";
  form.setAttribute("aria-busy", "true");
  form.classList.toggle("uploading", form.dataset.audioUpload === "1");
  document.body.classList.add("page-busy");
  const page = form.closest("[data-async-page]");
  page.inert = true;

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
    if (!response.ok) {
      throw new Error(nextDocument.querySelector(".error p:last-of-type")?.textContent
        || "Не удалось сохранить изменения. Повторите попытку.");
    }
    const nextMain = nextDocument.querySelector("main");
    if (!nextMain) throw new Error("Не удалось получить страницу. Проверьте результат перед повторной отправкой.");
    if (editor) {
      const nextCard = nextDocument.getElementById(cardId);
      if (!nextCard) throw new Error("Не удалось обновить карточку. Проверьте результат сохранения.");
      form.closest(".review-card").querySelectorAll("details").forEach((details, index) => {
        nextCard.querySelectorAll("details")[index].open = details.open;
      });
      form.closest(".review-card").replaceWith(nextCard);
      document.querySelector("[data-card-count]").textContent = nextDocument.querySelector("[data-card-count]").textContent;
    } else {
      document.querySelector("main").replaceWith(nextMain);
    }
    fitCardFields();
    document.title = nextDocument.title;
    history.replaceState(null, "", response.url);
    if (cardId) document.getElementById(cardId)?.scrollIntoView({ block: "nearest" });
  } catch (error) {
    const fallback = "Не удалось получить ответ. Проверьте результат перед повторной отправкой.";
    alert(error.name === "TypeError" ? fallback : error.message || fallback);
  } finally {
    if (submitter?.dataset.busyLabel) submitter.textContent = originalLabel;
    form.dataset.submitting = "0";
    form.removeAttribute("aria-busy");
    form.classList.remove("uploading");
    delete form.dataset.audioUpload;
    page.inert = false;
    document.body.classList.remove("page-busy");
  }
});

function fitCardFields(root = document) {
  root.querySelectorAll('.review-card textarea').forEach((field) => {
    if (!field.getClientRects().length) return;
    field.style.height = 'auto';
    field.style.height = `${field.scrollHeight + 2}px`;
  });
}

document.addEventListener('input', (event) => {
  const editor = event.target.closest('.editor');
  if (!editor) return;
  editor.dataset.unsaved = '1';
  editor.querySelector('.save-status').textContent = 'Есть несохранённые изменения';
  if (event.target.matches('textarea')) fitCardFields(editor);
});
window.addEventListener('resize', () => fitCardFields());
fitCardFields();

window.addEventListener("beforeunload", (event) => {
  if (!document.querySelector('[data-unsaved="1"]')) return;
  event.preventDefault();
  event.returnValue = "";
});
