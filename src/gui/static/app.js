document.querySelectorAll("input[data-auto-submit]").forEach((input) => {
  input.addEventListener("change", () => {
    if (input.files.length > 0) input.form.requestSubmit();
  });
});
