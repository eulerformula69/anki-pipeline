const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");
const path = require("node:path");

const script = readFileSync(path.join(__dirname, "../src/gui/static/app.js"), "utf8");

function setup({ editor = true, dirty = true, ok = true, offline = false } = {}) {
  const handlers = {};
  const alerts = [];
  const busy = new Set();
  const page = { inert: false };
  const secondCard = { value: "Unsaved draft", dataset: { unsaved: "1" } };
  const nextCard = { querySelectorAll: () => [], scrollIntoView() {} };
  let currentCard;
  let mainReplaced = false;
  let requests = 0;
  const card = {
    id: "card_0001",
    querySelectorAll: () => [],
    replaceWith(next) { currentCard = next; },
  };
  currentCard = card;
  const count = { textContent: "2 of 2" };
  const form = {
    method: "post",
    dataset: { unsaved: "1", audioUpload: "1" },
    classList: { toggle() {}, remove() {} },
    setAttribute() {}, removeAttribute() {},
    matches: (selector) => selector === ".editor" && editor,
    closest: (selector) => selector === "[data-async-page] form" ? form
      : selector === ".review-card" ? card : page,
    getAttribute: () => "/projects/test/cards/card_0001",
  };
  const document = {
    addEventListener: (name, handler) => { handlers[name] = handler; },
    body: { classList: {
      add: (name) => busy.add(name),
      remove: (name) => busy.delete(name),
      contains: (name) => busy.has(name),
    } },
    querySelectorAll: () => [],
    querySelector: (selector) => selector === '[data-unsaved="1"]' ? dirty && secondCard
      : selector === "[data-card-count]" ? count
      : { replaceWith() { mainReplaced = true; } },
    getElementById: () => nextCard,
  };
  const nextDocument = {
    title: "Saved",
    getElementById: () => nextCard,
    querySelector: (selector) => selector === ".error p:last-of-type" ? { textContent: "Audio text changed" }
      : selector === "[data-card-count]" ? { textContent: "1 of 2" } : {},
  };
  vm.runInNewContext(script, {
    document,
    window: { addEventListener: (name, handler) => { handlers[name] = handler; } },
    history: { replaceState() {} },
    alert: (message) => alerts.push(message),
    confirm: () => true,
    FormData: class { append() {} },
    DOMParser: class { parseFromString() { return nextDocument; } },
    fetch: async () => {
      requests += 1;
      assert.equal(page.inert, true);
      if (offline) throw new TypeError("Failed to fetch");
      return { ok, text: async () => "page", url: "/projects/test" };
    },
  });
  return {
    handlers, alerts, page, form, secondCard, count, busy, nextCard,
    currentCard: () => currentCard,
    mainReplaced: () => mainReplaced,
    requests: () => requests,
    submit: () => handlers.submit({ target: form, preventDefault() {} }),
  };
}

test("saving one card retains the other draft and updates the saved count", async () => {
  const ui = setup();
  await ui.submit();
  assert.equal(ui.currentCard(), ui.nextCard);
  assert.equal(ui.mainReplaced(), false);
  assert.equal(ui.secondCard.value, "Unsaved draft");
  assert.equal(ui.secondCard.dataset.unsaved, "1");
  assert.equal(ui.count.textContent, "1 of 2");
  assert.deepEqual(ui.alerts, []);
  assert.equal(ui.page.inert, false);
});

test("actions that replace the page stop before a request when drafts exist", async () => {
  const ui = setup({ editor: false });
  await ui.submit();
  assert.equal(ui.requests(), 0);
  assert.equal(ui.mainReplaced(), false);
  assert.equal(ui.alerts.length, 1);
});

test("actions that replace the page still work after all cards are saved", async () => {
  const ui = setup({ editor: false, dirty: false });
  await ui.submit();
  assert.equal(ui.requests(), 1);
  assert.equal(ui.mainReplaced(), true);
});

for (const options of [{ ok: false }, { offline: true }]) {
  test(`failed requests retain drafts and restore controls: ${JSON.stringify(options)}`, async () => {
    const ui = setup(options);
    const original = ui.currentCard();
    await ui.submit();
    assert.equal(ui.currentCard(), original);
    assert.equal(ui.mainReplaced(), false);
    assert.equal(ui.secondCard.value, "Unsaved draft");
    assert.equal(ui.form.dataset.unsaved, "1");
    assert.equal(ui.form.dataset.submitting, "0");
    assert.equal(ui.form.dataset.audioUpload, undefined);
    assert.equal(ui.page.inert, false);
    assert.equal(ui.busy.size, 0);
    assert.equal(ui.alerts.length, 1);
    if (!options.offline) assert.equal(ui.alerts[0], "Audio text changed");
    else assert.match(ui.alerts[0], /Не удалось получить ответ/);
  });
}

test("leaving the page warns only while drafts exist", () => {
  for (const dirty of [false, true]) {
    const ui = setup({ dirty });
    let prevented = false;
    ui.handlers.beforeunload({ preventDefault() { prevented = true; } });
    assert.equal(prevented, dirty);
  }
});
