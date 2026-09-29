const photoInput = document.querySelector(".photo-input");
const selectedFile = document.querySelector("#selected-file");
const recognitionStatus = document.querySelector("#recognition-status");
const queryPreview = document.querySelector("#query-preview");
const retryPhotoWrap = document.querySelector("#retry-photo-wrap");
const personalMatchStatus = document.querySelector("#personal-match-status");
const personalMatchScoreWrap = document.querySelector("#personal-match-score-wrap");
const personalMatchScore = document.querySelector("#personal-match-score");
const personalMatchStars = document.querySelector("#personal-match-stars");
const personalMatchReason = document.querySelector("#personal-match-reason");
const personalMatchRateCurrent = document.querySelector("#personal-match-rate-current");
const compareQueue = document.querySelector("#compare-queue");
const compareQueuedTitle = document.querySelector("#compare-queued-title");
const compareAction = document.querySelector("#compare-action");
const compareCards = document.querySelector("#compare-cards");
const wineOfficialAttributes = document.querySelector("#wine-official-attributes");
const wineOfficialStatus = document.querySelector("#wine-official-status");
const collectionNav = document.querySelector("#collection-nav");
const collectionCount = document.querySelector("#collection-count");
const collectionAction = document.querySelector("#collection-action");
const collectionDialog = document.querySelector("#collection-dialog");
const collectionStatus = document.querySelector("#collection-status");
const collectionRatingField = document.querySelector("#collection-rating-field");
const collectionRatingPicker = document.querySelector("#collection-rating-picker");
const collectionSave = document.querySelector("#collection-save");
const collectionFeedback = document.querySelector("#collection-feedback");
const collectionPageStatus = document.querySelector("#collection-page-status");
const collectionSearchWrap = document.querySelector("#collection-search-wrap");
const collectionSearch = document.querySelector("#collection-search");
const collectionSearchStatus = document.querySelector("#collection-search-status");
const personalMatchTitle = document.querySelector("#match-title");

const COLLECTION_STORAGE_KEY = "my-wine-collection-v1";
const MIN_RATED_WINES_FOR_MATCH = 5;
let currentPhotoUrl = "";
let currentPhotoFile = null;
let photoRevision = 0;
let processedPhotoRevision = 0;
let recognitionInFlight = false;
let queuedPhotoChange = false;
let uncertainRetryCount = 0;
let recognizedProduct = null;
let compareWines = [];
const officialWineDetails = new Map();
let collectionEntries = readCollectionEntries();
let collectionDraftProduct = null;
let selectedRating = 0;
let activeScreen = "start";

const retryGuidance = {
  glare: {
    title: "Уберите блик с этикетки",
    copy: "Немного измените угол съёмки или поверните бутылку, чтобы свет не отражался от этикетки.",
    steps: ["Держите этикетку прямо перед камерой.", "Переместите бутылку или источник света на несколько сантиметров."]
  },
  blur: {
    title: "Проверьте фокус на этикетке",
    copy: "Мелкие надписи на этикетке не в фокусе. Сделайте новый снимок, удерживая бутылку неподвижно, и проверьте резкость перед загрузкой.",
    steps: ["Сфокусируйтесь на названии вина.", "Загрузите исходное фото, а не сжатую копию из мессенджера."]
  },
  distance: {
    title: "Подойдите ближе к этикетке",
    copy: "Бутылка занимает мало места в кадре, поэтому детали трудно различить.",
    steps: ["Сделайте новый снимок крупнее, чтобы этикетка занимала большую часть кадра.", "Оставьте края этикетки внутри снимка."]
  },
  barcode: {
    title: "Сфотографируйте лицевую этикетку",
    copy: "На фото виден штрихкод. Для поиска нужна передняя этикетка с названием вина.",
    steps: ["Поверните бутылку лицевой стороной к камере.", "Включите в кадр название и основную часть этикетки."]
  },
  year: {
    title: "Покажите год на этикетке",
    copy: "У похожих вин могут отличаться год или серия. Снимите этикетку целиком, чтобы эта надпись тоже попала в кадр.",
    steps: ["Проверьте нижнюю часть этикетки.", "Не обрезайте края, где может быть указан год."]
  },
  resolution: {
    title: "Фото слишком маленькое",
    copy: "В этом файле мало деталей, чтобы надёжно прочитать этикетку. Выберите исходное фото в полном размере или загрузите снимок с более высоким разрешением.",
    steps: ["Не отправляйте миниатюру или изображение из предпросмотра.", "Для проверки нужно фото, у которого длинная сторона не меньше 640 пикселей."]
  },
  unreadable: {
    title: "Не удалось открыть это изображение",
    copy: "Файл повреждён или сохранён в формате, который браузер не смог прочитать. Выберите другое фото в формате JPG, PNG или WebP.",
    steps: ["Откройте фото на компьютере, чтобы проверить файл.", "Если оно открывается, сохраните копию в JPG или PNG и загрузите её."]
  },
  image_too_large: {
    title: "Файл слишком большой",
    copy: "Сохраните изображение в меньшем размере или выберите фото с камеры без максимального разрешения.",
    steps: ["Для загрузки нужен файл до 20 МБ.", "Не используйте RAW-файл или несжатый скан."]
  },
  front_label: {
    title: "Нужна лицевая этикетка",
    copy: "Не удалось надёжно прочитать название вина. Поверните бутылку лицевой стороной и снимите этикетку целиком.",
    steps: ["Включите в кадр название вина.", "Держите бутылку прямо и не обрезайте края этикетки."]
  },
  uncertain: {
    title: "Проверьте ракурс и этикетку",
    copy: "Визуальный поиск и текстовые признаки не подтвердили одну карточку. Сделайте ещё один снимок этикетки прямо и крупнее.",
    steps: ["Покажите название и всю этикетку.", "Уберите блики и убедитесь, что текст в фокусе."]
  }
};

function showState(value) {
  const [screen, reason] = value.split(":");
  const screenName = screen === "retry" ? "retry" : screen;
  activeScreen = screenName;
  if (screenName === "start") recognitionStatus.textContent = "";
  if (["retry", "success", "notfound", "compare", "collection"].includes(screenName)) recognitionStatus.textContent = "";
  document.querySelectorAll("[data-screen]").forEach((element) => {
    element.hidden = element.dataset.screen !== screenName;
  });

  if (screenName === "retry") {
    const guidance = retryGuidance[reason] ?? retryGuidance.uncertain;
    document.querySelector("#retry-advice-title").textContent = guidance.title;
    document.querySelector("#retry-advice-copy").textContent = guidance.copy;
    const steps = document.querySelector("#retry-steps");
    steps.replaceChildren(...guidance.steps.map((step) => {
      const item = document.createElement("span");
      item.className = "advice-step";
      item.textContent = step;
      return item;
    }));
    retryPhotoWrap.hidden = !currentPhotoUrl || reason === "unreadable";
  }

  if (screenName === "notfound") {
    document.querySelector("#notfound-copy").textContent =
      "После повторного фото карточку подтвердить не удалось. Возможно, этого вина нет в каталоге.";
  }

  const title = document.querySelector(`[data-screen="${screenName}"] h1`);
  title?.setAttribute("tabindex", "-1");
  title?.focus({ preventScroll: true });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (screenName === "success") schedulePersonalMatch();
}

async function inspectPhoto(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return { ok: false, reason: "unreadable" };
  }

  try {
    if (Math.max(bitmap.width, bitmap.height) < 640) {
      return { ok: false, reason: "resolution" };
    }

    const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(3, Math.round(bitmap.width * scale));
    canvas.height = Math.max(3, Math.round(bitmap.height * scale));
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return { ok: true, sharpness: null };
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

    const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
    const gray = new Float32Array(canvas.width * canvas.height);
    for (let pixel = 0, offset = 0; pixel < gray.length; pixel += 1, offset += 4) {
      gray[pixel] = 0.299 * data[offset] + 0.587 * data[offset + 1] + 0.114 * data[offset + 2];
    }

    let sum = 0;
    let sumSquares = 0;
    let samples = 0;
    for (let y = 1; y < canvas.height - 1; y += 1) {
      for (let x = 1; x < canvas.width - 1; x += 1) {
        const index = y * canvas.width + x;
        const laplacian = gray[index - 1] + gray[index + 1] + gray[index - canvas.width]
          + gray[index + canvas.width] - 4 * gray[index];
        sum += laplacian;
        sumSquares += laplacian * laplacian;
        samples += 1;
      }
    }
    const mean = sum / Math.max(samples, 1);
    const sharpness = sumSquares / Math.max(samples, 1) - mean * mean;
    if (sharpness < 12) return { ok: false, reason: "blur", sharpness };
    return { ok: true, sharpness };
  } catch {
    return { ok: false, reason: "unreadable" };
  } finally {
    bitmap.close();
  }
}

function renderWineProduct(product) {
  recognizedProduct = product;
  updateCompareAction(product);
  updateCollectionAction(product);
  const image = document.querySelector("#wine-image");
  image.src = `../${product.reference_image_path}`;
  image.alt = `Этикетка вина «${product.title}»`;
  document.querySelector("#wine-category").textContent = [product.category, product.color].filter(Boolean).join(" · ");
  document.querySelector("#wine-title").textContent = product.title || "Вино из каталога";
  document.querySelector("#wine-maker").textContent = product.winery || "";
  document.querySelector("#wine-region").textContent = product.region || "—";
  document.querySelector("#wine-grape").textContent = product.grape || "—";
  const link = document.querySelector("#wine-link");
  link.href = `https://vino-svoe.ru/wines/${encodeURIComponent(product.slug)}`;
  renderOfficialWineDetails(product.slug);
}

const officialDisplayFields = [
  ["alcohol", "Алкоголь"],
  ["serving_temperature", "Температура подачи"],
  ["pairings", "Сочетается с"],
  ["rating", "Оценка на сайте"],
  ["aroma", "Аромат"],
  ["taste", "Вкус"],
  ["description", "Описание"],
];

function ensureOfficialWineDetails(slug) {
  const cached = officialWineDetails.get(slug);
  if (cached && cached.state !== "unavailable") return cached;
  if (cached?.state === "unavailable" && Date.now() - (cached.failedAt || 0) < 60000) return cached;

  const entry = { state: "loading", available: false, fields: {}, promise: null };
  officialWineDetails.set(slug, entry);
  entry.promise = fetch(`/api/wine-details?slug=${encodeURIComponent(slug)}`, { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error("official_details_unavailable");
      return response.json();
    })
    .then((result) => {
      entry.available = result.available === true;
      entry.fields = entry.available && result.fields && typeof result.fields === "object" ? result.fields : {};
      entry.sourceUrl = typeof result.source_url === "string" ? result.source_url : "";
      entry.state = entry.available ? "loaded" : "unavailable";
      if (!entry.available) entry.failedAt = Date.now();
      return entry;
    })
    .catch(() => {
      entry.state = "unavailable";
      entry.available = false;
      entry.fields = {};
      entry.failedAt = Date.now();
      return entry;
    });
  return entry;
}

function setOfficialDetailRows(container, rows) {
  container.replaceChildren(...rows.map(([label, value]) => compareAttribute(label, value)));
}

function renderOfficialWineDetails(slug) {
  const entry = ensureOfficialWineDetails(slug);
  const fields = entry.fields;
  if (recognizedProduct?.slug === slug && entry.state === "loaded") {
    if (fields.region) document.querySelector("#wine-region").textContent = fields.region;
    if (fields.grape) document.querySelector("#wine-grape").textContent = fields.grape;
    if (fields.category || fields.color) {
      document.querySelector("#wine-category").textContent = [fields.category, fields.color].filter(Boolean).join(" · ");
    }
  }

  if (entry.state === "loading") {
    wineOfficialAttributes.replaceChildren();
    wineOfficialStatus.textContent = "Загружаем характеристики с сайта…";
  } else if (!entry.available) {
    wineOfficialAttributes.replaceChildren();
    wineOfficialStatus.textContent = "Сайт сейчас не ответил. Можно открыть официальную карточку по ссылке ниже.";
  } else {
    setOfficialDetailRows(wineOfficialAttributes, officialDisplayFields
      .filter(([key]) => fields[key])
      .map(([key, label]) => [label, fields[key]]));
    wineOfficialStatus.textContent = wineOfficialAttributes.childElementCount
      ? "Данные загружены с официальной карточки вина."
      : "Карточка сайта открылась, но дополнительные характеристики не удалось прочитать.";
  }

  entry.promise.then(() => {
    if (recognizedProduct?.slug === slug) renderOfficialWineDetailsLoaded(slug, entry);
    if (activeScreen === "compare" && compareWines.some((wine) => wine.slug === slug)) renderComparison();
  });
}

function renderOfficialWineDetailsLoaded(slug, entry) {
  const fields = entry.fields;
  if (recognizedProduct?.slug === slug) {
    if (fields.region) document.querySelector("#wine-region").textContent = fields.region;
    if (fields.grape) document.querySelector("#wine-grape").textContent = fields.grape;
    if (fields.category || fields.color) {
      document.querySelector("#wine-category").textContent = [fields.category, fields.color].filter(Boolean).join(" · ");
    }
    if (!entry.available) {
      wineOfficialAttributes.replaceChildren();
      wineOfficialStatus.textContent = "Сайт сейчас не ответил. Можно открыть официальную карточку по ссылке ниже.";
    } else {
      setOfficialDetailRows(wineOfficialAttributes, officialDisplayFields
        .filter(([key]) => fields[key])
        .map(([key, label]) => [label, fields[key]]));
      wineOfficialStatus.textContent = wineOfficialAttributes.childElementCount
        ? "Данные загружены с официальной карточки вина."
        : "Карточка сайта открылась, но дополнительные характеристики не удалось прочитать.";
    }
  }
}

function updateCompareAction(product) {
  const alreadyAdded = compareWines.some((wine) => wine.slug === product.slug);
  compareAction.disabled = alreadyAdded;
  if (alreadyAdded) {
    compareAction.textContent = "Уже добавлено";
  } else if (compareWines.length === 1) {
    compareAction.textContent = "Добавить и сравнить";
  } else {
    compareAction.textContent = "Добавить к сравнению";
  }
}

function renderCompareQueue() {
  const firstWine = compareWines[0];
  compareQueue.hidden = !firstWine;
  compareQueuedTitle.textContent = firstWine?.title || firstWine?.slug || "";
}

function clearCurrentPhoto() {
  if (currentPhotoUrl) URL.revokeObjectURL(currentPhotoUrl);
  currentPhotoUrl = "";
  currentPhotoFile = null;
  photoRevision += 1;
  processedPhotoRevision = photoRevision;
  queryPreview.removeAttribute("src");
  retryPhotoWrap.hidden = true;
  selectedFile.textContent = "";
  photoInput.value = "";
}

function compareAttribute(label, value) {
  const row = document.createElement("div");
  row.className = "compare-attribute";
  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value || "—";
  row.append(name, content);
  return row;
}

function renderCompareCard(product) {
  const card = document.createElement("article");
  card.className = "compare-card";
  const imageWrap = document.createElement("div");
  imageWrap.className = "compare-card-image";
  const image = document.createElement("img");
  image.src = `../${product.reference_image_path}`;
  image.alt = `Этикетка вина «${product.title}»`;
  imageWrap.append(image);

  const content = document.createElement("div");
  content.className = "compare-card-content";
  const entry = officialWineDetails.get(product.slug);
  const fields = entry?.fields || {};
  const type = document.createElement("p");
  type.className = "compare-card-type";
  type.textContent = [fields.category || product.category, fields.color || product.color].filter(Boolean).join(" · ");
  const title = document.createElement("h2");
  title.textContent = product.title || "Вино из каталога";
  const winery = document.createElement("p");
  winery.className = "compare-card-winery";
  winery.textContent = product.winery || "";
  const waiting = entry?.state === "loading";
  const status = document.createElement("p");
  status.className = "official-data-status";
  status.textContent = waiting
    ? "Загружаем данные с сайта «Своё Вино»…"
    : entry?.available
      ? "Характеристики с официальной карточки"
      : "Официальные характеристики сейчас недоступны";
  const link = document.createElement("a");
  link.className = "text-link";
  link.href = `https://vino-svoe.ru/wines/${encodeURIComponent(product.slug)}`;
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "Открыть карточку ↗";
  content.append(type, title, winery, status, link);
  card.append(imageWrap, content);
  return card;
}

function renderCompareMatrix() {
  const matrix = document.createElement("div");
  matrix.className = "compare-matrix";
  const [left, right] = compareWines;
  const leftEntry = officialWineDetails.get(left.slug);
  const rightEntry = officialWineDetails.get(right.slug);
  const rows = [
    ["Тип", leftEntry?.fields?.category || left.category, rightEntry?.fields?.category || right.category],
    ["Цвет", leftEntry?.fields?.color || left.color, rightEntry?.fields?.color || right.color],
    ["Регион", leftEntry?.fields?.region || left.region, rightEntry?.fields?.region || right.region],
    ["Сорт", leftEntry?.fields?.grape || left.grape, rightEntry?.fields?.grape || right.grape],
  ];
  const extraFields = officialDisplayFields.filter(([key]) => key !== "description"
    || leftEntry?.fields?.description || rightEntry?.fields?.description);
  extraFields.forEach(([key, label]) => {
    const valueFor = (entry) => entry?.fields?.[key]
      || (entry?.state === "loading" ? "Загружаем…" : "—");
    rows.push([label, valueFor(leftEntry), valueFor(rightEntry)]);
  });

  rows.forEach(([label, leftValue, rightValue]) => {
    const row = document.createElement("div");
    row.className = "compare-matrix-row";
    const name = document.createElement("span");
    name.className = "compare-matrix-label";
    name.textContent = label;
    const leftCell = document.createElement("strong");
    leftCell.textContent = leftValue || "—";
    const rightCell = document.createElement("strong");
    rightCell.textContent = rightValue || "—";
    row.append(name, leftCell, rightCell);
    matrix.append(row);
  });
  return matrix;
}

function renderComparison() {
  const entries = compareWines.map((wine) => ensureOfficialWineDetails(wine.slug));
  const cards = compareWines.map(renderCompareCard);
  if (compareWines.length === 2) cards.push(renderCompareMatrix());
  compareCards.replaceChildren(...cards);
  const pending = entries.filter((entry) => entry.state === "loading");
  if (pending.length) {
    Promise.all(pending.map((entry) => entry.promise)).then(() => {
      if (activeScreen === "compare") renderComparison();
    });
  }
}

function readCollectionEntries() {
  try {
    const stored = JSON.parse(localStorage.getItem(COLLECTION_STORAGE_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    return stored.filter((entry) => entry?.product?.slug).map((entry) => ({
      product: entry.product,
      status: entry.status === "tried" ? "tried" : "want_to_try",
      rating: Number.isInteger(entry.rating) && entry.rating >= 1 && entry.rating <= 5 ? entry.rating : null
    }));
  } catch {
    return [];
  }
}

function persistCollection(entries) {
  try {
    localStorage.setItem(COLLECTION_STORAGE_KEY, JSON.stringify(entries));
    return true;
  } catch {
    return false;
  }
}

function findCollectionEntry(slug) {
  return collectionEntries.find((entry) => entry.product.slug === slug);
}

function updateCollectionAction(product) {
  collectionAction.textContent = findCollectionEntry(product.slug)
    ? "Изменить запись"
    : "Добавить в коллекцию";
}

function renderCollection() {
  const collectionEmpty = collectionEntries.length === 0;
  if (collectionEmpty) collectionSearch.value = "";
  const query = normalizeCollectionSearch(collectionSearch.value);
  const visibleEntries = collectionEntries.filter((entry) => matchesCollectionSearch(entry, query));
  const wantEntries = visibleEntries.filter((entry) => entry.status === "want_to_try");
  const triedEntries = visibleEntries.filter((entry) => entry.status === "tried");
  const wantList = document.querySelector("#collection-want-list");
  const triedList = document.querySelector("#collection-tried-list");

  collectionSearchWrap.hidden = collectionEmpty;
  collectionCount.textContent = String(collectionEntries.length);
  document.querySelector("#collection-empty").hidden = !collectionEmpty;
  document.querySelector("#collection-board").hidden = collectionEmpty;
  document.querySelector("#collection-want-count").textContent = String(wantEntries.length);
  document.querySelector("#collection-tried-count").textContent = String(triedEntries.length);
  const wantEmpty = document.querySelector("#collection-want-empty");
  const triedEmpty = document.querySelector("#collection-tried-empty");
  wantEmpty.textContent = query ? "Совпадений нет." : "Пока нет вин в этом списке.";
  triedEmpty.textContent = query ? "Совпадений нет." : "Пока нет вин в этом списке.";
  wantEmpty.hidden = wantEntries.length > 0;
  triedEmpty.hidden = triedEntries.length > 0;
  collectionSearchStatus.textContent = query
    ? visibleEntries.length ? `Найдено: ${visibleEntries.length}` : "По запросу ничего не найдено."
    : "";
  wantList.replaceChildren(...wantEntries.map(renderCollectionCard));
  triedList.replaceChildren(...triedEntries.map(renderCollectionCard));
  collectionPageStatus.textContent = "";
}

function normalizeCollectionSearch(value) {
  return value.normalize("NFKC").trim().toLocaleLowerCase("ru-RU");
}

function matchesCollectionSearch(entry, query) {
  if (!query) return true;
  const fields = ["title", "winery", "category", "color", "region", "grape", "country", "vintage", "year", "slug"];
  const searchableText = fields
    .map((field) => entry.product[field])
    .filter((value) => typeof value === "string" || typeof value === "number")
    .join(" ");
  return normalizeCollectionSearch(searchableText).includes(query);
}

function renderCollectionCard(entry) {
  const { product } = entry;
  const card = document.createElement("article");
  card.className = "collection-card";

  const image = document.createElement("img");
  image.className = "collection-card-image";
  image.src = `../${product.reference_image_path}`;
  image.alt = `Этикетка вина «${product.title}»`;

  const body = document.createElement("div");
  body.className = "collection-card-body";
  const type = document.createElement("p");
  type.className = "collection-card-type";
  type.textContent = [product.category, product.color].filter(Boolean).join(" · ");
  const title = document.createElement("h3");
  title.textContent = product.title || "Вино из каталога";
  const winery = document.createElement("p");
  winery.className = "collection-card-winery";
  winery.textContent = product.winery || "";
  const detail = document.createElement("p");
  if (entry.status === "tried") {
    const rating = Number.isInteger(entry.rating) ? entry.rating : 0;
    detail.className = "collection-card-rating";
    detail.setAttribute("role", "img");
    detail.setAttribute("aria-label", rating ? `Ваша оценка: ${rating} из 5` : "Оценка не указана");
    detail.append(...Array.from({ length: 5 }, (_, index) => {
      const star = document.createElement("span");
      star.className = `collection-card-rating-star${index < rating ? " selected" : ""}`;
      star.textContent = "★";
      return star;
    }));
  } else {
    detail.className = "collection-card-detail";
    detail.textContent = [product.region, product.grape].filter(Boolean).join(" · ");
  }

  const actions = document.createElement("div");
  actions.className = "collection-card-actions";
  const primaryAction = document.createElement("button");
  primaryAction.type = "button";
  primaryAction.className = "collection-card-action-primary";
  primaryAction.textContent = entry.status === "tried" ? "Изменить" : "Оценить";
  primaryAction.addEventListener("click", () => openCollectionDialog(
    product,
    entry.status === "tried" ? null : "tried"
  ));
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "collection-card-action-secondary";
  remove.textContent = "Убрать";
  remove.addEventListener("click", () => removeCollectionEntry(product.slug));
  actions.append(primaryAction, remove);

  body.append(type, title, winery, detail, actions);
  card.append(image, body);
  return card;
}

function removeCollectionEntry(slug) {
  const nextEntries = collectionEntries.filter((entry) => entry.product.slug !== slug);
  if (!persistCollection(nextEntries)) {
    collectionPageStatus.textContent = "Не удалось сохранить изменения в браузере.";
    return;
  }
  collectionEntries = nextEntries;
  renderCollection();
  if (recognizedProduct?.slug === slug) updateCollectionAction(recognizedProduct);
}

function syncCollectionDialog() {
  const isTried = collectionStatus.value === "tried";
  collectionRatingField.hidden = !isTried;
  collectionSave.disabled = isTried && selectedRating === 0;
  document.querySelector("#collection-rating-hint").textContent = selectedRating
    ? `Выбрано: ${selectedRating} из 5.`
    : "Выберите оценку от 1 до 5.";
  collectionRatingPicker.querySelectorAll("[data-rating]").forEach((button) => {
    const rating = Number(button.dataset.rating);
    button.setAttribute("aria-pressed", String(rating === selectedRating));
    button.classList.toggle("selected", rating <= selectedRating);
  });
}

function openCollectionDialog(product, presetStatus = null) {
  collectionDraftProduct = product;
  const existing = findCollectionEntry(product.slug);
  document.querySelector("#collection-dialog-title").textContent = !existing
    ? "Добавить в коллекцию"
    : presetStatus === "tried" && existing.status !== "tried"
      ? "Оценить вино"
      : "Изменить запись";
  document.querySelector("#collection-dialog-wine").textContent = product.title || product.slug;
  collectionStatus.value = presetStatus || existing?.status || "want_to_try";
  selectedRating = existing?.rating || 0;
  collectionFeedback.textContent = "";
  syncCollectionDialog();
  collectionDialog.showModal();
}

function saveCollectionEntry() {
  if (!collectionDraftProduct) return;
  const status = collectionStatus.value;
  if (status === "tried" && selectedRating === 0) {
    collectionFeedback.textContent = "Поставьте оценку от 1 до 5.";
    return;
  }

  const product = Object.fromEntries([
    "slug", "title", "category", "color", "region", "grape", "winery", "reference_image_path"
  ].map((key) => [key, collectionDraftProduct[key] || ""]));
  const entry = { product, status, rating: status === "tried" ? selectedRating : null };
  const existingIndex = collectionEntries.findIndex((item) => item.product.slug === product.slug);
  const nextEntries = [...collectionEntries];
  if (existingIndex >= 0) nextEntries[existingIndex] = entry;
  else nextEntries.unshift(entry);

  if (!persistCollection(nextEntries)) {
    collectionFeedback.textContent = "Не удалось сохранить. Проверьте хранилище браузера.";
    return;
  }

  collectionEntries = nextEntries;
  renderCollection();
  if (recognizedProduct?.slug === product.slug) updateCollectionAction(recognizedProduct);
  collectionDialog.close();
  if (activeScreen === "success") schedulePersonalMatch();
}

async function runRecognition() {
  if (recognitionInFlight) {
    queuedPhotoChange = true;
    selectedFile.textContent = "Новое фото загружено. Подготовим его для поиска…";
    recognitionStatus.textContent = "Запустим поиск по новому фото сразу после текущей обработки.";
    return;
  }
  const file = currentPhotoFile;
  const submittedRevision = photoRevision;
  if (!file || !currentPhotoUrl) return;

  recognitionInFlight = true;
  queuedPhotoChange = false;
  const isCurrentPhoto = () => photoRevision === submittedRevision;
  selectedFile.textContent = "Проверяем качество фото…";
  recognitionStatus.textContent = "Проверяем фото этикетки…";
  try {
    const assessment = await inspectPhoto(file);
    if (!isCurrentPhoto()) return;
    if (!assessment.ok) {
      processedPhotoRevision = submittedRevision;
      showState(`retry:${assessment.reason}`);
      return;
    }

    let response;
    let result;
    let retryAttempt = 0;
    do {
      selectedFile.textContent = retryAttempt
        ? `Распознаватель занят, повторяем поиск (${retryAttempt}/3)…`
        : "Ищем совпадение в каталоге…";
      recognitionStatus.textContent = "Сравниваем фото с каталогом вин…";
      response = await fetch("/api/recognize", {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
        signal: AbortSignal.timeout(12000)
      });
      if (!isCurrentPhoto()) return;
      result = await response.json();
      if (!isCurrentPhoto()) return;
      if (response.status !== 503 || result.error !== "recognition_busy" || retryAttempt >= 3) break;
      retryAttempt += 1;
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      if (!isCurrentPhoto()) return;
    } while (true);

    selectedFile.textContent = "Сравниваем фото с каталогом вин…";
    if (!response.ok) {
      if (result.reason === "image_too_large") {
        processedPhotoRevision = submittedRevision;
        showState("retry:image_too_large");
      } else if (response.status === 503 && result.error === "recognition_busy") {
        processedPhotoRevision = submittedRevision;
        selectedFile.textContent = "Распознаватель занят. Повторите загрузку фото через несколько секунд.";
        recognitionStatus.textContent = "Не удалось запустить поиск: сервис занят.";
      } else {
        throw new Error(result.error || "Ошибка локального сервиса распознавания.");
      }
      return;
    }
    processedPhotoRevision = submittedRevision;

    if (result.status === "found" && result.product?.slug) {
      uncertainRetryCount = 0;
      renderWineProduct(result.product);
      selectedFile.textContent = "Карточка подтверждена визуальным поиском и дополнительными признаками.";
      recognitionStatus.textContent = "";
      showState("success");
      return;
    }

    if (result.status === "retry") {
      const reason = result.reason || "uncertain";
      if (reason === "uncertain") {
        uncertainRetryCount += 1;
        if (uncertainRetryCount >= 2) {
          showState("notfound");
          return;
        }
      }
      selectedFile.textContent = "Нужен другой снимок этикетки.";
      showState(`retry:${reason}`);
      return;
    }

    showState("notfound");
  } catch (error) {
    if (!isCurrentPhoto()) return;
    processedPhotoRevision = submittedRevision;
    if (error.name === "TimeoutError") {
      try {
        const health = await fetch("/api/health", { cache: "no-store" }).then((response) => response.json());
        if (!isCurrentPhoto()) return;
        recognitionStatus.textContent = health.inference_busy
          ? "Локальное распознавание не ответило. Перезапустите сервис и повторите попытку."
          : "Распознавание заняло слишком много времени. Подождите немного и повторите попытку.";
      } catch {
        recognitionStatus.textContent = "Распознавание заняло слишком много времени. Проверьте локальный сервис и повторите попытку.";
      }
    } else {
      recognitionStatus.textContent = "Не удалось связаться с локальным распознавателем. Проверьте, что сервис запущен, и повторите попытку.";
    }
  } finally {
    recognitionInFlight = false;
    if (queuedPhotoChange) {
      queuedPhotoChange = false;
      if (currentPhotoFile && photoRevision !== processedPhotoRevision) void runRecognition();
    }
  }
}

const tasteFeatureDefinitions = [
  { key: "category", label: "Тип" },
  { key: "grape", label: "Сорт" },
  { key: "region", label: "Регион" }
];
const tasteNeutralRating = 3;
const tasteNeutralPriorCount = 2;

function tasteFeatureValues(product, key) {
  const rawValue = String(product?.[key] || "");
  const values = key === "grape" ? rawValue.split(",") : [rawValue];
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function tasteFeatureId(key, value) {
  return `${key}:${value.toLocaleLowerCase("ru-RU")}`;
}

function buildTasteProfile(entries) {
  const ratedEntries = entries.filter((entry) => (
    entry.status === "tried" && Number.isFinite(entry.rating) && entry.rating >= 1 && entry.rating <= 5
  ));
  const profile = new Map();

  ratedEntries.forEach((entry) => {
    tasteFeatureDefinitions.forEach(({ key }) => {
      tasteFeatureValues(entry.product, key).forEach((value) => {
        const featureId = tasteFeatureId(key, value);
        const stats = profile.get(featureId) || { totalRating: 0, count: 0 };
        stats.totalRating += entry.rating;
        stats.count += 1;
        profile.set(featureId, stats);
      });
    });
  });

  return { profile, ratedCount: ratedEntries.length };
}

function scoreProductForTaste(product, profile) {
  const featureScores = [];
  tasteFeatureDefinitions.forEach(({ key, label }) => {
    const matches = tasteFeatureValues(product, key)
      .map((value) => ({ value, stats: profile.get(tasteFeatureId(key, value)) }))
      .filter(({ stats }) => stats);
    if (!matches.length) return;

    const count = matches.reduce((sum, match) => sum + match.stats.count, 0);
    const totalRating = matches.reduce((sum, match) => sum + match.stats.totalRating, 0);
    const averageRating = totalRating / count;
    const estimatedRating = (
      totalRating + tasteNeutralRating * tasteNeutralPriorCount
    ) / (count + tasteNeutralPriorCount);
    featureScores.push({
      label,
      values: matches.map((match) => match.value),
      averageRating,
      estimatedRating,
      count
    });
  });

  if (!featureScores.length) return null;
  return {
    score: featureScores.reduce((sum, feature) => sum + feature.estimatedRating, 0) / featureScores.length,
    featureScores
  };
}

function renderPersonalStars(rating, ariaLabel) {
  const scoreText = formatTasteRating(rating);
  personalMatchScore.textContent = scoreText;
  personalMatchScoreWrap.hidden = false;
  personalMatchStars.setAttribute("aria-label", `${ariaLabel}: ${scoreText} из 5`);
  personalMatchStars.replaceChildren(...Array.from({ length: 5 }, (_, index) => {
    const star = document.createElement("span");
    star.className = `personal-match-star${index < Math.round(rating) ? " selected" : ""}`;
    star.textContent = "★";
    return star;
  }));
}

function schedulePersonalMatch() {
  personalMatchTitle.textContent = "Match for You";
  personalMatchScoreWrap.hidden = true;
  personalMatchStars.replaceChildren();
  personalMatchReason.hidden = true;
  personalMatchReason.textContent = "";
  personalMatchRateCurrent.hidden = true;

  if (!recognizedProduct) {
    personalMatchStatus.textContent = "После сканирования покажем оценку именно этого вина.";
    return;
  }

  const currentEntry = findCollectionEntry(recognizedProduct.slug);
  if (currentEntry?.status === "tried") {
    personalMatchTitle.textContent = "Ваша оценка";
    if (Number.isFinite(currentEntry.rating)) {
      renderPersonalStars(currentEntry.rating, "Ваша оценка этому вину");
      personalMatchStatus.textContent = "Вы уже пробовали это вино.";
      personalMatchRateCurrent.textContent = "Изменить оценку";
    } else {
      personalMatchStatus.textContent = "Вы уже пробовали это вино, но ещё не поставили оценку.";
      personalMatchRateCurrent.textContent = "Оценить это вино";
    }
    personalMatchRateCurrent.hidden = false;
    return;
  }

  personalMatchRateCurrent.textContent = "Оценить это вино";
  personalMatchRateCurrent.hidden = false;
  const { profile, ratedCount } = buildTasteProfile(collectionEntries);
  if (ratedCount < MIN_RATED_WINES_FOR_MATCH) {
    personalMatchStatus.textContent = `У вас пока мало оценок (${ratedCount}). Чтобы мы могли рекомендовать вам вина, оцените минимум ${MIN_RATED_WINES_FOR_MATCH} вин в коллекции.`;
    return;
  }

  const taste = scoreProductForTaste(recognizedProduct, profile);
  if (!taste) {
    personalMatchStatus.textContent = "Пока не могу оценить это вино по коллекции: нет совпадений по его типу, сорту или региону. Оцените вина с похожими характеристиками, чтобы появился Match for You.";
    return;
  }

  renderPersonalStars(taste.score, "Оценка совпадения");
  const reasons = [...taste.featureScores]
    .sort((a, b) => b.estimatedRating - a.estimatedRating || b.count - a.count)
    .slice(0, 2)
    .map((feature) => `${feature.label.toLocaleLowerCase("ru-RU")} «${feature.values.join(", ")}» — ${formatTasteRating(feature.estimatedRating)}/5 в ваших оценках`);
  personalMatchStatus.textContent = `Оценка рассчитана по ${ratedCount} винам, которые вы уже пробовали.`;
  personalMatchReason.textContent = `Почему может подойти: ${reasons.join("; ")}.`;
  personalMatchReason.hidden = false;
}

function formatTasteRating(rating) {
  return Number.isInteger(rating) ? String(rating) : rating.toFixed(1);
}

personalMatchRateCurrent.addEventListener("click", () => {
  if (recognizedProduct) openCollectionDialog(recognizedProduct, "tried");
});

document.querySelectorAll("[data-go]").forEach((button) => {
  button.addEventListener("click", () => {
    uncertainRetryCount = 0;
    if (button.dataset.go === "start" && currentPhotoUrl && photoRevision === processedPhotoRevision) {
      clearCurrentPhoto();
    }
    showState("start");
  });
});

collectionNav.addEventListener("click", () => {
  renderCollection();
  showState("collection");
});

collectionSearch.addEventListener("input", renderCollection);

collectionAction.addEventListener("click", () => {
  if (recognizedProduct) openCollectionDialog(recognizedProduct);
});

collectionStatus.addEventListener("change", syncCollectionDialog);
collectionRatingPicker.addEventListener("click", (event) => {
  const button = event.target.closest("[data-rating]");
  if (!button) return;
  selectedRating = Number(button.dataset.rating);
  collectionFeedback.textContent = "";
  syncCollectionDialog();
});
collectionSave.addEventListener("click", saveCollectionEntry);
document.querySelector("#collection-close").addEventListener("click", () => collectionDialog.close());
document.querySelector("#collection-cancel").addEventListener("click", () => collectionDialog.close());
collectionDialog.addEventListener("close", () => {
  if (!collectionDialog.open) collectionDraftProduct = null;
});

compareAction.addEventListener("click", () => {
  if (!recognizedProduct || compareWines.some((wine) => wine.slug === recognizedProduct.slug)) return;
  compareWines.push(recognizedProduct);
  renderCompareQueue();

  if (compareWines.length === 1) {
    clearCurrentPhoto();
    showState("start");
    return;
  }

  renderComparison();
  showState("compare");
});

document.querySelector("#compare-cancel").addEventListener("click", () => {
  compareWines = [];
  renderCompareQueue();
});

document.querySelector("#compare-reset").addEventListener("click", () => {
  compareWines = [];
  recognizedProduct = null;
  renderComparison();
  renderCompareQueue();
  clearCurrentPhoto();
  showState("start");
});

photoInput.addEventListener("change", (event) => {
  const input = event.currentTarget;
  const [file] = input.files ?? [];
  if (!file) return;
  const supportedType = ["image/jpeg", "image/png", "image/webp"].includes(file.type)
    || /\.(jpe?g|png|webp)$/i.test(file.name);
  if (!supportedType) {
    currentPhotoFile = null;
    photoRevision += 1;
    processedPhotoRevision = photoRevision;
    if (currentPhotoUrl) URL.revokeObjectURL(currentPhotoUrl);
    currentPhotoUrl = "";
    queryPreview.removeAttribute("src");
    retryPhotoWrap.hidden = true;
    showState("start");
    selectedFile.textContent = "Выберите файл изображения.";
    recognitionStatus.textContent = "Поддерживаются JPG, PNG и WebP.";
    input.value = "";
    return;
  }
  if (currentPhotoUrl) URL.revokeObjectURL(currentPhotoUrl);
  currentPhotoFile = file;
  photoRevision += 1;
  currentPhotoUrl = URL.createObjectURL(file);
  selectedFile.textContent = `Фото загружено: ${file.name}`;
  queryPreview.src = currentPhotoUrl;
  retryPhotoWrap.hidden = false;
  input.value = "";
  showState("start");
  void runRecognition();
});

renderCollection();
showState("start");
