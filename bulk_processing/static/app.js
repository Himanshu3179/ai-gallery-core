const grid = document.getElementById("grid");
const emptyState = document.getElementById("empty-state");
const sentinel = document.getElementById("sentinel");
const bottomLoader = document.getElementById("bottom-loader");
const searchInput = document.getElementById("search-input");
const searchStatus = document.getElementById("search-status");

const modal = document.getElementById("modal");
const modalImage = document.getElementById("modal-image");
const modalFilename = document.getElementById("modal-filename");
const modalMeta = document.getElementById("modal-meta");
const modalTags = document.getElementById("modal-tags");
const modalClose = document.getElementById("modal-close");

const PAGE_SIZE = 40;
const DEBOUNCE_MS = 400;

let currentQuery = "";
let offset = 0;
let hasMore = true;
let isLoading = false;
let requestToken = 0; // guards against out-of-order responses
let activeController = null;

function buildUrl(query, off) {
  const params = new URLSearchParams({ offset: off, limit: PAGE_SIZE });
  if (query) {
    params.set("q", query);
    return `/api/search?${params.toString()}`;
  }
  return `/api/images?${params.toString()}`;
}

function cardHTML(item) {
  const tagStr = item.tags && item.tags.length ? item.tags.join(" · ") : "";
  return `
    <div class="card" data-id="${item.id}">
      <img src="/thumb/${encodeURIComponent(item.file_name)}" loading="lazy" alt="${item.file_name}" />
      ${tagStr ? `<div class="card-tags">${tagStr}</div>` : ""}
    </div>
  `;
}

async function loadPage(query, off, token) {
  if (activeController) activeController.abort();
  activeController = new AbortController();

  isLoading = true;
  if (off === 0) {
    searchStatus.textContent = query ? "Searching…" : "";
  } else {
    bottomLoader.classList.remove("hidden");
  }

  try {
    const res = await fetch(buildUrl(query, off), {
      signal: activeController.signal,
    });
    const data = await res.json();

    // Stale response from a superseded query/page — ignore it.
    if (token !== requestToken) return;

    if (off === 0) {
      grid.innerHTML = "";
    }

    if (data.items.length > 0) {
      grid.insertAdjacentHTML("beforeend", data.items.map(cardHTML).join(""));
    }

    hasMore = data.has_more;
    offset = off + data.items.length;

    const nothingAtAll = off === 0 && data.items.length === 0;
    emptyState.classList.toggle("hidden", !nothingAtAll);
    grid.classList.toggle("hidden", nothingAtAll);

    searchStatus.textContent = "";
  } catch (err) {
    if (err.name !== "AbortError") {
      searchStatus.textContent = "Error loading results";
    }
  } finally {
    isLoading = false;
    bottomLoader.classList.add("hidden");
  }
}

function resetAndLoad(query) {
  requestToken += 1;
  const token = requestToken;
  currentQuery = query;
  offset = 0;
  hasMore = true;
  loadPage(query, 0, token);
}

function maybeLoadMore() {
  if (isLoading || !hasMore) return;
  const token = requestToken;
  loadPage(currentQuery, offset, token);
}

// Debounced, no-Enter-required search
let debounceTimer = null;
searchInput.addEventListener("input", () => {
  const value = searchInput.value.trim();
  clearTimeout(debounceTimer);
  searchStatus.textContent = value ? "Typing…" : "";
  debounceTimer = setTimeout(() => resetAndLoad(value), DEBOUNCE_MS);
});

// Infinite scroll
const observer = new IntersectionObserver(
  (entries) => {
    if (entries[0].isIntersecting) maybeLoadMore();
  },
  { rootMargin: "600px" },
);
observer.observe(sentinel);

// Modal preview
grid.addEventListener("click", async (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  const id = card.dataset.id;

  try {
    const res = await fetch(`/api/image/${encodeURIComponent(id)}`);
    if (!res.ok) return;
    const detail = await res.json();
    openModal(detail);
  } catch (_) {
    // ignore
  }
});

function openModal(detail) {
  modalImage.src = `/media/${encodeURIComponent(detail.file_name)}`;
  modalFilename.textContent = detail.file_name;

  const metaParts = [];
  if (detail.width && detail.height)
    metaParts.push(`${detail.width}×${detail.height}`);
  if (detail.file_size_human) metaParts.push(detail.file_size_human);
  if (detail.date_taken) metaParts.push(detail.date_taken);
  if (detail.camera) metaParts.push(detail.camera);
  modalMeta.textContent = metaParts.join("  ·  ");

  modalTags.innerHTML = (detail.tags || [])
    .map((t) => `<span class="tag-pill">${t.tag} · ${t.confidence_pct}%</span>`)
    .join("");

  modal.classList.remove("hidden");
}

function closeModal() {
  modal.classList.add("hidden");
  modalImage.src = "";
}

modalClose.addEventListener("click", closeModal);
document.querySelector(".modal-backdrop").addEventListener("click", closeModal);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

// Initial load
resetAndLoad("");
