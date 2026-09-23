const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const errorBox = document.getElementById("errorBox");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

const emotionName = document.getElementById("emotionName");
const emotionEmoji = document.getElementById("emotionEmoji");
const emotionCore = document.getElementById("emotionCore");
const confidence = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const probabilityList = document.getElementById("probabilityList");
const resultBadge = document.getElementById("resultBadge");
const distributionState = document.getElementById("distributionState");

const EMOJIS = {
  sadness: "😢",
  anger: "😠",
  love: "❤️",
  surprise: "😲",
  fear: "😨",
  joy: "😄"
};

const LABELS = ["sadness", "anger", "love", "surprise", "fear", "joy"];

function updateCount() {
  charCount.textContent = `${textInput.value.length} / 2000`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.add("show");
}

function hideError() {
  errorBox.classList.remove("show");
}

function setLoading(isLoading) {
  analyzeBtn.classList.toggle("loading", isLoading);
  analyzeBtn.querySelector(".btn-content span:last-child").textContent =
    isLoading ? "Reading emotion…" : "Analyze emotion";
  analyzeBtn.querySelector(".btn-icon").textContent = isLoading ? "◌" : "✦";
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function renderProbabilities(probabilities, topEmotion) {
  probabilityList.innerHTML = "";

  const rows = LABELS
    .map(label => ({ label, value: Number(probabilities[label] ?? 0) }))
    .sort((a, b) => b.value - a.value);

  rows.forEach(({ label, value }, index) => {
    const row = document.createElement("div");
    row.className = `prob-row ${label === topEmotion ? "top" : ""}`;

    row.innerHTML = `
      <span class="prob-name">${label}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width: 0%"></div>
      </div>
      <span class="prob-value">${formatPercent(value)}</span>
    `;

    probabilityList.appendChild(row);

    requestAnimationFrame(() => {
      const fill = row.querySelector(".prob-fill");
      setTimeout(() => {
        fill.style.width = `${Math.max(value * 100, value > 0 ? 1 : 0)}%`;
      }, 40 + index * 55);
    });
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();

    if (response.ok && data.model_loaded) {
      statusPill.classList.add("online");
      statusPill.classList.remove("offline");
      statusText.textContent = "AI engine online";
    } else {
      throw new Error("Model is not loaded");
    }
  } catch {
    statusPill.classList.remove("online");
    statusPill.classList.add("offline");
    statusText.textContent = "AI engine unavailable";
  }
}

async function analyze() {
  const text = textInput.value.trim();

  if (!text) {
    textInput.focus();
    showError("Write something first — even one sentence is enough.");
    return;
  }

  hideError();
  setLoading(true);
  resultBadge.textContent = "ANALYZING";
  resultBadge.classList.remove("live");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("The server returned an invalid response.");
    }

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    const emotion = data.predicted_emotion;
    const score = Number(data.confidence);

    emotionName.textContent = emotion;
    emotionEmoji.textContent = EMOJIS[emotion] || "✦";
    confidence.textContent = `${formatPercent(score)} model confidence`;
    confidenceValue.textContent = formatPercent(score);
    confidenceBar.style.width = `${Math.max(0, Math.min(score * 100, 100))}%`;

    resultBadge.textContent = "LIVE RESULT";
    resultBadge.classList.add("live");
    distributionState.textContent = "Sorted by signal";
    emotionCore.classList.remove("pop");
    void emotionCore.offsetWidth;
    emotionCore.classList.add("pop");

    renderProbabilities(data.all_probabilities, emotion);
  } catch (error) {
    resultBadge.textContent = "ERROR";
    showError(error.message || "Something went wrong while analyzing the text.");
  } finally {
    setLoading(false);
  }
}

textInput.addEventListener("input", () => {
  updateCount();
  hideError();
});

textInput.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    analyze();
  }
});

analyzeBtn.addEventListener("click", analyze);

clearBtn.addEventListener("click", () => {
  textInput.value = "";
  updateCount();
  hideError();
  textInput.focus();
});

document.querySelectorAll(".example-chip").forEach(button => {
  button.addEventListener("click", () => {
    textInput.value = button.textContent;
    updateCount();
    hideError();
    textInput.focus();
  });
});

updateCount();
checkHealth();
setInterval(checkHealth, 30000);
