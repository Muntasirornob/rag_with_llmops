(() => {
  const MAX_FILES = 10;
  const MAX_FILE_BYTES = 20 * 1024 * 1024;

  const els = {
    uploadForm: document.getElementById("upload-form"),
    fileInput: document.getElementById("file-input"),
    uploadBtn: document.getElementById("upload-btn"),
    sessionId: document.getElementById("session-id"),
    newSessionBtn: document.getElementById("new-session-btn"),
    status: document.getElementById("status"),
    messages: document.getElementById("messages"),
    chatForm: document.getElementById("chat-form"),
    input: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-btn"),
  };

  let sessionId = null;
  let busy = false;

  function setStatus(text, kind = "") {
    els.status.textContent = text;
    els.status.className = `status ${kind}`.trim();
  }

  function updateControls() {
    const hasSession = Boolean(sessionId);
    els.input.disabled = !hasSession || busy;
    els.sendBtn.disabled = !hasSession || busy;
    els.uploadBtn.disabled = busy || els.fileInput.files.length === 0;
    els.newSessionBtn.disabled = busy;
  }

  function clearEmptyState() {
    const empty = els.messages.querySelector(".empty");
    if (empty) empty.remove();
  }

  function addMessage(text, role) {
    clearEmptyState();
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = text; // textContent avoids injecting HTML from responses
    els.messages.appendChild(div);
    els.messages.scrollTop = els.messages.scrollHeight;
    return div;
  }

  function validateFiles(files) {
    if (files.length > MAX_FILES) {
      return `Select at most ${MAX_FILES} files.`;
    }
    for (const f of files) {
      if (f.size > MAX_FILE_BYTES) {
        return `${f.name} is larger than 20 MB.`;
      }
    }
    return null;
  }

  async function readError(response) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return body.detail;
    } catch (_) {
      /* response was not JSON */
    }
    return `Request failed (${response.status})`;
  }

  async function uploadFiles(event) {
    event.preventDefault();
    const files = Array.from(els.fileInput.files);
    const problem = validateFiles(files);
    if (problem) {
      setStatus(problem, "error");
      return;
    }

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    if (sessionId) formData.append("session_id", sessionId);

    busy = true;
    updateControls();
    setStatus("Uploading and indexing...");

    try {
      const response = await fetch("/upload", { method: "POST", body: formData });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const data = await response.json();
      sessionId = data.session_id;
      els.sessionId.textContent = sessionId;
      els.fileInput.value = "";
      setStatus(`${data.message}. Ask a question below.`, "success");
    } catch (err) {
      setStatus(err.message || "Upload failed.", "error");
    } finally {
      busy = false;
      updateControls();
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = els.input.value.trim();
    if (!text || !sessionId || busy) return;

    addMessage(text, "user");
    els.input.value = "";
    busy = true;
    updateControls();

    const typing = addMessage("Thinking...", "bot");
    typing.classList.add("typing");

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      if (!response.ok) {
        throw new Error(await readError(response));
      }
      const data = await response.json();
      typing.remove();
      addMessage(data.answer, "bot");
    } catch (err) {
      typing.remove();
      addMessage(err.message || "Something went wrong.", "error");
    } finally {
      busy = false;
      updateControls();
      els.input.focus();
    }
  }

  function startNewSession() {
    sessionId = null;
    els.sessionId.textContent = "none";
    els.messages.innerHTML =
      '<div class="empty">Upload a document, then ask a question about it.</div>';
    els.fileInput.value = "";
    setStatus("Started a new session.");
    updateControls();
  }

  els.fileInput.addEventListener("change", updateControls);
  els.uploadForm.addEventListener("submit", uploadFiles);
  els.chatForm.addEventListener("submit", sendMessage);
  els.newSessionBtn.addEventListener("click", startNewSession);

  // Enter sends, Shift+Enter inserts a newline
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      els.chatForm.requestSubmit();
    }
  });

  updateControls();
})();
