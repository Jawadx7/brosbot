// Browser-side Socket.io logic for the BrosBot group chat UI.
// No build step, no framework -- plain DOM + the Socket.io client
// that the server auto-serves at /socket.io/socket.io.js.

(function () {
  "use strict";

  const socket = io();

  // -- DOM refs
  const joinScreen = document.getElementById("join-screen");
  const chatScreen = document.getElementById("chat-screen");
  const nameInput = document.getElementById("name-input");
  const joinBtn = document.getElementById("join-btn");
  const joinError = document.getElementById("join-error");
  const messagesEl = document.getElementById("messages");
  const messageForm = document.getElementById("message-form");
  const messageInput = document.getElementById("message-input");
  const statusDot = document.getElementById("connection-status");

  let myName = null;

  // ---------------------------------------------------------------------
  // Join flow
  // ---------------------------------------------------------------------

  function attemptJoin() {
    const name = nameInput.value.trim();
    if (!name) {
      joinError.textContent = "Enter a name to continue.";
      return;
    }
    joinError.textContent = "";
    myName = name;

    socket.emit("user:join", { name: name });

    joinScreen.classList.add("hidden");
    chatScreen.classList.remove("hidden");
    messageInput.focus();
  }

  joinBtn.addEventListener("click", attemptJoin);
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") attemptJoin();
  });

  // ---------------------------------------------------------------------
  // Sending messages
  // ---------------------------------------------------------------------

  messageForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (!text) return;

    socket.emit("user:message", { text: text });
    messageInput.value = "";
  });

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------

  function formatTime(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function renderMessage(msg) {
    const el = document.createElement("div");
    el.className = "message";

    if (msg.is_bot) {
      el.classList.add("bot");
    } else if (msg.sender === myName) {
      el.classList.add("self");
    }

    const senderEl = document.createElement("div");
    senderEl.className = "sender";
    senderEl.textContent = msg.sender;

    if (msg.is_bot) {
      const badge = document.createElement("span");
      badge.className = "ai-badge";
      badge.textContent = "AI";
      senderEl.appendChild(badge);
    }

    const textEl = document.createElement("div");
    textEl.className = "text";
    textEl.textContent = msg.text;

    const timeEl = document.createElement("div");
    timeEl.className = "time";
    timeEl.textContent = formatTime(msg.timestamp);

    el.appendChild(senderEl);
    el.appendChild(textEl);
    el.appendChild(timeEl);

    messagesEl.appendChild(el);
    scrollToBottom();
  }

  function renderHistory(history) {
    messagesEl.innerHTML = "";
    history.forEach(renderMessage);
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ---------------------------------------------------------------------
  // Socket events
  // ---------------------------------------------------------------------

  socket.on("history", renderHistory);
  socket.on("user:message", renderMessage);
  socket.on("bot:message", renderMessage);

  socket.on("error:message", (payload) => {
    console.error("Server error:", payload.error);
  });

  socket.on("connect", () => {
    statusDot.classList.remove("status-disconnected");
    statusDot.classList.add("status-connected");
    // Re-join automatically if we reconnect mid-session
    if (myName) {
      socket.emit("user:join", { name: myName });
    }
  });

  socket.on("disconnect", () => {
    statusDot.classList.remove("status-connected");
    statusDot.classList.add("status-disconnected");
  });
})();
