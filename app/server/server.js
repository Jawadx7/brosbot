// Express + Socket.io group chat server for BrosBot.
//
// Serves the static client (app/client) and maintains an in-memory
// history of the last MAX_HISTORY messages. Human users connect via
// the browser (Socket.io client). The Python bot agent (Step 10)
// posts its replies through the HTTP endpoint below rather than
// connecting as a full Socket.io client -- simpler on the Python side,
// and keeps the "who's allowed to speak" logic entirely in trigger.py.
//
// Message shape (matches agent/trigger.py and agent/prompt.py exactly):
//   {
//     sender:    string,   // display name, or BOT_NAME for the bot
//     text:      string,
//     is_bot:    boolean,
//     timestamp: number,   // unix SECONDS (not ms) -- matches Python's time.time()
//   }
//
// Run:
//   npm install
//   node server.js
//
// Env vars (see .env.example):
//   SERVER_PORT   default 3000
//   BOT_NAME      default "BrosBot"

const express = require("express");
const http = require("http");
const path = require("path");
const { Server } = require("socket.io");

const PORT = process.env.SERVER_PORT || 3000;
const BOT_NAME = process.env.BOT_NAME || "BrosBot";
const MAX_HISTORY = 100;

const app = express();
app.use(express.json());

// Serve the static frontend (app/client) at the site root.
app.use(express.static(path.join(__dirname, "..", "client")));

const server = http.createServer(app);

// Permissive CORS on the Socket.io layer -- useful once this is deployed
// behind Railway/ngrok (Step 11) and the client origin may not exactly
// match the server origin.
const io = new Server(server, {
  cors: { origin: "*" },
});

// ---------------------------------------------------------------------------
// In-memory history
// ---------------------------------------------------------------------------

/** @type {Array<{sender: string, text: string, is_bot: boolean, timestamp: number}>} */
let history = [];

function addToHistory(message) {
  history.push(message);
  if (history.length > MAX_HISTORY) {
    history.shift();
  }
}

function nowSeconds() {
  return Date.now() / 1000;
}

// ---------------------------------------------------------------------------
// Socket.io -- human participants
// ---------------------------------------------------------------------------

io.on("connection", (socket) => {
  socket.data.name = null;

  socket.on("user:join", (payload) => {
    const name = (payload && payload.name ? String(payload.name) : "").trim();
    socket.data.name = name || `Guest${Math.floor(Math.random() * 10000)}`;

    // Send the new arrival everything currently in memory so their
    // screen isn't blank on join.
    socket.emit("history", history);

    console.log(`[join] ${socket.data.name} (${socket.id})`);
  });

  socket.on("user:message", (payload) => {
    const text = (payload && payload.text ? String(payload.text) : "").trim();
    if (!text) return;

    if (!socket.data.name) {
      // Guard against messages sent before user:join completed.
      socket.emit("error:message", { error: "Join before sending messages." });
      return;
    }

    const message = {
      sender: socket.data.name,
      text: text,
      is_bot: false,
      timestamp: nowSeconds(),
    };

    addToHistory(message);
    io.emit("user:message", message);
  });

  socket.on("disconnect", () => {
    if (socket.data.name) {
      console.log(`[leave] ${socket.data.name} (${socket.id})`);
    }
  });
});

// ---------------------------------------------------------------------------
// HTTP -- bot agent posts replies here (Step 10 calls this, not Socket.io)
// ---------------------------------------------------------------------------

app.post("/api/bot-message", (req, res) => {
  const text = (req.body && req.body.text ? String(req.body.text) : "").trim();

  if (!text) {
    return res.status(400).json({ ok: false, error: "text is required" });
  }

  const message = {
    sender: BOT_NAME,
    text: text,
    is_bot: true,
    timestamp: nowSeconds(),
  };

  addToHistory(message);
  io.emit("bot:message", message);

  res.json({ ok: true, message: message });
});

// Lets the Python bot agent (or trigger.py's cooldown logic) pull current
// history over plain HTTP instead of needing a full Socket.io client.
app.get("/api/history", (req, res) => {
  res.json({ ok: true, history: history });
});

app.get("/api/health", (req, res) => {
  res.json({ ok: true, historyLength: history.length, botName: BOT_NAME });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

server.listen(PORT, () => {
  console.log(`BrosBot server running on http://localhost:${PORT}`);
});
