const express = require("express");
const crypto = require("crypto");
const path = require("path");
const { extractOtp } = require("./otp");

const custom = require("./providers/custom");
const providers = {
  ...(custom.enabled ? { custom } : {}),
  mailtm: require("./providers/mailtm"),
  guerrilla: require("./providers/guerrilla"),
};
const ORDER = ["custom", "mailtm", "guerrilla"];

const LIFETIME_MS = Number(process.env.INBOX_MINUTES || 10) * 60 * 1000;
const MAX_INBOXES = 10;
const USERNAME_RE = /^[a-z0-9][a-z0-9._-]{2,29}$/;

// id -> { provider, address, session, createdAt, expiresAt }   (provider tokens never leave the server)
const inboxes = new Map();

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

function getInbox(req, res) {
  const inbox = inboxes.get(req.params.id);
  if (!inbox) {
    res.status(404).json({ error: "Inbox not found or expired." });
    return null;
  }
  if (Date.now() > inbox.expiresAt) {
    expire(req.params.id);
    res.status(410).json({ error: "Inbox expired. Create a new one." });
    return null;
  }
  return inbox;
}

function expire(id) {
  const inbox = inboxes.get(id);
  if (!inbox) return;
  inboxes.delete(id);
  providers[inbox.provider].destroy(inbox.session).catch(() => {});
}

function publicView(id, i) {
  return { id, address: i.address, provider: i.provider, createdAt: i.createdAt, expiresAt: i.expiresAt };
}

function fail(res, err) {
  console.error("[provider]", err.message);
  res.status(err.status && err.status < 500 ? err.status : 502).json({ error: err.message || "Mail provider failed." });
}

app.get("/api/domains", async (_req, res) => {
  const out = [];
  await Promise.all(
    Object.values(providers).map(async (p) => {
      try {
        for (const d of await p.domains()) out.push({ domain: d, provider: p.name, label: p.label });
      } catch (e) {
        console.error(`[domains] ${p.name}:`, e.message);
      }
    })
  );
  if (!out.length) return res.status(502).json({ error: "Could not load domains from any mail provider." });
  out.sort((a, b) => ORDER.indexOf(a.provider) - ORDER.indexOf(b.provider));
  res.json({ domains: out, lifetimeSeconds: LIFETIME_MS / 1000 });
});

app.get("/api/inboxes", (_req, res) => {
  res.json({ inboxes: [...inboxes].map(([id, i]) => publicView(id, i)) });
});

app.post("/api/inboxes", async (req, res) => {
  const username = String(req.body.username || "").trim().toLowerCase();
  const domain = String(req.body.domain || "").trim().toLowerCase();
  const provider = providers[req.body.provider];

  if (!USERNAME_RE.test(username)) {
    return res.status(400).json({ error: "Name must be 3–30 characters: letters, numbers, dot, dash or underscore." });
  }
  if (!provider) return res.status(400).json({ error: "Pick a domain from the list." });
  if (inboxes.size >= MAX_INBOXES) return res.status(429).json({ error: `Max ${MAX_INBOXES} inboxes at once. Delete one first.` });

  try {
    const allowed = await provider.domains();
    if (!allowed.includes(domain)) return res.status(400).json({ error: `${domain} is not available. Pick a domain from the list.` });

    const { address, session } = await provider.create(username, domain);
    if ([...inboxes.values()].some((i) => i.address === address)) {
      return res.status(409).json({ error: `${address} is already open here.` });
    }
    const id = crypto.randomUUID();
    const now = Date.now();
    const inbox = { provider: provider.name, address, session, createdAt: now, expiresAt: now + LIFETIME_MS };
    inboxes.set(id, inbox);
    res.status(201).json(publicView(id, inbox));
  } catch (err) {
    fail(res, err);
  }
});

app.get("/api/inboxes/:id/messages", async (req, res) => {
  const inbox = getInbox(req, res);
  if (!inbox) return;
  try {
    const messages = await providers[inbox.provider].list(inbox.session);
    res.json({
      expiresAt: inbox.expiresAt,
      messages: messages.map((m) => ({ ...m, otp: extractOtp(m.subject, m.preview) })),
    });
  } catch (err) {
    fail(res, err);
  }
});

app.get("/api/inboxes/:id/messages/:mid", async (req, res) => {
  const inbox = getInbox(req, res);
  if (!inbox) return;
  try {
    const m = await providers[inbox.provider].read(inbox.session, req.params.mid);
    res.json({ ...m, otp: extractOtp(m.subject, m.text || m.html) });
  } catch (err) {
    fail(res, err);
  }
});

app.post("/api/inboxes/:id/extend", (req, res) => {
  const inbox = getInbox(req, res);
  if (!inbox) return;
  inbox.expiresAt = Date.now() + LIFETIME_MS;
  res.json(publicView(req.params.id, inbox));
});

app.delete("/api/inboxes/:id", (req, res) => {
  expire(req.params.id);
  res.status(204).end();
});

setInterval(() => {
  const now = Date.now();
  for (const [id, i] of inboxes) if (now > i.expiresAt) expire(id);
}, 15 * 1000).unref();

const PORT = Number(process.env.PORT || 4000);
if (require.main === module) {
  app.listen(PORT, () => console.log(`Temp Inbox running at http://localhost:${PORT}`));
  if (custom.enabled) custom.startSmtp();
}

module.exports = { app, providers, inboxes };
