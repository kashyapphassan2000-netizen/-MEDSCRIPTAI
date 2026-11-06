require("dotenv").config();
const express = require("express");
const crypto = require("crypto");
const path = require("path");
const { sendOtpEmail, mailMode } = require("./mailer");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const OTP_LENGTH = 6;
const OTP_TTL_MS = Number(process.env.OTP_TTL_MINUTES || 5) * 60 * 1000;
const RESEND_COOLDOWN_MS = 30 * 1000;
const MAX_ATTEMPTS = 5;
const MAX_SENDS_PER_HOUR = 5;

// In-memory store: email -> { hash, salt, expiresAt, attempts, lastSentAt, sends: [timestamps] }
// Swap for Redis in production (survives restarts, works across instances).
const store = new Map();

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

function normalizeEmail(email) {
  return String(email || "").trim().toLowerCase();
}

function generateOtp() {
  // Cryptographically secure, uniform 6-digit code
  return crypto.randomInt(0, 10 ** OTP_LENGTH).toString().padStart(OTP_LENGTH, "0");
}

function hashOtp(otp, salt) {
  return crypto.createHmac("sha256", salt).update(otp).digest("hex");
}

app.post("/api/send-otp", async (req, res) => {
  const email = normalizeEmail(req.body.email);
  if (!EMAIL_RE.test(email) || email.length > 254) {
    return res.status(400).json({ error: "Enter a valid email address." });
  }

  const now = Date.now();
  const entry = store.get(email) || { sends: [] };
  entry.sends = (entry.sends || []).filter((t) => now - t < 60 * 60 * 1000);

  if (entry.lastSentAt && now - entry.lastSentAt < RESEND_COOLDOWN_MS) {
    const wait = Math.ceil((RESEND_COOLDOWN_MS - (now - entry.lastSentAt)) / 1000);
    return res.status(429).json({ error: `Wait ${wait}s before requesting a new OTP.`, retryAfter: wait });
  }
  if (entry.sends.length >= MAX_SENDS_PER_HOUR) {
    return res.status(429).json({ error: "Too many OTP requests for this email. Try again in an hour." });
  }

  const otp = generateOtp();
  const salt = crypto.randomBytes(16).toString("hex");

  try {
    const info = await sendOtpEmail(email, otp, OTP_TTL_MS / 60000);
    store.set(email, {
      hash: hashOtp(otp, salt),
      salt,
      expiresAt: now + OTP_TTL_MS,
      attempts: 0,
      lastSentAt: now,
      sends: [...entry.sends, now],
    });
    const body = {
      message: `OTP sent to ${email}`,
      expiresInSeconds: OTP_TTL_MS / 1000,
      resendInSeconds: RESEND_COOLDOWN_MS / 1000,
    };
    if (info.previewUrl) body.previewUrl = info.previewUrl; // test mode only
    res.json(body);
  } catch (err) {
    console.error("Email send failed:", err.message);
    res.status(502).json({ error: "Could not send email. Check SMTP settings." });
  }
});

app.post("/api/verify-otp", (req, res) => {
  const email = normalizeEmail(req.body.email);
  const otp = String(req.body.otp || "").trim();
  const entry = store.get(email);

  if (!entry || !entry.hash) {
    return res.status(400).json({ error: "No OTP requested for this email." });
  }
  if (Date.now() > entry.expiresAt) {
    entry.hash = null;
    return res.status(400).json({ error: "OTP expired. Request a new one." });
  }
  if (entry.attempts >= MAX_ATTEMPTS) {
    entry.hash = null;
    return res.status(429).json({ error: "Too many wrong attempts. Request a new OTP." });
  }

  entry.attempts += 1;
  const candidate = Buffer.from(hashOtp(otp, entry.salt), "hex");
  const expected = Buffer.from(entry.hash, "hex");
  const ok = /^\d{6}$/.test(otp) && crypto.timingSafeEqual(candidate, expected);

  if (!ok) {
    const left = MAX_ATTEMPTS - entry.attempts;
    return res.status(400).json({ error: `Incorrect OTP. ${left} attempt(s) left.`, attemptsLeft: left });
  }

  entry.hash = null; // one-time use
  res.json({ verified: true, message: `${email} verified successfully.` });
});

// Purge stale entries every 10 minutes
setInterval(() => {
  const now = Date.now();
  for (const [email, e] of store) {
    if (now - (e.lastSentAt || 0) > 60 * 60 * 1000) store.delete(email);
  }
}, 10 * 60 * 1000).unref();

const PORT = Number(process.env.PORT || 3000);
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Email OTP app running at http://localhost:${PORT}  (mail mode: ${mailMode()})`);
  });
}

module.exports = app;
