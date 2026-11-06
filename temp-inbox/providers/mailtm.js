// mail.tm — private inboxes (password protected), free public API.
// Docs: https://docs.mail.tm
const crypto = require("crypto");

const BASE = process.env.MAILTM_BASE || "https://api.mail.tm";

async function call(path, { method = "GET", token, body } = {}) {
  const res = await fetch(BASE + path, {
    method,
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data["hydra:description"] || data.detail || data.message || `mail.tm error ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

module.exports = {
  name: "mailtm",
  label: "mail.tm (private)",

  async domains() {
    const data = await call("/domains");
    return (data["hydra:member"] || []).filter((d) => d.isActive && !d.isPrivate).map((d) => d.domain);
  },

  async create(username, domain) {
    const address = `${username}@${domain}`;
    const password = crypto.randomBytes(18).toString("base64url");
    let account;
    try {
      account = await call("/accounts", { method: "POST", body: { address, password } });
    } catch (e) {
      if (e.status === 422) {
        const taken = new Error(`${address} is already taken. Pick another name.`);
        taken.status = 409;
        throw taken;
      }
      throw e;
    }
    const { token } = await call("/token", { method: "POST", body: { address, password } });
    return { address, session: { token, accountId: account.id } };
  },

  async list(session) {
    const data = await call("/messages", { token: session.token });
    return (data["hydra:member"] || []).map((m) => ({
      id: m.id,
      from: m.from?.address || "",
      fromName: m.from?.name || "",
      subject: m.subject || "(no subject)",
      preview: m.intro || "",
      receivedAt: m.createdAt,
    }));
  },

  async read(session, id) {
    const m = await call(`/messages/${encodeURIComponent(id)}`, { token: session.token });
    return {
      id: m.id,
      from: m.from?.address || "",
      fromName: m.from?.name || "",
      subject: m.subject || "(no subject)",
      text: m.text || "",
      html: Array.isArray(m.html) ? m.html.join("") : m.html || "",
      receivedAt: m.createdAt,
    };
  },

  async destroy(session) {
    await call(`/accounts/${session.accountId}`, { method: "DELETE", token: session.token }).catch(() => {});
  },
};
