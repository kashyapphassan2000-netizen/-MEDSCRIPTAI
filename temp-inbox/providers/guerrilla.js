// Guerrilla Mail — lets you pick any username on several domains.
// WARNING: inboxes are PUBLIC. Anyone who types the same username can read them.
// Docs: https://www.guerrillamail.com/GuerrillaMailAPI.html
const BASE = process.env.GUERRILLA_BASE || "https://api.guerrillamail.com/ajax.php";

// All of these deliver to the same username inbox.
const DOMAINS = [
  "sharklasers.com",
  "grr.la",
  "guerrillamail.com",
  "guerrillamail.net",
  "guerrillamail.org",
  "guerrillamailblock.com",
  "pokemail.net",
  "spam4.me",
];

async function call(params) {
  const url = BASE + "?" + new URLSearchParams(params).toString();
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const err = new Error(`Guerrilla Mail error ${res.status}`);
    err.status = 502;
    throw err;
  }
  return res.json();
}

function isWelcome(m) {
  return /guerrillamail/i.test(m.mail_from || "") && /welcome/i.test(m.mail_subject || "");
}

module.exports = {
  name: "guerrilla",
  label: "Guerrilla Mail (public inbox)",

  async domains() {
    return DOMAINS;
  },

  async create(username, domain) {
    const start = await call({ f: "get_email_address", lang: "en" });
    const set = await call({ f: "set_email_user", email_user: username, lang: "en", sid_token: start.sid_token });
    const sid = set.sid_token || start.sid_token;
    const user = String(set.email_addr || `${username}@x`).split("@")[0];
    return { address: `${user}@${domain}`, session: { sid } };
  },

  async list(session) {
    const data = await call({ f: "check_email", seq: 0, sid_token: session.sid });
    if (data.sid_token) session.sid = data.sid_token;
    return (data.list || []).filter((m) => !isWelcome(m)).map((m) => ({
      id: String(m.mail_id),
      from: m.mail_from || "",
      fromName: "",
      subject: m.mail_subject || "(no subject)",
      preview: m.mail_excerpt || "",
      receivedAt: new Date(Number(m.mail_timestamp) * 1000).toISOString(),
    }));
  },

  async read(session, id) {
    const m = await call({ f: "fetch_email", email_id: id, sid_token: session.sid });
    const html = m.mail_body || "";
    return {
      id: String(m.mail_id),
      from: m.mail_from || "",
      fromName: "",
      subject: m.mail_subject || "(no subject)",
      text: html.replace(/<style[\s\S]*?<\/style>/gi, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim(),
      html,
      receivedAt: new Date(Number(m.mail_timestamp) * 1000).toISOString(),
    };
  },

  async destroy(session) {
    await call({ f: "forget_me", sid_token: session.sid }).catch(() => {});
  },
};
