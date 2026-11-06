// Your own domain(s). Runs a built-in SMTP receiver, so ANY name@yourdomain works.
//
// Requirements (this is how email works, no way around it):
//   1. You own the domain (e.g. work.euron.one) and can edit its DNS.
//   2. DNS:  MX record  ->  the hostname/IP of the server running this app.
//   3. The server accepts inbound TCP port 25 (VPS like Hetzner, DigitalOcean, AWS EC2, Oracle Cloud).
//
// Enable with:  CUSTOM_DOMAINS=work.euron.one,mydomain.com  SMTP_PORT=25
const crypto = require("crypto");
const { SMTPServer } = require("smtp-server");
const { simpleParser } = require("mailparser");

const DOMAINS = String(process.env.CUSTOM_DOMAINS || "")
  .split(",").map((d) => d.trim().toLowerCase()).filter(Boolean);

const MAX_SIZE = 5 * 1024 * 1024;     // 5 MB per email
const MAX_PER_BOX = 50;

// address -> [messages]. Only addresses with a live inbox accept mail; everything else is rejected.
const boxes = new Map();

function startSmtp(port = Number(process.env.SMTP_PORT || 25)) {
  const server = new SMTPServer({
    authOptional: true,
    disabledCommands: ["AUTH"],
    size: MAX_SIZE,
    banner: "temp-inbox",
    logger: false,
    onRcptTo(address, _session, cb) {
      const rcpt = address.address.toLowerCase();
      if (!boxes.has(rcpt)) {
        const err = new Error("No such mailbox");
        err.responseCode = 550;
        return cb(err);
      }
      cb();
    },
    onData(stream, session, cb) {
      simpleParser(stream, { skipImageLinks: true })
        .then((mail) => {
          if (stream.sizeExceeded) {
            const err = new Error("Message too big");
            err.responseCode = 552;
            return cb(err);
          }
          const from = mail.from?.value?.[0] || {};
          for (const r of session.envelope.rcptTo) {
            const box = boxes.get(r.address.toLowerCase());
            if (!box) continue;
            box.unshift({
              id: crypto.randomUUID(),
              from: from.address || session.envelope.mailFrom?.address || "",
              fromName: from.name || "",
              subject: mail.subject || "(no subject)",
              text: mail.text || "",
              html: mail.html || "",
              preview: (mail.text || "").replace(/\s+/g, " ").trim().slice(0, 140),
              receivedAt: new Date().toISOString(),
            });
            box.length = Math.min(box.length, MAX_PER_BOX);
          }
          cb();
        })
        .catch(cb);
    },
  });
  server.on("error", (e) => console.error("[smtp]", e.message));
  server.listen(port, () => console.log(`[smtp] Receiving mail for ${DOMAINS.join(", ")} on port ${port}`));
  return server;
}

module.exports = {
  name: "custom",
  label: "Your domain (any name)",
  enabled: DOMAINS.length > 0,
  startSmtp,
  _boxes: boxes,

  async domains() {
    return DOMAINS;
  },

  async create(username, domain) {
    const address = `${username}@${domain}`;
    if (boxes.has(address)) {
      const e = new Error(`${address} is already open.`);
      e.status = 409;
      throw e;
    }
    boxes.set(address, []);
    return { address, session: { address } };
  },

  async list(session) {
    return (boxes.get(session.address) || []).map(({ text, html, ...m }) => m);
  },

  async read(session, id) {
    const m = (boxes.get(session.address) || []).find((x) => x.id === id);
    if (!m) {
      const e = new Error("Message not found.");
      e.status = 404;
      throw e;
    }
    return m;
  },

  async destroy(session) {
    boxes.delete(session.address);
  },
};
