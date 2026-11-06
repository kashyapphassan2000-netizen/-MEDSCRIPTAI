// Real SMTP delivery into the built-in receiver: name@work.euron.one style custom domain.
process.env.CUSTOM_DOMAINS = "work.euron.one";
const { test, before, after } = require("node:test");
const assert = require("node:assert");
const nodemailer = require("nodemailer");

let web, smtp, base, port;
before(async () => {
  delete require.cache[require.resolve("../providers/custom")];
  delete require.cache[require.resolve("../server")];
  const custom = require("../providers/custom");
  const { app } = require("../server");
  port = 2500 + Math.floor(Math.random() * 1000);
  smtp = custom.startSmtp(port);
  await new Promise((ok) => (web = app.listen(0, ok)));
  base = `http://localhost:${web.address().port}`;
});
after(() => { web.close(); smtp.close(); });

const req = async (method, path, body) => {
  const r = await fetch(base + path, { method, headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });
  return { status: r.status, body: r.status === 204 ? {} : await r.json() };
};
const send = (to, subject, text) =>
  nodemailer.createTransport({ host: "127.0.0.1", port, secure: false, ignoreTLS: true })
    .sendMail({ from: "Euron <noreply@euron.one>", to, subject, text });

test("any name on your own domain receives real SMTP mail with OTP", async () => {
  const d = await req("GET", "/api/domains");
  assert.strictEqual(d.body.domains[0].domain, "work.euron.one");

  const c = await req("POST", "/api/inboxes", { username: "raman", domain: "work.euron.one", provider: "custom" });
  assert.strictEqual(c.status, 201);
  assert.strictEqual(c.body.address, "raman@work.euron.one");

  await send("raman@work.euron.one", "Verify your account", "Your verification code is 604918. It expires in 10 minutes.");
  const list = await req("GET", `/api/inboxes/${c.body.id}/messages`);
  assert.strictEqual(list.body.messages.length, 1);
  assert.strictEqual(list.body.messages[0].otp, "604918");
  assert.strictEqual(list.body.messages[0].from, "noreply@euron.one");

  const full = await req("GET", `/api/inboxes/${c.body.id}/messages/${list.body.messages[0].id}`);
  assert.match(full.body.text, /604918/);
});

test("mail to an address with no open inbox is rejected (550)", async () => {
  await assert.rejects(send("nobody@work.euron.one", "x", "y"), /550|No such mailbox/);
});

test("after delete, mail to that address is rejected", async () => {
  const c = await req("POST", "/api/inboxes", { username: "kiran", domain: "work.euron.one", provider: "custom" });
  await req("DELETE", `/api/inboxes/${c.body.id}`);
  await assert.rejects(send("kiran@work.euron.one", "x", "y"), /550|No such mailbox/);
});

test("capture mode: an app that logs in with SMTP user/pass delivers into the inbox", async () => {
  const c = await req("POST", "/api/inboxes", { username: "sede", domain: "work.euron.one", provider: "custom" });
  assert.strictEqual(c.status, 201);
  await nodemailer.createTransport({ host: "127.0.0.1", port, secure: false, ignoreTLS: true, auth: { user: "myapp", pass: "anything" } })
    .sendMail({ from: "My App <noreply@work.euron.one>", to: "sede@work.euron.one", subject: "Login code", text: "Your OTP is 771204" });
  const list = await req("GET", `/api/inboxes/${c.body.id}/messages`);
  assert.strictEqual(list.body.messages[0].otp, "771204");
});
