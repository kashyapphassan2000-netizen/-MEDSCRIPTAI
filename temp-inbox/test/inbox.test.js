// End-to-end tests against local fake mail.tm and Guerrilla Mail servers.
const { test, before, after } = require("node:test");
const assert = require("node:assert");
const express = require("express");
const { extractOtp } = require("../otp");

let fakeMailtm, fakeGuerrilla, server, base;
const mailtmAccounts = new Map(); // address -> { id, password, messages }
const guerrillaBox = { user: null, messages: [] };

function startFakes() {
  const m = express();
  m.use(express.json());
  m.get("/domains", (_q, r) => r.json({ "hydra:member": [{ domain: "fldubai.com", isActive: true, isPrivate: false }] }));
  m.post("/accounts", (q, r) => {
    if (mailtmAccounts.has(q.body.address)) return r.status(422).json({ "hydra:description": "address: This value is already used." });
    const id = "acc" + mailtmAccounts.size;
    mailtmAccounts.set(q.body.address, { id, password: q.body.password, messages: [] });
    r.status(201).json({ id, address: q.body.address });
  });
  m.post("/token", (q, r) => {
    const a = mailtmAccounts.get(q.body.address);
    if (!a || a.password !== q.body.password) return r.status(401).json({ message: "Invalid credentials." });
    r.json({ token: "tok:" + q.body.address });
  });
  const auth = (q) => mailtmAccounts.get(String(q.headers.authorization || "").replace("Bearer tok:", ""));
  m.get("/messages", (q, r) => r.json({ "hydra:member": auth(q).messages }));
  m.get("/messages/:id", (q, r) => r.json({ ...auth(q).messages.find((x) => x.id === q.params.id), text: "Use 482913 to sign in.", html: ["<p>Your verification code is <b>482913</b></p>"] }));
  m.delete("/accounts/:id", (q, r) => {
    for (const [k, v] of mailtmAccounts) if (v.id === q.params.id) mailtmAccounts.delete(k);
    r.status(204).end();
  });

  const g = express();
  g.get("/ajax.php", (q, r) => {
    const f = q.query.f;
    if (f === "get_email_address") return r.json({ email_addr: "rnd@guerrillamailblock.com", sid_token: "sid1" });
    if (f === "set_email_user") { guerrillaBox.user = q.query.email_user; return r.json({ email_addr: `${q.query.email_user}@guerrillamailblock.com`, sid_token: "sid2" }); }
    if (f === "check_email") return r.json({ list: guerrillaBox.messages });
    if (f === "fetch_email") return r.json({ ...guerrillaBox.messages.find((x) => x.mail_id === q.query.email_id), mail_body: "<div>Code: 7731</div>" });
    if (f === "forget_me") return r.json(true);
    r.status(400).json({});
  });

  return Promise.all([
    new Promise((ok) => (fakeMailtm = m.listen(0, ok))),
    new Promise((ok) => (fakeGuerrilla = g.listen(0, ok))),
  ]);
}

before(async () => {
  await startFakes();
  process.env.MAILTM_BASE = `http://localhost:${fakeMailtm.address().port}`;
  process.env.GUERRILLA_BASE = `http://localhost:${fakeGuerrilla.address().port}/ajax.php`;
  process.env.INBOX_MINUTES = "10";
  const { app } = require("../server");
  await new Promise((ok) => (server = app.listen(0, ok)));
  base = `http://localhost:${server.address().port}`;
});
after(() => { server.close(); fakeMailtm.close(); fakeGuerrilla.close(); });

const req = async (method, path, body) => {
  const r = await fetch(base + path, { method, headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });
  return { status: r.status, body: r.status === 204 ? {} : await r.json() };
};

test("lists domains from both providers", async () => {
  const r = await req("GET", "/api/domains");
  assert.strictEqual(r.status, 200);
  const names = r.body.domains.map((d) => d.domain);
  assert.ok(names.includes("fldubai.com"));
  assert.ok(names.includes("sharklasers.com"));
  assert.strictEqual(r.body.lifetimeSeconds, 600);
});

test("mail.tm: create chosen name, receive mail, OTP extracted, 10 min expiry", async () => {
  const c = await req("POST", "/api/inboxes", { username: "Raman", domain: "fldubai.com", provider: "mailtm" });
  assert.strictEqual(c.status, 201);
  assert.strictEqual(c.body.address, "raman@fldubai.com");
  const ttl = c.body.expiresAt - c.body.createdAt;
  assert.strictEqual(ttl, 10 * 60 * 1000);

  mailtmAccounts.get("raman@fldubai.com").messages.push({
    id: "m1", from: { address: "noreply@shop.com", name: "Shop" }, subject: "Your OTP is 482913", intro: "Use it within 5 minutes", createdAt: new Date().toISOString(),
  });
  const list = await req("GET", `/api/inboxes/${c.body.id}/messages`);
  assert.strictEqual(list.body.messages.length, 1);
  assert.strictEqual(list.body.messages[0].otp, "482913");

  const full = await req("GET", `/api/inboxes/${c.body.id}/messages/m1`);
  assert.match(full.body.html, /482913/);
  assert.strictEqual(full.body.otp, "482913");

  // tokens never leak to the client
  assert.ok(!JSON.stringify(c.body).includes("tok:"));
});

test("mail.tm: taken name returns 409", async () => {
  const r = await req("POST", "/api/inboxes", { username: "raman", domain: "fldubai.com", provider: "mailtm" });
  assert.strictEqual(r.status, 409);
});

test("guerrilla: chosen name on chosen domain, welcome mail hidden, OTP from body", async () => {
  guerrillaBox.messages = [
    { mail_id: "1", mail_from: "no-reply@guerrillamail.com", mail_subject: "Welcome to Guerrilla Mail", mail_excerpt: "", mail_timestamp: "1793985285" },
    { mail_id: "2", mail_from: "auth@site.io", mail_subject: "Sign in", mail_excerpt: "Your login code: 7731", mail_timestamp: "1793985300" },
  ];
  const c = await req("POST", "/api/inboxes", { username: "raman", domain: "sharklasers.com", provider: "guerrilla" });
  assert.strictEqual(c.status, 201);
  assert.strictEqual(c.body.address, "raman@sharklasers.com");
  const list = await req("GET", `/api/inboxes/${c.body.id}/messages`);
  assert.strictEqual(list.body.messages.length, 1);
  assert.strictEqual(list.body.messages[0].otp, "7731");
});

test("rejects bad names and unknown domains", async () => {
  assert.strictEqual((await req("POST", "/api/inboxes", { username: "a", domain: "fldubai.com", provider: "mailtm" })).status, 400);
  assert.strictEqual((await req("POST", "/api/inboxes", { username: "raman2", domain: "work.euron.com", provider: "mailtm" })).status, 400);
});

test("delete removes inbox and provider account", async () => {
  const c = await req("POST", "/api/inboxes", { username: "tempuser", domain: "fldubai.com", provider: "mailtm" });
  assert.ok(mailtmAccounts.has("tempuser@fldubai.com"));
  assert.strictEqual((await req("DELETE", `/api/inboxes/${c.body.id}`)).status, 204);
  assert.strictEqual((await req("GET", `/api/inboxes/${c.body.id}/messages`)).status, 404);
  await new Promise((r) => setTimeout(r, 50));
  assert.ok(!mailtmAccounts.has("tempuser@fldubai.com"));
});

test("expired inbox returns 410", async () => {
  const { inboxes } = require("../server");
  const c = await req("POST", "/api/inboxes", { username: "shortlived", domain: "fldubai.com", provider: "mailtm" });
  inboxes.get(c.body.id).expiresAt = Date.now() - 1;
  assert.strictEqual((await req("GET", `/api/inboxes/${c.body.id}/messages`)).status, 410);
});

test("OTP extractor", () => {
  assert.strictEqual(extractOtp("Your OTP is 123456"), "123456");
  assert.strictEqual(extractOtp("Verification code: 482-913"), "482913");
  assert.strictEqual(extractOtp("Sign in", "Enter 90817 to verify your account"), "90817");
  assert.strictEqual(extractOtp("Newsletter", "Our 2026 sale is live"), null);
  assert.strictEqual(extractOtp("Use code AB12CD"), "AB12CD");
  assert.strictEqual(extractOtp("Verify", "Your verification code is 604918. It expires soon."), "604918");
  assert.strictEqual(extractOtp("Your code was 3391"), "3391");
  assert.strictEqual(extractOtp("Welcome", "Thanks for joining, enjoy your stay"), null);
});
