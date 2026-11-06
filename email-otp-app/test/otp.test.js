process.env.MAIL_MODE = "console";
delete process.env.SMTP_HOST;

const { test, before, after } = require("node:test");
const assert = require("node:assert");
const app = require("../server");

let server, base;
const sent = {};
const origLog = console.log;
console.log = (...args) => {
  const m = String(args[0]).match(/OTP for (\S+): (\d{6})/);
  if (m) sent[m[1]] = m[2];
  else origLog(...args);
};

before(() => new Promise((r) => { server = app.listen(0, () => { base = `http://localhost:${server.address().port}`; r(); }); }));
after(() => { server.close(); console.log = origLog; });

const post = async (path, body) => {
  const r = await fetch(base + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return { status: r.status, body: await r.json() };
};

test("rejects invalid email", async () => {
  const r = await post("/api/send-otp", { email: "not-an-email" });
  assert.strictEqual(r.status, 400);
});

test("send + verify happy path, OTP is single-use", async () => {
  const email = "Alice@Example.com";
  const s = await post("/api/send-otp", { email });
  assert.strictEqual(s.status, 200);
  const otp = sent["alice@example.com"];
  assert.match(otp, /^\d{6}$/);

  const v = await post("/api/verify-otp", { email, otp });
  assert.strictEqual(v.status, 200);
  assert.strictEqual(v.body.verified, true);

  const again = await post("/api/verify-otp", { email, otp });
  assert.strictEqual(again.status, 400);
});

test("wrong OTP counts attempts and locks after 5", async () => {
  const email = "bob@example.com";
  await post("/api/send-otp", { email });
  const real = sent[email];
  const wrong = real === "000000" ? "111111" : "000000";
  for (let i = 0; i < 5; i++) {
    const r = await post("/api/verify-otp", { email, otp: wrong });
    assert.strictEqual(r.status, 400);
  }
  const locked = await post("/api/verify-otp", { email, otp: real });
  assert.strictEqual(locked.status, 429);
});

test("resend cooldown enforced", async () => {
  const email = "carol@example.com";
  assert.strictEqual((await post("/api/send-otp", { email })).status, 200);
  const r = await post("/api/send-otp", { email });
  assert.strictEqual(r.status, 429);
  assert.ok(r.body.retryAfter > 0);
});
