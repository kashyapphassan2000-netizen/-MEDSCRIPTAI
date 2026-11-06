# Email OTP App

Enter any email address, get a 6-digit OTP in that inbox, then verify it.

Node.js + Express backend, Nodemailer for delivery, plain HTML/JS frontend.

## Run

```bash
cd email-otp-app
npm install
cp .env.example .env   # fill in SMTP to send real emails
npm start              # http://localhost:3000
```

## Email delivery modes

| Mode | When | What happens |
|---|---|---|
| `smtp` | `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS` set | Real email to the address entered |
| `ethereal` | no SMTP set (default) | Free test inbox; UI shows a link to view the email. Nothing reaches the real inbox |
| `console` | `MAIL_MODE=console` | OTP printed to server log |

**Gmail:** turn on 2-Step Verification → create an App Password → use it as `SMTP_PASS`.
Gmail caps at ~500 emails/day. For production volume use a transactional provider
(Resend, Brevo, SendGrid, Amazon SES) — they all give SMTP credentials that plug straight into `.env`.

## API

`POST /api/send-otp` `{ "email": "a@b.com" }` → `{ message, expiresInSeconds, resendInSeconds }`

`POST /api/verify-otp` `{ "email": "a@b.com", "otp": "123456" }` → `{ verified: true }`

## Security built in

- OTP generated with `crypto.randomInt` (CSPRNG)
- Stored only as salted HMAC-SHA256, compared with `timingSafeEqual`
- Expires after 5 min (`OTP_TTL_MINUTES`), single-use
- Max 5 wrong attempts per code, 30 s resend cooldown, max 5 sends/email/hour

## Limits (know these before deploying)

- OTP store is in memory: restarts wipe pending codes and it won't work across multiple instances. Move it to Redis for production.
- No per-IP rate limit; add `express-rate-limit` before exposing publicly or someone can use it to spam arbitrary addresses.

## Test

```bash
npm test
```
