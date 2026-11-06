# Temp Inbox

Type any name, pick a domain, press **+** → you get a real working email address for 10 minutes.
Put that address into any website's signup/OTP form. The email lands here within ~5 seconds
and the OTP is pulled out with a **Copy OTP** button.

## Run

```bash
cd temp-inbox
npm install
npm start          # http://localhost:4000
```

## Where the mail actually goes (3 modes)

| Mode | Example | Setup | Who can read it |
|---|---|---|---|
| **Your domain** | `raman@work.euron.one`, `anything@yourdomain.com` | You own the domain + 1 DNS record (below) | Only you |
| mail.tm | `raman@fldubai.com` | None | Only you (password-protected) |
| Guerrilla Mail | `raman@sharklasers.com`, `@grr.la`, … | None | **Public** – anyone typing the same name sees it |

### Hard rule of email (no tool can bypass this)

Mail for `name@some-domain.com` is delivered to whatever server that domain's **MX record** points to.
So you can receive at a domain **only if you control its DNS**. You cannot receive at
`@gmail.com`, `@euron.one`, or a company domain you don't own — those emails go to Google / that company.

## Your-domain mode (any name @ any domain you own)

1. Buy/own a domain (Namecheap, Cloudflare, GoDaddy… ~$1–10/yr). Subdomains work too (`work.euron.one` only if you own `euron.one`).
2. Get a VPS with a public IP and **inbound port 25 open** (Hetzner, DigitalOcean, AWS EC2, Oracle Cloud free tier).
3. DNS at your registrar:
   ```
   A    mail.yourdomain.com     -> <your VPS IP>
   MX   yourdomain.com          -> mail.yourdomain.com   (priority 10)
   ```
   For a subdomain: `MX  work.yourdomain.com -> mail.yourdomain.com`.
4. Run on the VPS:
   ```bash
   CUSTOM_DOMAINS=yourdomain.com,work.yourdomain.com SMTP_PORT=25 sudo -E npm start
   ```
   (or keep port 4000 for the web UI behind nginx; port 25 needs root or `setcap cap_net_bind_service`).
5. Check: `dig MX yourdomain.com` shows your server. Create `test@yourdomain.com` in the UI and send it a mail from Gmail.

Any name works instantly — no account creation. Mail to names with no open inbox is rejected (550),
so you don't collect spam. After 10 minutes (or **Delete**) the address stops accepting mail.

## Know before relying on it

- **Many sites block disposable domains** (mail.tm / Guerrilla domains are on public blocklists). Your own domain is far less likely to be blocked.
- Inboxes are in memory: server restart = all inboxes and mail gone. That's intended for a 10-minute tool.
- Use **+10 min** to extend an inbox before it expires.
- No login on the web UI. Don't expose it publicly without adding auth (basic auth / Cloudflare Access), or anyone can open inboxes on your domain.

## Config

| Env | Default | Meaning |
|---|---|---|
| `PORT` | 4000 | Web UI port |
| `INBOX_MINUTES` | 10 | Inbox lifetime |
| `CUSTOM_DOMAINS` | – | Comma-separated domains you own; enables your-domain mode |
| `SMTP_PORT` | 25 | Inbound SMTP port for your-domain mode |

## Test

```bash
npm test   # 11 tests: fake mail.tm/Guerrilla APIs + real SMTP delivery into your-domain mode
```
