const nodemailer = require("nodemailer");

let transporterPromise = null;
let mode = null;

function hasSmtp() {
  return Boolean(process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS);
}

function mailMode() {
  if (hasSmtp()) return "smtp";
  return process.env.MAIL_MODE === "console" ? "console" : "ethereal";
}

async function getTransporter() {
  if (transporterPromise) return transporterPromise;
  mode = mailMode();

  if (mode === "smtp") {
    const port = Number(process.env.SMTP_PORT || 587);
    transporterPromise = Promise.resolve(
      nodemailer.createTransport({
        host: process.env.SMTP_HOST,
        port,
        secure: port === 465,
        auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
      })
    );
  } else if (mode === "ethereal") {
    // Free throwaway test inbox — no real delivery, gives a preview URL.
    transporterPromise = nodemailer.createTestAccount().then((acc) => {
      console.log(`[mail] No SMTP configured. Using Ethereal test inbox: ${acc.user}`);
      return nodemailer.createTransport({
        host: acc.smtp.host,
        port: acc.smtp.port,
        secure: acc.smtp.secure,
        auth: { user: acc.user, pass: acc.pass },
      });
    }).catch((err) => {
      transporterPromise = null;
      throw err;
    });
  } else {
    transporterPromise = Promise.resolve(nodemailer.createTransport({ jsonTransport: true }));
  }
  return transporterPromise;
}

function buildHtml(otp, minutes) {
  return `
  <div style="font-family:Arial,sans-serif;max-width:420px;margin:auto;padding:24px;border:1px solid #e5e7eb;border-radius:12px">
    <h2 style="margin:0 0 12px;color:#111827">Your verification code</h2>
    <p style="color:#4b5563">Use this one-time password to verify your email:</p>
    <div style="font-size:32px;letter-spacing:8px;font-weight:bold;text-align:center;padding:16px;background:#f3f4f6;border-radius:8px;color:#111827">${otp}</div>
    <p style="color:#6b7280;font-size:13px;margin-top:16px">This code expires in ${minutes} minutes. If you didn't request it, ignore this email.</p>
  </div>`;
}

async function sendOtpEmail(to, otp, minutes) {
  const transporter = await getTransporter();
  const from = process.env.MAIL_FROM || process.env.SMTP_USER || "OTP Service <no-reply@example.com>";

  const info = await transporter.sendMail({
    from,
    to,
    subject: `Your OTP is ${otp}`,
    text: `Your OTP is ${otp}. It expires in ${minutes} minutes.`,
    html: buildHtml(otp, minutes),
  });

  if (mode === "console") {
    console.log(`[mail:console] OTP for ${to}: ${otp}`);
    return {};
  }
  if (mode === "ethereal") {
    const previewUrl = nodemailer.getTestMessageUrl(info);
    console.log(`[mail:ethereal] OTP email for ${to}: ${previewUrl}`);
    return { previewUrl };
  }
  return {};
}

module.exports = { sendOtpEmail, mailMode };
