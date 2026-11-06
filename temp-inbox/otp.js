// Pull the most likely OTP / verification code out of an email.
const KEYWORDS = /(otp|code|passcode|password|pin|verification|verify|one[- ]time|security|token|confirm)/i;

function extractOtp(...parts) {
  const text = parts.filter(Boolean).join(" \n ").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ");

  // 1) A code right after a keyword: "Your OTP is 482913", "code: AB12CD"
  const near = text.match(
    /(?:otp|code|passcode|pin|password)(?:\s+(?:is|was))?[^A-Za-z0-9]{0,25}([A-Z0-9]{4,8}|\d{3}[\s-]\d{3})\b/i
  );
  if (near && /\d/.test(near[1])) return near[1].replace(/[\s-]/g, "");

  // 2) Any standalone 4–8 digit number in a message that mentions a keyword
  if (KEYWORDS.test(text)) {
    const nums = [...text.matchAll(/(?<![\d.,:/-])(\d{4,8})(?!\d|[.,:/-]\d)/g)].map((m) => m[1]);
    const likely = nums.find((n) => !/^(19|20)\d{2}$/.test(n)) || nums[0]; // skip years when possible
    if (likely) return likely;
  }
  return null;
}

module.exports = { extractOtp };
