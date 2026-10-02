# Mom's Android phone: setting it up

Mom uses Rakshak as an installed web app on her Android phone (Chrome). It appears in the share sheet, so she can share a message or screenshot from WhatsApp or Messages straight to it, or open it and hold a button to describe a call.

The hub runs on the MacBook and listens only on `127.0.0.1`. The phone reaches it through Tailscale and `tailscale serve`, which gives it an `https://` address inside the private tailnet. Nothing is exposed to the public internet.

## 1. Put the hub on the tailnet with `tailscale serve`

On the Mac:

1. Install Tailscale (Mac App Store or `brew install tailscale`) and sign in. Make sure the Mac shows as connected.
2. In the Tailscale admin console (login.tailscale.com), open **DNS** and check that **MagicDNS** is on and **HTTPS Certificates** is enabled. Without HTTPS the app can't be installed and the share sheet entry won't appear.
3. Start the hub (it binds to `127.0.0.1:8000`):

   ```bash
   cd ~/rakshak
   uv run uvicorn hub.app:app --host 127.0.0.1 --port 8000
   ```

4. In another terminal, publish it on the tailnet:

   ```bash
   tailscale serve --bg 8000
   ```

   This is the syntax for current Tailscale releases (1.52 and later): `--bg` keeps it running in the background and `8000` proxies `https://<mac-name>.<tailnet>.ts.net` on port 443 to `http://127.0.0.1:8000`. On older releases the command was `tailscale serve https / http://127.0.0.1:8000`. The Tailscale command line isn't installed on the machine where this was written, so this has not been run here; check it against `tailscale serve --help` on the Mac, and if the flags differ, follow that help text.

5. Check it and read the address:

   ```bash
   tailscale serve status
   ```

   It prints the URL, for example `https://muaaz-mbp.tailXXXX.ts.net`. If the `tailscale` command isn't found when installed from the Mac App Store, use `/Applications/Tailscale.app/Contents/MacOS/Tailscale` in its place.

6. Do not use `tailscale funnel`. That would put the hub on the public internet.

To stop sharing later: `tailscale serve reset`.

## 2. Set PUBLIC_BASE_URL

Add the address from step 1 to `.env` in the repo (no trailing slash):

```
PUBLIC_BASE_URL=https://muaaz-mbp.tailXXXX.ts.net
```

`.env` is git-ignored; never commit it. The hub reads this value as a setting and the notes and shortcuts use the same address. Restart the hub after editing `.env`.

If you want the "Call Muaaz" button on the verdict screen, put the number in `pwa/.env.local` as `VITE_SON_PHONE=...` (also git-ignored) and rebuild the app:

```bash
cd pwa && npm run build
```

The hub serves the built app from `pwa/dist`, so rebuild whenever the app changes, then reload it on the phone.

Quick check from the Mac (replace the address):

```bash
curl -s https://muaaz-mbp.tailXXXX.ts.net/health
curl -s -X POST https://muaaz-mbp.tailXXXX.ts.net/api/check \
  -H 'content-type: application/json' \
  -d '{"parent_id":"mom","text":"Dear customer your account will be blocked today. Share your OTP now.","channel":"sms"}'
```

The second one should come back as JSON with `"verdict":"SCAM"`.

Keep the Mac on power with the lid awake while the family is using it: `caffeinate -dimsu &`.

## 3. Tailscale on Mom's phone

1. Install **Tailscale** from the Play Store.
2. Sign in with the same account as the Mac. If Mom has a different Google account, share the Mac with it from the admin console (Machines, the Mac's menu, Share) and accept the invite on her phone.
3. Switch it on. A key icon shows in the status bar while the VPN is connected.
4. Keep it connected: Android Settings, Network and internet, VPN, tap the gear beside Tailscale and turn on **Always-on VPN**. Also set Tailscale's battery use to **Unrestricted** (Settings, Apps, Tailscale, Battery) so Android doesn't stop it in the background.
5. In Chrome on the phone open `PUBLIC_BASE_URL/health`. It should show a small block of JSON. If it doesn't load, fix this before going on.

## 4. Install the app from Chrome

1. In Chrome on Mom's phone open:

   ```
   PUBLIC_BASE_URL/?parent=mom
   ```

   The `?parent=mom` part is how the app learns who is using it. It is remembered on the phone, so Mom never types it again.

2. Tap the three-dot menu, then **Add to Home screen** (on some versions **Install app**). Accept the prompt. Installing as an app (rather than a plain shortcut) is what makes Rakshak show up in the share sheet.
3. Open Rakshak from the new Home screen icon once. You should see the Rakshak start screen, not the browser address bar.
4. Allow the microphone when asked the first time she uses "Hold to talk". If she says no, Chrome's site settings (lock icon, Permissions) can turn it back on.

## 5. Check that Rakshak is in the share sheet

Install first (step 4); the share sheet only learns about the app after it is installed.

1. **WhatsApp.** Long-press a message, tap the three-dot menu and choose **Share** (on some versions under More). In the share sheet scroll the app row, or tap **More**, and look for **Rakshak**. Tap it. The "Checking" screen appears, then the verdict card.
2. **Messages (Google Messages).** Long-press a message, three-dot menu, **Share**, then **Rakshak**.
3. **A screenshot.** In Photos or Gallery open a screenshot, tap **Share**, then **Rakshak**.

If Rakshak isn't listed: remove the app from the Home screen, clear Chrome's data for the hub's address, reopen `PUBLIC_BASE_URL/?parent=mom` and install it again. Check that the address starts with `https://`. After a fresh install, give the share sheet a minute, or restart the phone.

Tip: ask Android to pin Rakshak at the top of the share sheet (long-press it in the share sheet, **Pin**) so Mom always finds it in the same place.

## 6. Test text, screenshot and voice

Do all three on Mom's phone, with Tailscale connected:

1. **Text.** Share `Dear customer your account will be blocked today. Share your OTP now.` from Messages. Expect the red SCAM card, a spoken explanation, the quote highlighted, and (if you added the number) a **Call Muaaz** button.
2. **Screenshot.** Share a screenshot of a bank OTP message. Expect a green or amber card, spoken. Take another screenshot of an obvious scam message to see red.
3. **Voice.** Open Rakshak, tap **Hold to talk**, hold the button and say "Someone called saying my account will be blocked and asked for my OTP", then let go. A verdict card appears within about 12 seconds.
4. **Alert.** Each SCAM should buzz Muaaz's phone through ntfy ("Mom got a likely SCAM ... Call them."). Make sure the ntfy app on Muaaz's phone is subscribed to the topic in `NTFY_TOPIC`.
5. **Hub down.** Stop the hub (or switch Tailscale off on the phone) and share a message. The app must say it couldn't check; it must never say the message looks normal.

## 7. Check the spoken voice (en-IN)

The app speaks with the phone's own text-to-speech. It asks for an English (India) voice. If the phone has none, the card still shows the text, but nothing is spoken.

1. Android Settings, search for **Text-to-speech output** (usually under System, Languages, or Accessibility).
2. Preferred engine: **Speech Recognition & Synthesis from Google** (Google Text-to-speech).
3. Tap the gear beside it, **Install voice data**, choose **English (India)** and download a voice.
4. Back in Text-to-speech output set **Language** to **English (India)** and tap **Listen to an example**. If you hear it, Chrome can use it.
5. Raise the media volume, since the spoken verdict plays on the media channel, not the ringer.

Then, in the app, open any verdict and tap **Listen again**. It should speak with an Indian English voice.
