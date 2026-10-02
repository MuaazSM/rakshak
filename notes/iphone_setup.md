# Dad's iPhone: setting up the two Shortcuts

Dad checks messages with two iOS Shortcuts that talk to the hub on the Mac over Tailscale:

- **Rakshak**: in the share sheet. Dad shares a message or a screenshot to it and hears the verdict.
- **Rakshak — Bolo**: on the Home Screen. Dad taps it, describes a phone call out loud (or plays a voice note), and hears the verdict.

Both do the same three things with the answer: show an alert with the verdict word and the first sentence, speak the full explanation aloud, and open the verdict card in Safari so Dad sees the same card Mom sees.

Replace `PUBLIC_BASE_URL` below with the real address, for example `https://muaaz-mbp.tailXXXX.ts.net` (see `notes/android_setup.md` step 1 for how to get it). It must be the `https://` address from `tailscale serve`, with no trailing slash.

Everything below was written from the hub's actual API. Action names are the ones in the Shortcuts app on iOS 17/18; Apple renames things now and then, so if a name differs slightly, pick the closest one. It has not been tested on a physical iPhone yet, so do step 9 before handing the phone over.

## What the hub expects and returns

| Shortcut | Request | Form fields |
|---|---|---|
| Rakshak, text | `POST PUBLIC_BASE_URL/api/share` (multipart form) | `parent_id` = `dad`, `text` = the shared text |
| Rakshak, image | `POST PUBLIC_BASE_URL/api/share` (multipart form) | `parent_id` = `dad`, `image` = the picture (file) |
| Rakshak — Bolo | `POST PUBLIC_BASE_URL/voice` (multipart form) | `parent_id` = `dad`, `audio` = the recording (file, m4a) |

The answer is JSON. The fields the shortcuts use:

- `verdict`: one of `SCAM`, `SUSPICIOUS`, `SAFE`, `UNKNOWN`
- `explanation`: the spoken explanation, in English unless Dad's profile is changed
- `event_id`: used to open `PUBLIC_BASE_URL/v/{event_id}`

Errors (empty input, recording over 60 seconds or 2 MB, picture over 12 MB) come back as JSON with no `verdict` field, so the shortcuts treat "no `verdict`" as "couldn't check".

## 1. Tailscale on the iPhone

1. Install **Tailscale** from the App Store.
2. Open it and sign in with the same account as the Mac. If Dad has his own Tailscale account, share the Mac with it instead: in the Tailscale admin console go to Machines, open the Mac's menu, choose Share, and send Dad's account the invite.
3. Tap the switch to connect. The iPhone shows a VPN icon in the status bar.
4. Keep it on: iPhone Settings, General, VPN & Device Management, VPN, tap the (i) next to Tailscale and turn on **Connect On Demand**.
5. In Safari on the iPhone open `PUBLIC_BASE_URL/health`. You should see a small block of JSON starting with `"status"`. If the page doesn't load, stop here: nothing else will work until it does (see "If the Mac is unreachable" at the end).

## 2. Build "Rakshak" (the share-sheet shortcut)

Open the **Shortcuts** app, tap **+**, and name the shortcut `Rakshak`. Then add these actions in order.

1. **Text**. Type `PUBLIC_BASE_URL`. Tap the action's output and rename it to `Base` (long-press the variable, Rename). This is the only place the address appears, so it is easy to change later.
2. **Show Notification**. Title `Rakshak`, body `Checking this message. If you see an error instead of an answer, call Muaaz.` Turn the sound off. This is the safety net for the "Mac unreachable" case, explained at the end.
3. **Get Images from** `Shortcut Input`.
4. **Count** `Images` (action: *Count Items*; the input is the output of step 3).
5. **If** `Count` **is greater than** `0`.
   1. **Convert Image** to **JPEG**, input `Images` (this also turns iPhone HEIC photos into something the hub can read). Quality: High.
   2. **Get Contents of URL**:
      - URL: `Base` followed by `/api/share`
      - Show More, Method: `POST`
      - Request Body: `Form`
      - Add a **Text** field: name `parent_id`, value `dad`
      - Add a **File** field: name `image`, value = the converted image from step 5.1
   3. **Set Variable** `Answer` to the result.
6. **Otherwise**:
   1. **Get Contents of URL**:
      - URL: `Base` followed by `/api/share`
      - Method `POST`, Request Body `Form`
      - **Text** field: name `parent_id`, value `dad`
      - **Text** field: name `text`, value = `Shortcut Input`
   2. **Set Variable** `Answer` to the result.
7. **End If**.
8. Continue with the shared "show the answer" steps in section 4 below.

Then open the shortcut's settings (the (i) at the bottom or the shortcut name at the top):

- Turn on **Show in Share Sheet**.
- Under **Share Sheet Types**, leave only **Images** and **Text** ticked (untick everything else).
- Set the icon: see section 5.

## 3. Build "Rakshak — Bolo" (voice)

New shortcut named `Rakshak — Bolo`.

1. **Text**: `PUBLIC_BASE_URL`, rename the output `Base`.
2. **Show Notification**: title `Rakshak`, body `Tell me what they said. Tap Stop when you finish.` Sound off.
3. **Record Audio**. Show More: Audio Quality `Normal`, Start Recording `Immediately`, Finish Recording `On Tap`. Tell Dad to keep it under one minute (the hub refuses anything longer than 60 seconds).
4. **Get Contents of URL**:
   - URL: `Base` followed by `/voice`
   - Method `POST`, Request Body `Form`
   - **Text** field: name `parent_id`, value `dad`
   - **File** field: name `audio`, value = `Recorded Audio`
5. **Set Variable** `Answer` to the result.
6. Continue with the shared steps in section 4.

Do not turn on Show in Share Sheet for this one. It lives on the Home Screen.

## 4. The shared "show the answer" steps (end of both shortcuts)

Add these after the request has stored its result in `Answer`.

1. **Get Dictionary from** `Answer`.
2. **Get Dictionary Value**: Get `Value` for `verdict` in `Dictionary`. Rename the output `Verdict`.
3. **If** `Verdict` **has no value**:
   1. **Show Alert**: title `Rakshak couldn't check`, message `Rakshak couldn't check — call Muaaz`, with only an OK button.
   2. **Speak Text**: `Rakshak couldn't check — call Muaaz` Language: English (India).
   3. **Stop This Shortcut**.
4. **End If**.
5. **Get Dictionary Value**: Get `Value` for `explanation` in `Dictionary`. Rename the output `Explanation`.
6. **Get Dictionary Value**: Get `Value` for `event_id` in `Dictionary`. Rename the output `EventID`.
7. **Dictionary** (the action). Add four **Text** items:
   - key `SCAM`, value `SCAM`
   - key `SUSPICIOUS`, value `BE CAREFUL`
   - key `SAFE`, value `LOOKS NORMAL`
   - key `UNKNOWN`, value `COULDN'T CHECK`
8. **Get Dictionary Value**: Get `Value` for `Verdict` in that dictionary (tap the key field and insert the `Verdict` variable). Rename the output `Word`.
9. **Split Text** `Explanation` by **Custom**, separator `.` (a full stop).
10. **Get Item from List**: `First Item`. Rename the output `FirstSentence`.
11. **Show Alert**: title `Word`, message `FirstSentence` followed by a full stop. Hide the Cancel button (Show More, turn off "Show Cancel Button").
12. **Speak Text**: input `Explanation`. Show More: Language `English (India)`, Voice pick one of the English (India) voices (for example Rishi, if it is listed), Rate about 45%, Wait Until Finished on.
13. **Text**: `Base` followed by `/v/` followed by `EventID`.
14. **Show Web Page** (Web category) with that text as the URL. If your iOS version doesn't have it, use **Open URLs** instead (it opens Safari itself).

If an English (India) voice isn't listed in step 12: iPhone Settings, Accessibility, Spoken Content, Voices, English, India, download one, then reopen the action.

If Dad's explanation language is ever changed to Hindi in his profile, the response's `language` field is `hi` and the Speak Text voice would need to be Hindi too. English is the default and what these steps set up.

## 5. Home Screen icons

Do this for both shortcuts.

1. Save the app icon to the iPhone's Photos: AirDrop `brand/rakshak-icon-ink-1024.png` from the Mac (the full-bleed ink icon; iOS adds its own rounded corners).
2. In the shortcut: tap the shortcut name at the top, **Add to Home Screen**.
3. Under Home Screen Name and Icon, tap the small icon, **Choose Photo**, pick the Rakshak icon.
4. Name: `Rakshak` for the share-sheet one (only needed if Dad also wants it on the Home Screen) and `Bolo` for the voice one, or leave `Rakshak — Bolo`.
5. Tap **Add**.

## 6. First run: the network permission

The first time each shortcut makes its request, iOS asks something like "Allow Rakshak to send data to muaaz-mbp.tailXXXX.ts.net?". Tap **Always Allow**. Run each shortcut once yourself before giving Dad the phone, so he never sees this prompt:

1. Run **Rakshak — Bolo**, say a sentence, tap stop, tap **Always Allow** when asked.
2. Share a text from Messages to **Rakshak**, tap **Always Allow**.
3. Share a screenshot from Photos to **Rakshak** (a separate request, but the same permission).

Also tick, if offered, "Allow Microphone" the first time Record Audio runs.

## 7. Sharing by iCloud link

Share the finished shortcuts so they can be reinstalled without rebuilding.

1. In the Shortcuts app, long-press the shortcut, **Share**, **Copy iCloud Link**.
2. Send yourself the link. On a new iPhone, open it and tap **Add Shortcut**.
3. Do this for both shortcuts. The link contains the address and `parent_id=dad` as they were when shared; if `PUBLIC_BASE_URL` ever changes, edit action 1 (the `Base` text) and share a fresh link.
4. After importing, redo section 5 (icons) and section 6 (permissions).

Keep the iCloud links out of any public place: anyone with the link can see the address of the hub. Don't put them in the repo or the post.

## 8. If the Mac is unreachable

If the Mac is asleep, offline, or Tailscale is off on either side, the request can't be sent at all. Shortcuts has no "try again if it failed" action, so what Dad sees is a plain iOS error banner (for example "Could not connect to the server") after the notification from step 2 of each shortcut. That notification is there so he knows what it means: **an error instead of an answer means Rakshak couldn't check, so call Muaaz.**

When the hub is up but the models are down, the hub answers properly with the verdict `UNKNOWN`, and the shortcut shows "COULDN'T CHECK" with "Couldn't check right now. The home computer is offline. Don't open any links, and call Muaaz." and speaks it. Dad never gets a "looks normal" from a check that didn't happen.

On the Mac side: keep it on power with `caffeinate -dimsu &`, keep Ollama and both llama-servers running, and check `PUBLIC_BASE_URL/health` from the iPhone's Safari if Dad reports a problem.

Write down, on the card Dad gets with the phone: "If it shows an error or says it couldn't check, don't tap anything in the message. Call Muaaz."

## 9. Test checklist before handing over

On Dad's iPhone, with Tailscale on:

- [ ] Share the text `Dear customer your account will be blocked today. Share your OTP now.` from Messages to **Rakshak**: alert says SCAM, the voice reads the explanation, Safari opens the red card.
- [ ] Share a screenshot of a bank OTP message from Photos: LOOKS NORMAL (or BE CAREFUL), spoken, card opens.
- [ ] Run **Rakshak — Bolo**, say "someone called saying my account will be blocked, they want my OTP", tap Stop: a verdict is spoken.
- [ ] Switch Tailscale off on the iPhone and run **Rakshak** again: an error banner appears (this is the "couldn't check" case). Switch Tailscale back on.
- [ ] Muaaz's phone buzzes (ntfy) for each SCAM, naming Dad.
