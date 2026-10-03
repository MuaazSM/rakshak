# Failure cases — PROVISIONAL (synthetic dev)

> **PROVISIONAL (synthetic dev).** Everything below is measured on `data/splits/dev.jsonl`
> (166 items, all synthetic, sha256 `9d387b98…`). There is no real test set yet. The examples
> quoted here are synthetic dev items (no real messages). These notes will be redone after the
> retrain on real data (PRD §12.5 asks for three documented failure cases with fixes).

Sources: cached per-item Tinker dev predictions (`training/runs/20261003-004911-full-*/tuned_tinker_e*_dev.items.jsonl`,
greedy, same §8.2 prompt as inference), summarized in `eval/results/runs_summary_dev.json`
(`uv run python -m eval.error_analysis`).

## Why this dev set is saturated

The chosen checkpoint scores macro-F1 1.000, recall 1.000, FPR 0. That says more about the dev
set than the model:

- **One seed group per stratum.** Dev has 21 seed groups for 166 items: one seed per scam
  category, two for `other_scam`, one per safe category, plus two call-description groups. Each
  group holds ~8 paraphrases of a single seed (Gemma variants across en / Hinglish / Devanagari,
  obfuscation, channel). The effective sample size is closer to **21 decisions than 166**, and
  the bootstrap CIs (which resample items, not groups) are too narrow.
- **Gemma-generated phrasing with label-correlated artifacts.** Counts by gold verdict:

  | Feature | train SAFE | train SCAM | dev SAFE | dev SCAM |
  |---|---|---|---|---|
  | any link | 0/255 | 218/428 | 0/55 | 55/96 |
  | appended "Case ID" / "Txn ID" | 0/255 | 74/428 | 0/55 | 22/96 |
  | emoji | 0/255 | 53/428 | 1/55 | 13/96 |
  | "ji" / "जी" greeting | 64/255 | 341/428 | 10/55 | 78/96 |

  No genuine message contains a link, yet real bank, delivery and govt SMS almost always do
  (`amzn.in/…`, `sbi.co.in`, `https://…gov.in`). A model can reach 100% here by learning "link ⇒
  scam", which is exactly the false-positive mode NFR-4 forbids.
- **SUSPICIOUS = 2 seed groups.** All 15 gold SUSPICIOUS items come from `seed-043` (8:
  "pre-approved loan / investment chance, reply YES") and `calls-c9` (7: "caller from an
  insurance company mentioned a special scheme"). The SUSPICIOUS class F1, and with it macro-F1,
  flips on two decisions.

Same-family baselines are close to ceiling too (base Qwen few-shot 0.908 macro-F1 on the
earlier 152-item dev build), so the "+10 macro-F1 over base" criterion (§12.5) can't be judged
on this dev set.

## Confusion matrix — chosen checkpoint (`full-lr1x`, epoch 3)

Verdict (rows gold, columns predicted):

| gold \ pred | SCAM | SUSPICIOUS | SAFE |
|---|---|---|---|
| SCAM (96) | **96** | 0 | 0 |
| SUSPICIOUS (15) | 0 | **15** | 0 |
| SAFE (55) | 0 | 0 | **55** |

Category on true scams: 78/96 correct (0.812). The errors (gold → predicted, count, seed group):

| Gold | Predicted | n | Seed group |
|---|---|---|---|
| other_scam | digital_arrest | 5 | seed-038 |
| other_scam | kyc_account_block | 3 | seed-038 |
| courier_parcel | digital_arrest | 3 | seed-018 |
| courier_parcel | (off-vocabulary → JSON invalid) | 1 | seed-018 |
| lottery_prize | kyc_account_block | 2 | seed-031 |
| lottery_prize | otp_phishing | 1 | seed-031 |
| family_emergency | upi_collect_refund / digital_arrest | 1 + 1 | seed-022 |
| upi_collect_refund | kyc_account_block (+ JSON invalid) | 1 | seed-032 |

Other near-misses at the chosen epoch: JSON validity 0.988 (2/166 replies failed strict §8.2
validation but still parsed, so the verdict was usable); grounding 0.993 (one item,
`syn-seed-023-s1-v2`, had 2 of 3 quotes not exact substrings: paraphrased instead of copied);
span F1 0.669.

## Errors grouped by pattern

### 1. SUSPICIOUS ↔ SAFE flips on whole seed groups (instability)

Gold-SUSPICIOUS predictions per seed group, all runs and epochs:

| Run | Epoch | calls-c9 (7) | seed-043 (8) | Macro-F1 |
|---|---|---|---|---|
| lr05x | 1 | 7 SUS | 6 SUS, 2 SCAM | 0.946 |
| lr05x | 2 ★ | 4 SUS, 3 SAFE | 8 SUS | 0.954 |
| lr05x | 3 | 1 SUS, 6 SAFE | 7 SUS, 1 SAFE | 0.879 |
| lr1x | 1 | 7 SUS | 8 SUS | 0.973 |
| lr1x | 2 | **7 SAFE** | 7 SUS, 1 SAFE | 0.856 |
| lr1x | 3 ★ | 7 SUS | 8 SUS | 1.000 |
| lr2x | 1 | 7 SAFE | 3 SUS, 5 SAFE | 0.741 |
| lr2x | 2 ★ | 7 SUS | 8 SUS | 0.976 |
| lr2x | 3 | **7 SAFE** | 8 SUS | 0.879 |

The whole 0.86 ↔ 1.00 macro-F1 swing is one seed group (`calls-c9`, e.g. *"Caller said he was
from an insurance company and mentioned a special scheme, suggesting the person call back on a
specific number if interested."*) moving as a block between SUSPICIOUS and SAFE. The item has
no ask, no link and no threat, so it sits on the SAFE/SUSPICIOUS boundary. Train has only 9
SUSPICIOUS call descriptions (`calls-c8`, `calls-c10`). **Epoch selection is therefore close to
a coin flip on this dev set.** The chosen "best" epoch (lr1x e3) won on this one group, not on a
general improvement. lr1x e3 is still a sensible pick (lowest loss, highest JSON validity of
lr1x, recall 1.0, FPR 0), but the 1.000 must not be reported as a result.

### 2. Category confusion between neighbouring scam types (label overlap)

All 18 category errors at the chosen epoch fall in 5 seed groups, and the model's answers are
mostly defensible:

- `seed-018` courier_parcel → digital_arrest: *"aapke Amazon order Nagpur mein customs mein
  block ho gaya hai… turant Rs 7800 payment karein"*. Customs + "block" + payment is also the
  digital-arrest opening (parcel with drugs, customs officer). The §10.1 schema has no rule for
  which category wins when both apply.
- `seed-038` other_scam → kyc_account_block / digital_arrest: *"your mobile connection will be
  suspended in 2 hours because of KYC mismatch. Please dial *9810022334#"*. This is a KYC
  suspension scam. The gold `other_scam` label looks wrong, not the model.
- `seed-031` lottery_prize → kyc_account_block / otp_phishing: *"₹199 has been credited to your
  account… click this link immediately to verify your identity"*. Nothing here is a lottery
  or prize. Again the gold label is the weak side.
- `seed-022` family_emergency → upi_collect_refund / digital_arrest (2 items, both obfuscated).

Category doesn't change the verdict, so the parent-facing alert is unaffected. But the category
appears in the ntfy alert and the explanation, so a wrong category reads as a wrong warning.

### 3. Genuine messages with money words → false SUSPICIOUS / SCAM (earlier epochs)

The chosen epoch has no false positives. Earlier epochs and the other runs show the pattern NFR-4
cares about:

- `seed-060` personal (WhatsApp from a contact): *"the bank transfer for the rent has been done.
  Amount around 36000 rupees. Check your account statement."* → SCAM family_emergency (lr2x e2)
  and SUSPICIOUS (lr1x e1, lr2x e2).
- `seed-052` delivery_update: *"aapka parcel… aaj delivery ke liye ready hai… Agar 24 ghante mein
  parcel collect nahi hua, toh parcel return kar diya jayega"* → SCAM courier_parcel (lr2x e2).
  A genuine deadline was read as scam urgency.
- `seed-055` legit_promo screenshot (*"विशेष ऑफर! … ₹2999 तक की बचत"*) and `calls-c6` gas-cylinder
  delivery call → SUSPICIOUS (e1 of lr05x / lr1x).
- One miss (lr2x e1): `seed-015` electricity_disconnect from a +91 number → SAFE govt_genuine.

The amounts ₹36000 / ₹7800 / ₹1500 / ₹2999 recur across both scam and genuine seeds (the
generator reuses them), so amount alone is not the cue. "Bank transfer" + amount + "check your
account" overlaps with KYC-scam wording.

### 4. JSON / grounding near-misses

- 2/166 replies at the chosen epoch failed strict §8.2 validation. One had an off-vocabulary
  `category` (`syn-seed-018-s4-v1`). The other (`syn-seed-032-s1-v3`) had a valid category, so
  most likely an off-vocabulary `reason`. Both still parsed, so the verdict was usable. In the
  hub, FR-13's constrained retry repairs these.
- Span F1 tops out at ~0.67 in every run. Part of this is gold noise from the synthetic labels:
  quotes such as `"Singh ji"` → `too_good_to_be_true`, `"mobile aacount"` → `urgency_deadline`
  and `"emergency"` → `secrecy_request` are not real red flags, and the model (correctly) doesn't
  reproduce them.

## Three data fixes for the retrain with real data

1. **Break the link / ID / emoji shortcuts with real hard negatives.** Add real genuine
   messages from both phones and the own inbox (§10.2: 100–200) that carry official links
   (`amzn.in`, `sbi.co.in`, `*.gov.in`, `bit.ly` used by real brands), tracking or transaction
   IDs, and amounts with deadlines ("collect within 24 hours", "pay by 5 Oct to avoid late
   fee"). Strip the generator's appended `Case ID` / `Txn ID` suffixes from synthetic scams, or
   add them to genuine items at the same rate. Target: in train, the share of SAFE items with a
   link is ≥ 30% and no surface feature is > 2× more frequent in one verdict than the other.
2. **Make SUSPICIOUS a real class, split by seed group.** Write ≥ 8 distinct SUSPICIOUS seeds
   (unknown number "is this your number?", loan pre-approval with no ask, insurance / scheme
   calls, "wrong number" chats) and keep SUSPICIOUS < 10% of data (§10.1). Put ≥ 3 seed groups
   in dev and in train, and stratify dev by seed group, not item. Report SUSPICIOUS-class F1 and
   a seed-group bootstrap alongside item-level CIs, so one group can't move macro-F1 by 0.14.
3. **Fix category precedence and relabel the ambiguous seeds.** Add a precedence rule to
   §10.1 (e.g. threat of arrest / police / customs officer ⇒ `digital_arrest`; account, SIM
   or KYC suspension ⇒ `kyc_account_block`; money "credited" + verify link ⇒
   `kyc_account_block`, not `lottery_prize`), relabel `seed-018`, `seed-031` and `seed-038`
   variants under it, and hand-check gold red-flag quotes so every quote is a real cue with a
   fitting reason. This should lift category accuracy and span F1 without touching verdicts.
