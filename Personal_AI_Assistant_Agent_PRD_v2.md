# PRD: Personal AI Assistant Agent (Backend/Agent-First Build)

**Author:** [Your Name]
**Status:** Draft v2
**Last updated:** July 19, 2026

> **Phase framing (applies to the whole document):** v1 delivers the agent and backend only — reasoning engine, knowledge base, tool integrations, and safety layers — exposed through a clean, documented HTTP API. No frontend/UI is built in this phase. The system is developed and validated via direct API calls (CLI script, curl, or Postman), and architected so a chat widget can be integrated later with configuration changes only, no backend rework. Where this document describes "visitor" behavior, it is validated by the owner and trusted testers calling the API in place of real website visitors.

---

## 1. Overview

This product is a conversational AI agent that acts as a digital stand-in for the site owner, intended to eventually be embedded on a personal website. It answers questions about the owner from a curated knowledge base, facilitates call bookings against the owner's real calendar, and refuses everything else.

The primary purpose is to serve as a portfolio piece demonstrating applied AI engineering skill — specifically safe, thoughtful agent design (tool scoping, prompt-injection defense, RAG, orchestration) rather than a general-purpose chatbot. Deferring frontend work concentrates effort on the part of the project with the most engineering depth, and keeps the system modular and easier to test.

**Why now:** LLM APIs, tool-use capabilities, and lightweight hosting have matured to the point where one person can build and deploy a sophisticated, safe agent without dedicated infrastructure — making this both feasible and an increasingly expected form of technical self-presentation.

---

## 2. Goals & Success Metrics

**What success looks like:**
- A fully functional agent (backend + API) that reliably answers questions about the owner, resists misuse, and completes end-to-end bookings against a real calendar — all validated via API calls.
- A stable, documented API contract that a future frontend can integrate against without backend changes.
- A credible, explainable portfolio artifact: every design decision (safety, architecture, scoping) can be articulated in an interview or review.

**Business goals:**
- Build the reusable technical core of the owner's eventual website assistant, decoupled from any frontend.
- Serve as a referenceable project in the owner's job search, even before a frontend exists.

**KPIs and targets** *(targets are initial placeholders — tune after the first test pass)*:

| Metric | Target (v1) | Measured via |
|---|---|---|
| Groundedness on "about me" test set | ≥ 95% accurate, 0 fabricated claims | Curated Q&A test set, manually graded |
| Off-topic / misuse block rate | ≥ 95% blocked; 0 full jailbreaks to sustained off-topic use | Scripted adversarial prompt suite, re-run each release |
| End-to-end booking completion | ≥ 90% of valid test scenarios complete without manual intervention | Scripted booking scenarios |
| API latency | p95 ≤ 6s per non-streaming turn | Request logs |
| Security incidents | 0 PII leaks; 0 unauthorized calendar actions; system prompt never disclosed | Red-team pass + log review |

**Instrumentation (v1 requirement, not future work):**
- Structured logging of every conversation turn and tool call.
- Classifier-sourced tagging of each turn as on-topic / off-topic / refused.
- Booking funnel tracking: initiated → slots proposed → slot selected → confirmed → created.
- A periodic query script (or minimal dashboard) to review logs; a full analytics UI is a Could-have.

Because the API contract and schemas are stable, these same metrics carry over unchanged to real visitor traffic once a frontend exists.

---

## 3. Target Users / Personas

**Primary — Recruiter / Hiring Manager.** Evaluating the owner as a candidate. Wants quick answers about experience, skills, and availability for a call. Today they must read a static resume and separately email or use a scheduling link — disjointed and slow.

**Secondary — Technical Peer / Collaborator.** Exploring the owner's technical background or a potential collaboration. Often meta-curious about how the agent was built, and likely to probe it with technical or adversarial questions.

**Tertiary — Casual Visitor.** Arrived via LinkedIn/GitHub/social. Low intent, but useful top-of-funnel awareness.

**Shared needs and pain points:**
- Fast, specific answers without reading a full resume/portfolio.
- Low-friction scheduling without email back-and-forth.
- The current workaround (static site + Calendly-style link) is passive and can't answer questions.

---

## 4. Problem Statement

Static personal websites are passive: visitors must self-serve through pages of content, and scheduling requires leaving the site for a separate tool. There is no mechanism for a visitor to ask a direct question ("Does this person have experience with X?") and get an immediate, accurate answer. This friction likely causes some interested visitors — particularly time-constrained recruiters — to disengage before reaching out. No first-party usage data exists yet (pre-launch), so baseline metrics will be established through the instrumentation above rather than backward-looking data.

---

## 5. Scope

**In scope (v1 — Agent & Backend):**
- Backend agent service exposed via a documented HTTP API, callable via scripts, curl/Postman, or a minimal CLI.
- Scripted, deterministic conversation opener (AI self-identification, capability summary, soft name/LinkedIn ask) returned as the first message of a new session.
- RAG-based Q&A grounded in a curated "about me" knowledge base.
- Booking tool integrated with the owner's Google Calendar (availability check + tentative booking with confirmation), driven by an explicit state machine (Section 6).
- Session-based conversation memory via a caller-supplied `session_id`.
- Safety layer: on-topic classifier, prompt-injection defenses, rate limiting, PII protection.
- Logging of all conversations, tool calls, and bookings to a database.
- **Frontend compatibility (not frontend build):** stable JSON request/response schemas, CORS configurable per origin, client-generated session IDs, and a contract that can accommodate streaming (SSE) additively later.
- Deployment of the backend as a hosted HTTPS API.

**Out of scope (v1):**
- Any frontend/UI work — chat widget, embedding, wireframes, visual design. (Future phase; see Section 10.)
- Cross-session memory, visitor accounts, or login.
- Multi-language support / localization.
- Voice interface.
- Irreversible agent actions without confirmation (e.g., auto-sending email on the owner's behalf).
- General-purpose task assistance (coding help, writing, unrelated Q&A) — explicitly blocked, and not a future roadmap item.
- Canceling/rescheduling existing bookings via the agent (managed directly in the calendar in v1).
- Admin dashboard UI (direct DB/log inspection is acceptable in v1).

---

## 6. Features & Requirements

### Functional requirements

**Must-have (M):**
- Answer questions about the owner using only the curated knowledge base (RAG), never open-ended general knowledge.
- Refuse and redirect off-topic requests (coding help, unrelated tasks), including under adversarial rephrasing.
- Never expose the system prompt, internal instructions, or other sessions' data.
- Return the scripted introduction as the first response of any new session, including a soft, non-blocking ask for name and LinkedIn.
- Support the full booking flow: availability check → slot proposal → contact collection → confirmed tentative booking.
- Require confirmation before any booking is finalized.
- Enforce per-session and per-IP (or per-API-key during direct testing) rate limits on messages and booking attempts.
- Log all conversations and tool calls.
- Expose a stable, documented JSON contract (endpoints, schemas, `session_id` handling) — see Section 8 for the proposed API surface.
- Structure CORS and auth so enabling a frontend origin later is a config change, not a code change.

**Should-have (S):**
- On ambiguous or low-confidence questions, offer to forward the message to the owner by email rather than guessing.
- Notify the owner when a booking is created or a conversation is flagged as suspicious.

**Could-have (C):**
- Streaming responses (SSE).
- Simple analytics view of common questions and booking patterns.
- Resume/brief upload for context in hiring conversations.

**Won't-have (this version):**
- Any frontend/UI build.
- Persistent visitor accounts.
- Voice or multi-modal interaction (beyond optional file upload).
- Tasks unrelated to the owner or scheduling, under any framing.

### User stories

- As a **recruiter**, I want to ask whether the owner has experience with a specific technology, so I can assess fit without reading a full resume.
- As a **visitor**, I want to book a call directly in the chat, so I don't have to leave for a separate tool.
- As a **visitor**, I want to optionally share my name and LinkedIn, so the owner can follow up.
- As a **visitor**, I want to know I'm talking to an AI agent, so I have accurate expectations.
- As the **owner**, I want the agent to refuse general-purpose use, so my API costs aren't abused.
- As the **owner**, I want to be notified of bookings and flagged conversations, so I stay informed without manual monitoring.
- As a **future frontend developer**, I want a clearly documented API contract, so I can build a widget without modifying the backend.

### Booking workflow requirements

The booking sub-flow is the highest-friction part of the agent and is designed explicitly rather than left to free-form LLM judgment.

**Must-have (M):**
- Model booking as an explicit **state machine** per session: `intent_detected → slots_proposed → slot_selected → contact_info_collected → confirmed → booking_created`. The orchestration layer persists this state in the DB alongside conversation history, keeping agent behavior at each step narrow and predictable.
- On booking intent, **proactively propose 3–5 concrete open slots** from real availability — never an open-ended "when are you free?"
- Support **natural-language availability requests** ("sometime next week afternoon"), resolved by querying free/busy for that window and returning matching structured slots.
- Return **structured slot data** (ISO timestamps + human-readable labels), so slot selection is matched deterministically rather than inferred from free text.
- **Capture the caller's timezone before proposing slots** (explicit API parameter in this phase; inferred from browser locale once a frontend exists). Timezone is stored and passed through the whole flow so all displays and the final event are consistent.
- **Re-check availability at confirmation time**, immediately before event creation, to guard against race conditions. If the slot is gone, gracefully re-offer alternatives.
- Present an **explicit confirmation summary** (date, time, timezone, name) and create the event only after affirmative confirmation.
- **Cap the negotiation loop at 2 rounds** of slot proposals; after that, widen the search window once, then offer direct email follow-up rather than looping.
- Rate-limit booking *attempts* separately from chat messages (booking abuse is costlier than Q&A abuse).
- Validate email format before finalizing a booking.

**Should-have (S):**
- A **short server-side soft-hold** (5–10 minutes) on a selected slot, with expiry checked on each interaction so an abandoned flow doesn't block a slot. (Pending confirmation that Google Calendar supports this cleanly — see Open Questions.)
- Return a **structured booking confirmation payload** (not just prose), including next steps for rescheduling/canceling outside the agent.
- Lightweight **email verification** before finalizing (see Open Questions — adds friction).

**Could-have (C):**
- Batch candidate windows across multiple days when the caller's stated preference doesn't match any single day well.

---

## 7. Interaction Contract (in place of UX/Design)

No visual UI is designed in this phase. This section defines the interaction contract the API must support, so the experience is consistent regardless of which frontend eventually consumes it.

**Core conversation flow:**

1. Caller starts a new session (generates and sends a `session_id`) → API returns the hardcoded introduction (AI self-identification, capability summary, soft name/LinkedIn ask).
2. Caller sends a free-form question or proceeds directly to booking — the intro ask is never a gate.
3. Agent answers with RAG-grounded responses, returned as text (optionally with citations) in the JSON response.
4. On booking intent, the agent enters the state machine flow (Section 6): propose slots → collect name/email → re-confirm availability → confirmation summary → create tentative booking → return structured confirmation payload.
5. Out-of-scope requests get a brief, friendly refusal and redirect — returned as a normal chat response, not an error code, so a future frontend renders it inline like any message.

**API design implications:**
- Every response uses one consistent JSON shape with a `type` field (e.g., `message`, `refusal`, `booking_proposal`, `booking_confirmation_request`, `booking_confirmed`), so a frontend needs no special-case parsing.
- Booking proposals carry structured slot data so a future UI can render selectable options.
- The booking state machine is exposed implicitly through `type` and response content — a frontend never needs its own parallel state tracking.
- The response format must be streaming-compatible in principle, even though streaming is not built in v1.

---

## 8. Technical Considerations

**Architecture:** API caller (test script / future frontend) → Backend API (FastAPI) → LLM API → Tools (RAG retrieval, Google Calendar) + Database.

**Stack:**
- **Backend:** Python + FastAPI, custom orchestration loop (no heavy agent framework — chosen deliberately for transparency and learning value).
- **LLM:** Anthropic Claude API or OpenAI API with tool use (see Open Questions).
- **RAG:** lightweight embedding model + SQLite-based or in-memory vector store.
- **Calendar:** Google Calendar API, OAuth scoped to the owner's account only (free/busy read + tentative event write).
- **Database:** SQLite (Postgres if needed) for conversation logs, booking records, booking state, rate-limit counters.
- **Testing harness:** simple CLI script or Postman/curl collection against the API.
- **Hosting:** Render, Railway, or Fly.io (free-tier friendly, GitHub-based deploy).

**Proposed API surface (v1):**

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat` | Main conversation endpoint. Request: `{session_id, message, timezone?}`. Response: `{reply, type, data?}` where `data` carries structured payloads (slots, confirmation details) when applicable. |
| `GET /health` | Liveness/readiness check for hosting and monitoring. |

- Version the API path (`/v1/`) from day one so future breaking changes don't strand a deployed frontend.
- Errors return a consistent shape (`{error: {code, message}}`) with appropriate HTTP status codes; in-scope refusals are *not* errors (they return `type: "refusal"` with HTTP 200).
- If streaming is added later, it is additive (separate SSE endpoint or query param), never a breaking change to the synchronous contract.

**Session handling:** caller-driven — the client generates and persists a `session_id` (e.g., a `localStorage` UUID in a future browser frontend); the backend stays stateless per-request beyond that lookup.

**Data requirements:**
- Curated "about me" content (bio, project descriptions, FAQ), authored and maintained by the owner.
- Visitor-provided data limited to: name (optional), LinkedIn URL (optional), email (only if booking).
- Booking state (current step, proposed slots, selected slot, timezone, contact info) persisted per session so the flow survives across API calls.

**Security, performance, and scale:**
- API keys/secrets in environment variables only; never exposed to any client. All traffic over HTTPS.
- CORS restricted to explicitly allowed origins, configurable via environment variable (none needed yet; mechanism in place).
- **Layered misuse defense**, in priority order: (1) narrowly scoped tools — read-only knowledge base, minimal calendar permissions — as the primary structural defense; (2) prompt-level scoping instructions; (3) an input classifier for off-topic/adversarial detection; (4) rate limiting and token/cost caps.
- Expected load is low-to-moderate (personal-site scale); the architecture should tolerate moderate growth without rework, but no enterprise-scale requirements apply.

---

## 9. Non-Functional Requirements

- **Reliability:** Degrade gracefully with a clear error response if the LLM or calendar API is unavailable — never fail silently. Target p95 latency ≤ 6s per non-streaming turn.
- **Security:** No caller input can escalate privileges, alter the system prompt's effect, or reach tools/data beyond the defined scope. Verified via the adversarial test suite before each release.
- **Privacy/compliance:** Document what data is collected and why (name/LinkedIn/email); no PII is required for core Q&A. Define a conversation-log retention period and a process for the owner to delete visitor data on request. This documentation becomes the visible privacy notice once a frontend exists.
- **Accessibility:** Deferred to the frontend phase (not applicable to an API-only deliverable).
- **Localization:** English only in v1.

---

## 10. Timeline / Milestones

*(Solo project; phases are sequential build milestones. No hard external dates; pacing is self-directed.)*

| Phase | Milestone | Exit criteria |
|---|---|---|
| 1 | Basic `/v1/chat` endpoint calling the LLM, no tools | Request/response loop works via curl/script |
| 2 | System prompt with scoping + intro handling | Persona and refusal behavior hold in manual testing |
| 3 | RAG knowledge base built and integrated | Groundedness test set passes target |
| 4 | Booking tool + Google Calendar integration | Full state-machine flow completes end-to-end, incl. confirmation |
| 5 | Safety layer: classifier, rate limiting, PII protection, logging | Adversarial suite passes block-rate target; red-teaming begins |
| 6 | API contract finalized and documented | Schemas, CORS config, and error shapes documented; a stranger could integrate from docs alone |
| 7 | Backend deployed (hosted HTTPS API) + end-to-end testing | All KPIs in Section 2 measured; injection-testing pass complete |
| 8 | Agent phase "done" | Launch checklist passed; frontend integration tracked as a separate future PRD |

---

## 11. Risks & Assumptions

**Risks:**

| Risk | Mitigation |
|---|---|
| Visitor jailbreaks the agent into off-topic use (cost inflation, embarrassment) | Layered defenses (Section 8); ongoing red-teaming; token/cost caps |
| Agent hallucinates inaccurate claims about the owner | Strict RAG grounding; instructed to say "I'm not sure" and offer direct-contact escalation rather than guess |
| Spam/bot-driven fake bookings clutter the calendar | Separate booking rate limits; confirmation step; possible CAPTCHA (Open Question) |
| Concurrent sessions double-book a slot | Availability re-check at confirmation time; optional soft-hold (Section 6) |
| LLM API costs scale unexpectedly | Token/session caps; usage monitoring and spike alerts |
| Visitor PII mishandling creates privacy/legal exposure | Minimal collection; clear notice; defined retention policy |
| Google OAuth token expires or is revoked, silently breaking bookings | Use refresh tokens with monitored renewal; health check surfaces calendar-auth failures; graceful error to callers |

**Assumptions:**
- The owner maintains the knowledge base over time; stale content is a content-ops risk, not a system risk.
- Traffic remains at personal-site scale.
- The owner has (or will set up) a Google account/calendar suitable for OAuth integration.

---

## 12. Stakeholders & Ownership

Solo portfolio project; the owner holds all active roles:

| Role | Owner |
|---|---|
| Product/PM | [Your Name] |
| Engineering (backend, agent orchestration) | [Your Name] |
| Security / red-teaming review | [Your Name] |
| Frontend/UI, Design | Deferred — not active this phase |

**Approval process:** Self-approved, gated on a personal launch checklist: safety tests passed at target rates, rate limits verified, at least one real booking completed end-to-end, secrets audit clean.

---

## 13. Open Questions

Each question includes a working recommendation to make this list decision-ready rather than open-ended.

1. **LLM provider (Anthropic vs. OpenAI)?** Affects cost and tool-use ergonomics. *Recommendation: prototype the orchestration loop provider-agnostically in Phase 1, decide by end of Phase 2 based on tool-use reliability in testing.*
2. **Knowledge base editing: admin interface or edit-source-and-redeploy?** *Recommendation: source files + redeploy for v1; revisit only if update frequency becomes painful.*
3. **Exact PII/conversation retention period and deletion mechanism?** *Recommendation: set a concrete default (e.g., 90 days) before Phase 5 and document it; owner-side deletion via a DB script is sufficient in v1.*
4. **CAPTCHA before booking finalization?** *Recommendation: defer until a frontend exists (CAPTCHA is a UI concern); rely on rate limits + confirmation in the API-only phase, but leave a hook in the flow.*
5. **Fallback for abusive/harmful messages (vs. merely off-topic)?** *Recommendation: distinct, terse refusal message + flag-and-notify the owner; decide before Phase 5.*
6. **Soft-hold implementation: does Google Calendar support it cleanly, and what expiry window?** *Recommendation: investigate during Phase 4; if provider support is awkward, implement the hold purely in the app DB (never on the calendar) with a 10-minute expiry.*
7. **Email verification before finalizing bookings?** *Recommendation: skip in v1 (adds friction and an async step that's hard to test API-only); revisit if junk bookings appear post-launch.*
8. **Test tooling: Python CLI script, Postman collection, or minimal terminal chat client?** *Recommendation: a small Python CLI chat client — doubles as living API documentation and a demo artifact.*
9. **Frontend phase: separate PRD or amendment?** *Recommendation: separate PRD, referencing this document's API contract as its dependency.*
