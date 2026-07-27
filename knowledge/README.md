# Knowledge base — authoring guide

This directory is the source content the agent answers from (retrieved via RAG, Issue #14/#15). **Content quality caps answer quality** — the agent never knows more than what's written here, and it's instructed to say "I'm not sure" rather than guess when a question isn't covered.

## Files

| File | Contents |
|---|---|
| `bio.md` | Background/experience narrative |
| `projects.md` | One `##` section per project: what it is, stack, role, outcomes |
| `skills.md` | Technology → depth/years/context |
| `faq.md` | Anticipated recruiter questions, answered directly |
| `availability_policy.md` | Working hours, meeting length, buffer, minimum notice — read directly by the booking flow (Issue #18+), not only retrieved via RAG |
| `intro.md` | *(already exists, Issue #9)* the deterministic turn-zero greeting — not part of the RAG corpus, don't add `##` chunk headings to it |

## Style rules — apply to every file below

1. **Factual, no fluff.** No marketing language, no vague superlatives ("passionate," "results-driven"). State what's true, plainly.
2. **Third person about the owner, not first person.** Write "Wayne has..." / "Wayne built...", not "I have..." — the agent is a tool speaking *about* the owner, not speaking *as* the owner.
3. **Every claim must be true and current as of when you write it.** The agent states these as established fact to visitors, with no further verification on its end. Don't write anything aspirational, outdated, or exaggerated.
4. **`##` headings are chunk boundaries, not just formatting.** The ingestion pipeline (Issue #14) splits each file on `##` headings, merges any tiny leftover sections together, and hard-splits anything over ~500 tokens. Aim for **~300–500 tokens per `##` section** (roughly 1,200–2,000 characters). Each section should be a complete, self-contained unit — retrieval returns whole chunks to the agent, so a section needs to make sense on its own without the surrounding text.
5. **Don't repeat framing inside the content.** No need to write "this is Wayne's bio" inside `bio.md` — the ingestion pipeline already records which file and heading every chunk came from.

## Updating content later

Whenever you edit a file here, re-run the ingestion script (Issue #14, `scripts/ingest_kb.py`) so the agent's retrieval index picks up the change. Ingestion is a full-replace per source file — editing an existing section, not just adding new ones, takes effect the next time it runs.

## Acceptance bar

Every claim in every file must be fact-checked by the owner before it's live. This is the one place in the whole system where "the agent said something wrong about you" is a content problem, not a code problem.
