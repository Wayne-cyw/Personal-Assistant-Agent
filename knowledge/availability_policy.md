<!--
TEMPLATE — see knowledge/README.md for the full style guide before writing.

Unlike the other files, this one is read by two different consumers:
  1. The agent, via RAG, if a visitor asks "when are you free?" in plain
     conversation.
  2. app/booking/slots.py's parser (Issue #19) directly, to actually
     compute free/busy windows — so the exact format below matters, not
     just readability. Fill in the value after each label, keep the label
     text and "- **Label:**" formatting exactly as shown, and remove the
     surrounding [ ] brackets — the parser rejects a value that still
     looks like an unfilled placeholder (still starting with "[") with a
     clear startup error, specifically so this can't ship half-filled-in
     by accident.

Format each line must follow after filling in:
  - Timezone: a real IANA name, e.g. "America/New_York" (not "Pacific" or
    an abbreviation like "PST") — this is the timezone every other line
    below is interpreted in.
  - Working hours: "<StartDay>-<EndDay>, HH:MM-HH:MM", 24-hour clock, e.g.
    "Monday-Friday, 09:00-17:00". One contiguous day range, one daily time
    range — if availability genuinely varies by day, use the narrowest
    range true every day and handle exceptions manually for now.
  - Meeting length / Buffer / Minimum notice: "<number> <unit>", unit is
    minutes/hours/days (singular or plural both fine), e.g. "30 minutes",
    "1 day".

Keep it to one or two ## sections total — this file is short by nature.
Delete this comment block once you're done.
-->

## Availability

- **Timezone:** [e.g. America/New_York]
- **Working hours:** [e.g. Monday-Friday, 09:00-17:00]
- **Meeting length:** [e.g. 30 minutes]
- **Buffer:** [e.g. 15 minutes]
- **Minimum notice:** [e.g. 24 hours]
