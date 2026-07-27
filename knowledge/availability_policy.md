<!--
TEMPLATE — see knowledge/README.md for the full style guide before writing.

Unlike the other files, this one is read by two different consumers:
  1. The agent, via RAG, if a visitor asks "when are you free?" in plain
     conversation.
  2. The booking flow's slot-proposal logic directly (Issue #18+), which
     needs these values to actually compute free/busy windows — so be
     precise and consistent here, not just readable. Issue #18 will define
     the exact parsing/schema it needs when it lands; write this now as
     clear, consistently-labeled prose so that's a straightforward lift
     rather than a rewrite.

Keep it to one or two ## sections total — this file is short by nature.
Delete this comment block once you're done.
-->

## Availability

- **Working hours:** [e.g. Monday-Friday, 9am-5pm Pacific]
- **Meeting length:** [default duration for an intro/screening call, e.g. 30 minutes]
- **Buffer:** [minimum gap required between meetings, e.g. 15 minutes]
- **Minimum notice:** [how far in advance a slot must be booked, e.g. 24 hours]
- **Timezone:** [the timezone the above hours are stated in — be explicit, since visitors will be in different timezones]
