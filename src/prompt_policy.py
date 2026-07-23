"""Shared editorial policy for turning source videos into learning material."""

LEARNING_CONTENT_POLICY = """CONTENT FILTER (apply before producing any item):
- Keep only knowledge or skills that a learner can use outside this specific video.
- Exclude greetings, introductions, outros, thanks, calls to subscribe, channel news,
  sponsorships, and descriptions of what the video will or did cover.
- Exclude the author's biography, tastes, preferences, feelings, plans, workflow, and
  personal anecdotes unless the video explicitly teaches that information as its subject.
- Do not turn phrases such as "I like...", "I prefer...", "in this video...", or
  "next we will..." into facts merely because they are stated clearly.
- An anecdote or opinion may supply context for a genuine lesson, but it is not itself
  a learning item. Extract only the transferable lesson that the source explicitly supports.
- If a passage contains no transferable learning content, return no item for it.
When uncertain, omit the item. Precision is more important than coverage."""
