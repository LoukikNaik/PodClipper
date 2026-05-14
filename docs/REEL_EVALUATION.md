# Reel Evaluation Framework

A systematic approach to scoring generated reels before publishing. Combines automated technical metrics (free, objective) with LLM-as-judge content evaluation (one API call per reel) to produce an actionable scorecard.

---

## Why evaluate at all?

The pipeline currently has **zero quality gate** after rendering. Every reel the LLM selects gets produced and written to `outputs/`. Bad reels ship silently alongside good ones. In a typical run of 3–5 reels, 1–2 may have weak hooks, abrupt endings, or off-topic titles. Without evaluation, the user has to watch every reel manually to decide which to post.

Evaluation solves this by:
- **Auto-skipping** reels below a quality threshold (saves disk + user time).
- **Ranking** reels when multiple are produced so the user knows which to post first.
- **Diagnosing** specific weaknesses ("hook is weak", "ending cuts off", "title doesn't match content").
- **Tracking quality over time** as prompt engineering and pipeline logic evolve.

---

## Two-layer evaluation

### Layer 1: Technical metrics (automated, no LLM)

These are objective measurements computed from data the pipeline already produces. Zero cost.

| Metric | Source | How to compute | Good range | What it catches |
|---|---|---|---|---|
| **Face visibility %** | `detect.py` `face_hits` counter | `face_hits / total_frames × 100` | ≥ 80% | Frames where the crop drifts off-face (gestures, back-of-head) |
| **Crop stability** | Per-frame `bbox.x_center` list | `1 - (std_dev(x_centers) / source_width)` | ≥ 0.85 | Jittery/jumpy crop that looks unprofessional |
| **Speaker coverage %** | `detect.py` non-null bbox count | `non_null / total_frames × 100` | ≥ 70% | Clips with lots of empty/static frames (title cards, transitions) |
| **Duration sweet spot** | `clip.duration` | Range check | 25–60s | Too short (not enough arc) or too long (loses attention) |
| **Words per second** | `len(words) / clip_duration` | Division | 1.5–3.5 wps | Too fast (subtitle unreadable) or too slow (dead air) |
| **Subtitle coverage %** | `sum(word_durations) / clip_duration` | Sum word time ranges | ≥ 60% | Long silent gaps where nothing is being said |

#### Composite technical score

```
tech_score = weighted_average(
    face_visibility   × 0.25,
    crop_stability    × 0.20,
    speaker_coverage  × 0.15,
    duration_in_range × 0.15,   # binary: 1.0 if 25-60s, 0.5 otherwise
    wps_in_range      × 0.15,   # binary: 1.0 if 1.5-3.5, 0.5 otherwise
    subtitle_coverage × 0.10,
)
```

All inputs normalized to 0.0–1.0. Output: 0.0–1.0.

---

### Layer 2: LLM content evaluation (one call per reel)

An LLM judges the reel's **content quality** by reading the title + transcript cold — as if encountering it on Instagram for the first time.

#### Avoiding self-confirmation bias

The evaluation prompt must NOT reveal:
- That an AI selected the clip (the judge shouldn't know it's reviewing its own work).
- The original `hook_score` or `reason` from the selection step.
- Any selection criteria or prompt text.

The evaluator receives ONLY:
```json
{
  "title": "3 Things Ruining Your Peace",
  "transcript": "So anybody teenager or in their twenties...",
  "duration_seconds": 48,
  "word_count": 102,
  "first_sentence": "So anybody teenager or in their twenties, three things are your biggest enemy.",
  "last_sentence": "So if you want peace, cut these three."
}
```

#### Six evaluation dimensions

Each dimension targets a specific content failure mode. The LLM provides chain-of-thought reasoning BEFORE scoring (reduces lazy/random outputs).

##### 1. Hook strength (first 2 seconds)

> Does the opening line make a stranger stop scrolling?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Generic opening: "So today...", "Let me tell you...", filler words. No reason to keep watching. |
| 2 | Mildly interesting but not urgent. Could scroll past without feeling you missed something. |
| 3 | Has a clear topic but no tension or curiosity gap. You know what it's about but not why you should care. |
| 4 | Strong hook: a bold claim, a specific number, a provocative question, or a named authority. You want to hear more. |
| 5 | Irresistible: you MUST know what happens next. The kind of opening that makes you take your thumb off the screen mid-scroll. |

What to look for: Does the first sentence contain a question, a number, a bold claim, a named person, or directly address the viewer ("you")? Or does it start with throat-clearing ("basically", "so like", "today we're going to")?

##### 2. Arc completeness (setup → payoff)

> Does the clip tell a complete story or argument?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Random fragment — no discernible point or structure. Feels like a slice of a longer conversation. |
| 2 | Has a topic but meanders. Multiple tangents, no clear throughline. |
| 3 | Clear topic and some structure, but the payoff is weak or missing. You understand the setup but didn't get the "so what". |
| 4 | Clean setup → development → payoff. One idea, explored and resolved. Minor filler but not distracting. |
| 5 | Tight, zero-filler arc. Every sentence advances the argument. Clear resolution that reframes the opening hook. |

What to look for: Can you summarize the clip in one sentence (setup) + one sentence (payoff)? If the payoff sentence is hard to identify, the arc is incomplete.

##### 3. Ending satisfaction (last sentence)

> Does the final line feel like a natural, satisfying close?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Cuts off mid-sentence or mid-thought. The speaker was clearly about to say more. |
| 2 | Ends on a complete sentence but the topic wasn't resolved. Feels like "part 1 of 2". |
| 3 | Acceptable ending — the speaker finished a thought, but it's not the climax. The best line was 10 seconds earlier. |
| 4 | Good ending — lands on the payoff sentence. The viewer feels they got the point. |
| 5 | Mic-drop ending — the last line reframes everything, delivers a revelation, or leaves the viewer thinking. The kind of ending that makes someone rewatch. |

What to look for: Read the last sentence in isolation. Does it sound like a concluding statement, or like a random place to stop? Does the segment AFTER the clip (if you had it) likely continue the same thought?

##### 4. Self-containment (context-free understanding)

> Could a stranger with zero context understand and enjoy this?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Requires significant prior knowledge. References "what I said earlier", "as we discussed", or unnamed people/events the viewer wouldn't know. |
| 2 | Mostly requires context. The main point depends on understanding something from before the clip. |
| 3 | Partially self-contained. The topic makes sense but some references are unclear without context. |
| 4 | Mostly self-contained. A stranger gets 90%+ of it. One or two references might be slightly opaque but don't block understanding. |
| 5 | Fully self-contained. A stranger dropping in cold would understand the full point, find it interesting, and not feel they missed anything. |

What to look for: Are there dangling references ("he", "that thing", "the third one") without antecedents? Does the speaker set up the topic from scratch or assume the viewer already knows?

##### 5. Shareability (engagement potential)

> Would someone share this with a friend, comment on it, or save it?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Forgettable. Technically fine but no emotional resonance. No reason to engage. |
| 2 | Mildly interesting but not share-worthy. You'd watch it but not think about it again. |
| 3 | Good content but not remarkable. You might like it but wouldn't share it. |
| 4 | Share-worthy. Contains a specific insight, fact, or perspective that makes you want to tag someone. "My friend needs to hear this." |
| 5 | Viral potential. Provokes a strong reaction — surprise, recognition, disagreement, or inspiration. The kind of reel people screenshot or DM to three friends. |

What to look for: Is there a "quotable" line? Does the content challenge a common belief? Does it name a specific feeling the viewer has experienced? Would it start a conversation?

##### 6. Title-content alignment

> Does the title accurately represent the content and create appropriate curiosity?

Rubric:
| Score | Criteria |
|---|---|
| 1 | Misleading — the title promises something the clip doesn't deliver. Or generic ("Interesting Thoughts"). |
| 2 | Loosely related but the title oversells or misrepresents the content. |
| 3 | Accurate but boring — the title describes the content but doesn't create curiosity. |
| 4 | Good match — the title hooks the right audience and the content delivers on the implied promise. |
| 5 | Perfect — the title creates a specific curiosity gap that the content resolves satisfyingly. After watching, you think "that title was exactly right." |

What to look for: After reading the transcript, does the title feel like it belongs? Would a viewer who clicked because of the title feel satisfied or deceived?

---

#### LLM output format

```json
{
  "hook": {
    "reasoning": "Opens with 'anybody teenager or in their twenties' — directly addresses a demographic, implies universal pain. But uses 'So' as a filler opener which weakens impact slightly.",
    "score": 4
  },
  "arc": {
    "reasoning": "Lists 3 enemies (comparison, overthinking, instant gratification) with one example each, then closes with 'cut these three'. Clean listicle arc, zero tangents.",
    "score": 5
  },
  "ending": {
    "reasoning": "Final line 'if you want peace, cut these three' is a clean directive that resolves the title's promise. Satisfying close.",
    "score": 4
  },
  "standalone": {
    "reasoning": "No prior context needed. The concept of '3 things ruining your peace' is fully introduced and explained within the clip.",
    "score": 5
  },
  "shareability": {
    "reasoning": "The '3 enemies' framing is quotable and relatable. Someone struggling with anxiety would tag a friend. Not controversial enough for viral debate, but solid save-worthy content.",
    "score": 4
  },
  "title_match": {
    "reasoning": "'3 Things Ruining Your Peace' — the clip delivers exactly 3 things and they are specifically about inner peace. Title creates mild curiosity (what are the 3 things?) and the content resolves it.",
    "score": 5
  },
  "overall": 4.5,
  "verdict": "publish",
  "one_line_feedback": "Strong listicle reel. Consider tightening the opener — replace 'So' with a direct statement for a harder hook."
}
```

---

#### Verdict thresholds

| Overall score | Verdict | Action |
|---|---|---|
| ≥ 4.0 | **publish** | Ship it — meets quality bar for social posting. |
| 3.0 – 3.9 | **review** | Has potential but needs human review. The `one_line_feedback` identifies the weakness. User decides. |
| < 3.0 | **skip** | Below quality bar. Auto-excluded from the output or moved to a `skipped/` subdirectory. |

---

## Combining both layers

### Final scorecard (written to sidecar `.txt`)

```
=== QUALITY SCORECARD ===

Technical:
  Face visibility:     96%     ✓
  Crop stability:      0.93    ✓
  Speaker coverage:    100%    ✓
  Duration:            48s     ✓ (sweet spot)
  Words/sec:           2.1     ✓ (good pace)
  Subtitle coverage:   78%     ✓
  Tech score:          0.95

Content (LLM evaluation):
  Hook:                4/5     "Direct address, slight filler opener"
  Arc:                 5/5     "Clean 3-part listicle, zero tangents"
  Ending:              4/5     "Directive close resolves the title"
  Standalone:          5/5     "Fully self-contained"
  Shareability:        4/5     "Quotable, relatable, save-worthy"
  Title match:         5/5     "Exact promise-delivery alignment"
  Content score:       4.5/5

Overall:               4.5/5   → PUBLISH
Feedback:              "Tighten the opener — cut the 'So' for a harder hook."
```

### Score weighting

```
final_score = (tech_score × 0.3) + (content_score / 5.0 × 0.7)
```

Technical is weighted lower (30%) because a technically perfect reel with bad content is still bad. Content is king (70%).

---

## Edge cases and failure modes

### LLM evaluation pitfalls

| Pitfall | Mitigation |
|---|---|
| **Self-confirmation bias** | Evaluator prompt is isolated from selection prompt. No shared context. |
| **Central tendency** (always scores 3-4) | Rubric has concrete anchors per score level. Chain-of-thought forces reasoning before number. |
| **Inconsistency across runs** | Accept ±0.5 variance. Average 2-3 eval runs for critical decisions. |
| **Language/cultural blindness** | Evaluator may underrate content in languages it knows less about. Flag non-English reels for human review regardless of score. |
| **Positivity bias** | Claude tends to be generous. Calibrate by manually scoring 10 reels and comparing. Adjust thresholds if needed. |

### When to skip LLM evaluation

- **Batch processing** (100+ reels): use only technical metrics to filter, LLM-evaluate only the top 20%.
- **Same video re-runs** (prompt iteration): skip LLM eval since the content hasn't changed, only the technical execution.
- **Cost sensitivity**: each eval is ~1K tokens. At scale, switch to a cheaper model (Haiku) for evaluation.

---

## Calibration protocol

Before trusting the evaluator in production, calibrate it:

1. **Generate 10–15 reels** from 2–3 different source videos.
2. **Watch each reel yourself**. Score each dimension 1–5 using the rubrics above.
3. **Run LLM evaluation** on the same reels.
4. **Compare**: compute correlation between your scores and the LLM's per dimension.
5. **Adjust**:
   - If the LLM is consistently generous → lower the publish threshold from 4.0 to 4.3.
   - If a specific dimension is miscalibrated → add more examples to that dimension's rubric.
   - If reasoning is lazy → add "You MUST cite a specific sentence from the transcript" to the prompt.
6. **Lock in thresholds** once correlation is > 0.7 across dimensions.

---

## Implementation notes

### Module: `src/evaluate.py`

```python
def evaluate_reel(
    title: str,
    words: list[Word],
    clip_duration: float,
    face_hits: int,
    total_frames: int,
    x_centers: list[float],
    provider: LLMProvider,
    cfg: SimpleNamespace,
) -> ReelScorecard:
    """Run both technical and LLM evaluation. Returns a scorecard."""
    ...
```

### Pipeline integration

Called in `pipeline.py` after `burn_subtitles`, before writing the sidecar `.txt`. The scorecard is appended to the sidecar.

If `verdict == "skip"` and `cfg.evaluate.auto_skip` is True, the reel is moved to `outputs/<timestamp>/skipped/` instead of the main directory.

### Config

```yaml
evaluate:
  enabled: true
  auto_skip: false          # if true, reels scoring < 3.0 are moved to skipped/
  skip_threshold: 3.0
  publish_threshold: 4.0
```

---

## Future extensions

- **A/B scoring**: generate 2 title variants per reel, LLM picks the stronger one.
- **Thumbnail scoring**: extract the "best frame" (highest face confidence + good composition), LLM rates it.
- **Audience targeting**: score reels for specific audience segments ("would a 25-year-old interested in spirituality engage with this?").
- **Historical tracking**: log scores over time to measure whether pipeline improvements actually produce better reels.
- **Human-in-the-loop calibration**: after the user publishes reels, feed back actual engagement metrics (views, shares, saves) to recalibrate thresholds.
