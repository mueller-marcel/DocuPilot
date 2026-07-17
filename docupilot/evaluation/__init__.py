"""
Evaluation: how much does each modality contribute to segmentation quality?

NOT IMPLEMENTED YET — this package exists to name the third backend module and to
hold the constraints the implementation has to respect, so they are not
rediscovered later.

What belongs here:

  - Boundary-F1 (precision / recall) of predicted boundaries against the
    annotated ground truth, at a TOLERANCE.
  - The full 2^3 factorial ablation over {video, audio, events}.
  - Shapley values over those eight subsets — exact, since three players means
    eight coalitions and no sampling is needed.

What must NOT happen here, learned the hard way:

  - **Candidates must be generated PER SUBSET.** A deleted earlier version pooled
    the peaks of all three modalities into one candidate list and merged nearby
    ones into their evidence-weighted mean TIME. Every subset then reused that
    list, so the "audio only" arm was scored on candidate times that video had
    helped place: audio never had to localise anything, and its contribution came
    out inflated while video's came out understated. The independent variable was
    not actually being manipulated.

  - **Tolerance is a factor, not a constant.** The ground truth is anchored on the
    visual settling moment, so at ±1 s the video modality wins by construction and
    the saturation question answers itself. Report F1 and Shapley per tolerance
    (0.5 / 1 / 2 / 3 / 5 s); how the contributions move along that axis is the
    result, not a robustness check.

  - **The session is the unit of analysis, not the boundary.** Boundaries within
    one recording are correlated, so cross-validation has to be grouped
    (leave-one-session-out, better leave-one-app-out) and confidence intervals
    bootstrapped over sessions.

  - **Calibrate on a dev split.** `_COMPLETION_POSITION` (audio) and
    `_REST_FULL_S` (events) are marked PROVISIONAL in their modules and must never
    be tuned on the evaluation set.
"""
