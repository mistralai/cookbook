"""Constants for layout attribution pass/fail evaluation."""

# Localization pass thresholds (bidirectional IoA):
# IoA(gt, pred) >= 0.50 and IoA(pred, gt) >= 0.20
LOCALIZATION_IOA_THRESHOLD = 0.50
LOCALIZATION_IOA_PRED_THRESHOLD = 0.20

# Attribution overlap threshold for collecting predicted blocks
ATTRIBUTION_OVERLAP_IOA_THRESHOLD = 0.30

# Token F1 threshold for attribution pass
ATTRIBUTION_TOKEN_F1_THRESHOLD = 0.80
