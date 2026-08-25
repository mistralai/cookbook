"""Per-payment-details association F1 metric.

For multi-PD remittance documents whose schema carries a `claim_numbers`
field on each PaymentDetails, this metric scores how accurately the
extractor associates claims to checks.

Pairing rule: pair an extracted PD with a GT PD when their
``check.check_number`` values match. For each pair, compare the
``claim_numbers`` SETs and compute precision / recall / F1. Aggregate
two ways:

- micro F1: sum tp / sum (tp+fp), sum tp / sum (tp+fn) across all
  matched PD pairs in the doc.
- macro F1: simple mean over PD pairs.

Returns the micro F1 as the primary metric value; both micro and macro
plus per-PD details surface in metadata.
"""

from __future__ import annotations

from typing import Any

from extract_bench.evaluation.metrics.base import Metric
from extract_bench.schemas.evaluation import MetricValue


def _check_number(pd: Any) -> str | None:
    if not isinstance(pd, dict):
        return None
    chk = pd.get("check") or {}
    if isinstance(chk, dict):
        cn = chk.get("check_number")
        if cn:
            return str(cn)
    return None


def _claim_number_set(pd: Any) -> set[str]:
    if not isinstance(pd, dict):
        return set()
    cns = pd.get("claim_numbers") or []
    if not isinstance(cns, list):
        return set()
    return {str(x) for x in cns if x is not None and str(x).strip()}


class ExtractAssociationF1Metric(Metric):
    """Per-PD association F1 on claim_numbers, paired by check_number."""

    @property
    def name(self) -> str:
        return "association_f1"

    def compute(
        self,
        expected: dict[str, Any] | None,
        actual: dict[str, Any] | None,
        **kwargs: Any,
    ) -> MetricValue:
        gt = expected or {}
        ext = actual or {}

        gt_pds = gt.get("payment_details") if isinstance(gt, dict) else None
        ext_pds = ext.get("payment_details") if isinstance(ext, dict) else None

        # Inapplicable cases: emit under a different metric name so they do
        # NOT inflate the aggregate `avg_association_f1`. The dashboard's
        # average then reflects honest F1 over applicable docs only, and the
        # applicable rate is recoverable via `avg_association_f1_applicable`
        # on the sibling metric below.
        if not isinstance(gt_pds, list) or not gt_pds:
            return MetricValue(
                metric_name="association_f1_inapplicable",
                value=1.0,
                metadata={"reason": "GT has no payment_details list"},
            )
        if not any(_claim_number_set(pd) for pd in gt_pds):
            return MetricValue(
                metric_name="association_f1_inapplicable",
                value=1.0,
                metadata={"reason": "GT payment_details have no claim_numbers populated"},
            )

        ext_pds_list = ext_pds if isinstance(ext_pds, list) else []

        # Index extracted PDs by check_number.
        ext_by_chk: dict[str, dict] = {}
        for pd in ext_pds_list:
            cn = _check_number(pd)
            if cn:
                ext_by_chk.setdefault(cn, pd)

        pairs: list[dict[str, Any]] = []
        used_chks: set[str] = set()
        for gt_pd in gt_pds:
            cn = _check_number(gt_pd)
            gt_set = _claim_number_set(gt_pd)
            if cn and cn in ext_by_chk:
                ext_set = _claim_number_set(ext_by_chk[cn])
                matched_by = "check_number"
                used_chks.add(cn)
            else:
                ext_set = set()
                matched_by = "none"
            inter = ext_set & gt_set
            tp = len(inter)
            fp = len(ext_set) - tp
            fn = len(gt_set) - tp
            precision = tp / len(ext_set) if ext_set else 0.0
            recall = tp / len(gt_set) if gt_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            pairs.append(
                {
                    "check_number": cn,
                    "matched_by": matched_by,
                    "ext_size": len(ext_set),
                    "gt_size": len(gt_set),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )

        # Positional fallback for GT PDs without check_number — not applicable
        # to association-bearing schemas (every PD has check.check_number),
        # but harmless to support.
        gt_no_chk = [pd for pd in gt_pds if not _check_number(pd)]
        ext_no_chk = [pd for pd in ext_pds_list if not _check_number(pd)]
        for i, gt_pd in enumerate(gt_no_chk):
            ext_pd = ext_no_chk[i] if i < len(ext_no_chk) else {}
            gt_set = _claim_number_set(gt_pd)
            ext_set = _claim_number_set(ext_pd)
            inter = ext_set & gt_set
            tp = len(inter)
            fp = len(ext_set) - tp
            fn = len(gt_set) - tp
            precision = tp / len(ext_set) if ext_set else 0.0
            recall = tp / len(gt_set) if gt_set else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            pairs.append(
                {
                    "check_number": None,
                    "matched_by": "positional",
                    "ext_size": len(ext_set),
                    "gt_size": len(gt_set),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            )

        # Aggregations
        sum_tp = sum(p["tp"] for p in pairs)
        sum_fp = sum(p["fp"] for p in pairs)
        sum_fn = sum(p["fn"] for p in pairs)
        micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
        micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0
        macro_f1 = (sum(p["f1"] for p in pairs) / len(pairs)) if pairs else 0.0

        weighted_sum = sum(p["f1"] * p["gt_size"] for p in pairs)
        weighted_div = sum(p["gt_size"] for p in pairs)
        weighted_f1 = (weighted_sum / weighted_div) if weighted_div else 0.0

        return MetricValue(
            metric_name=self.name,
            value=round(micro_f1, 4),
            metadata={
                "micro_precision": round(micro_p, 4),
                "micro_recall": round(micro_r, 4),
                "micro_f1": round(micro_f1, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "pd_count_ext": len(ext_pds_list),
                "pd_count_gt": len(gt_pds),
                "pd_pairs": pairs,
            },
        )
