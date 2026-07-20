#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SERIES_TAGS = {
    "vopd_loss": "actor/vopd_loss",
    "raw_jsd": "self_distillation/raw_jsd_token_mean",
    "grad_norm": "actor/grad_norm",
    "rollout_kl": "rollout_corr/kl",
    "training_ppl": "rollout_corr/training_ppl",
    "rollout_ppl": "rollout_corr/rollout_ppl",
    "ppl_ratio": "rollout_corr/ppl_ratio",
    "log_ppl_abs_diff": "rollout_corr/log_ppl_abs_diff",
    "response_clip": "response_length/clip_ratio",
    "distill_tokens": "self_distillation/num_distill_tokens",
}
EXACT_ONE_TAGS = (
    "self_distillation/teacher_always_on_fraction",
    "self_distillation/teacher_image_swap_fraction",
    "self_distillation/self_distillation_mask.mean()",
)
EXACT_ZERO_TAGS = (
    "self_distillation/policy_fallback_fraction",
    "actor/policy_fallback_fraction",
    "self_distillation/empty_target_batch",
    "prompt_length/clip_ratio",
    "response/aborted_ratio",
)


def mean(values):
    return statistics.fmean(values) if values else math.nan


def fmt(value, digits=6):
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def latest_event_file(directory):
    files = list(directory.glob("events.out.tfevents.*"))
    if not files:
        raise SystemExit(f"no TensorBoard event file in {directory}")
    return max(files, key=lambda path: path.stat().st_mtime_ns)


def load_events(event_path):
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))

    def scalar(tag):
        if tag not in available:
            return {}
        return {event.step: float(event.value) for event in accumulator.Scalars(tag)}

    return scalar


def trend_check(name, reference, recent, pass_ratio=0.90, warn_ratio=1.05):
    ratio = recent / reference if reference > 0 else math.inf
    if ratio <= pass_ratio:
        status = "PASS"
    elif ratio <= warn_ratio:
        status = "WARN"
    else:
        status = "FAIL"
    return {
        "name": name,
        "status": status,
        "reference": reference,
        "recent": recent,
        "ratio": ratio,
    }


def plot_curves(series, output_path, label):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ("vopd_loss", "VOPD loss"),
        ("raw_jsd", "Raw token JSD"),
        ("rollout_kl", "Rollout KL"),
        ("grad_norm", "Gradient norm"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        points = sorted(series[key].items())
        axis.plot([step for step, _ in points], [value for _, value in points], marker="o", markersize=3)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    figure.suptitle(f"{label}: strict Vision-OPD training health")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Summarize early health of a strict Vision-OPD run")
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--fail-on-red", action="store_true")
    args = parser.parse_args()

    event_path = latest_event_file(args.tensorboard_dir.resolve())
    scalar = load_events(event_path)
    series = {name: scalar(tag) for name, tag in SERIES_TAGS.items()}
    if not series["vopd_loss"]:
        raise SystemExit("actor/vopd_loss has no events")
    max_step = args.expected_step or max(series["vopd_loss"])
    expected_steps = set(range(1, max_step + 1))
    findings = []

    for name, values in series.items():
        actual = set(values) & expected_steps
        if actual != expected_steps:
            findings.append({"name": f"contiguous {name}", "status": "FAIL", "detail": f"{len(actual)}/{max_step}"})
        selected = [values[step] for step in sorted(actual)]
        if any(not math.isfinite(value) for value in selected):
            findings.append({"name": f"finite {name}", "status": "FAIL", "detail": "non-finite value"})

    invariant_failures = []
    for tag in EXACT_ONE_TAGS:
        values = scalar(tag)
        if any(not math.isclose(values.get(step, math.nan), 1.0, abs_tol=1e-8) for step in expected_steps):
            invariant_failures.append(tag)
    for tag in EXACT_ZERO_TAGS:
        values = scalar(tag)
        if any(not math.isclose(values.get(step, math.nan), 0.0, abs_tol=1e-8) for step in expected_steps):
            invariant_failures.append(tag)
    ema = scalar("timing_s/update_actor/teacher_ema_update")
    if any(ema.get(step, 0.0) <= 0 for step in expected_steps):
        invariant_failures.append("timing_s/update_actor/teacher_ema_update")
    findings.append({
        "name": "teacher/bbox/fallback/EMA invariants",
        "status": "FAIL" if invariant_failures else "PASS",
        "detail": ", ".join(invariant_failures) if invariant_failures else "all steps valid",
    })

    reference_steps = list(range(11, min(15, max_step) + 1)) if max_step >= 15 else list(range(1, min(5, max_step) + 1))
    recent_steps = list(range(max(1, max_step - 4), max_step + 1))
    for key, title in (("vopd_loss", "VOPD loss trend"), ("raw_jsd", "raw JSD trend"), ("rollout_kl", "rollout KL trend")):
        reference = mean([series[key][step] for step in reference_steps if step in series[key]])
        recent = mean([series[key][step] for step in recent_steps if step in series[key]])
        finding = trend_check(title, reference, recent)
        finding["detail"] = f"recent/reference={finding['ratio']:.3f}"
        findings.append(finding)

    recent_grad = [series["grad_norm"][step] for step in recent_steps if step in series["grad_norm"]]
    grad_status = "PASS" if recent_grad and min(recent_grad) > 1e-8 and max(recent_grad) < 10 else "FAIL"
    findings.append({"name": "gradient stability", "status": grad_status, "detail": f"recent range={fmt(min(recent_grad))}-{fmt(max(recent_grad))}"})
    recent_ppl_ratio = [series["ppl_ratio"][step] for step in recent_steps if step in series["ppl_ratio"]]
    recent_log_ppl_diff = [
        series["log_ppl_abs_diff"][step] for step in recent_steps if step in series["log_ppl_abs_diff"]
    ]
    max_ratio_deviation = (
        max(abs(value - 1.0) for value in recent_ppl_ratio) if recent_ppl_ratio else math.inf
    )
    max_log_ppl_diff = max(recent_log_ppl_diff) if recent_log_ppl_diff else math.inf
    if max_ratio_deviation <= 0.05 and max_log_ppl_diff <= 0.15:
        ppl_status = "PASS"
    elif max_ratio_deviation <= 0.25 and max_log_ppl_diff <= 0.30:
        ppl_status = "WARN"
    else:
        ppl_status = "FAIL"
    findings.append({
        "name": "rollout/training PPL alignment",
        "status": ppl_status,
        "detail": (
            f"recent max |ratio-1|={max_ratio_deviation:.3f}, "
            f"max |log-PPL diff|={max_log_ppl_diff:.3f}"
        ),
    })
    recent_clip = mean([series["response_clip"][step] for step in recent_steps if step in series["response_clip"]])
    clip_status = "PASS" if recent_clip <= 0.05 else "WARN" if recent_clip <= 0.10 else "FAIL"
    findings.append({"name": "response truncation", "status": clip_status, "detail": f"recent mean={recent_clip:.2%}"})

    statuses = {finding["status"] for finding in findings}
    verdict = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    if max_step < 15 and verdict == "PASS":
        verdict = "INSUFFICIENT"

    plot_curves(series, args.output_plot.resolve(), args.label)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "event_file": str(event_path),
        "step": max_step,
        "verdict": verdict,
        "reference_steps": reference_steps,
        "recent_steps": recent_steps,
        "findings": findings,
        "latest": {key: values.get(max_step) for key, values in series.items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_link = args.output_plot.name if args.output_plot.parent == args.output_md.parent else str(args.output_plot)
    lines = [
        f"# {args.label} Early Training Health",
        "",
        f"**Verdict: {verdict}** at optimizer step **{max_step}**.",
        "",
        "> This is an early failure detector, not a substitute for the official six-benchmark alignment gate. "
        "The Vision-OPD paper reports objective/regularization ablations and final benchmark scores, but no "
        "per-step training-loss, KL, or gradient-norm reference curve.",
        "",
        f"![Training health curves]({plot_link})",
        "",
        f"Reference window: steps `{reference_steps[0]}-{reference_steps[-1]}`; recent window: steps `{recent_steps[0]}-{recent_steps[-1]}`.",
        "",
        "| Check | Status | Reference | Recent | Detail |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for finding in findings:
        lines.append(
            f"| {finding['name']} | {finding['status']} | {fmt(finding.get('reference', math.nan))} | "
            f"{fmt(finding.get('recent', math.nan))} | {finding['detail']} |"
        )
    lines.extend([
        "",
        "## Latest Scalars",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ])
    for key, value in payload["latest"].items():
        lines.append(f"| `{SERIES_TAGS[key]}` | {fmt(value if value is not None else math.nan)} |")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{verdict} {args.label} early health through step {max_step}; wrote {args.output_md}")
    if args.fail_on_red and verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
