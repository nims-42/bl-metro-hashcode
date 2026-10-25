"""Repeatable straggler scenarios and run summaries (CONTEXT.md 3.4, 8, 9).

The observer plane: it starts the cluster, injects faults, logs ground truth,
then reads the per-node JSONL files and scores them. Nothing in src/ reads
anything this script writes.

    python experiments/run_experiments.py list
    python experiments/run_experiments.py run compute_fault --n 5
    python experiments/run_experiments.py all --n 5
    python experiments/run_experiments.py overhead --n 5
    python experiments/run_experiments.py summarize <run_id>

By default every run uses Docker Compose with a compose file generated for N
nodes. `--local` runs the same nodes as plain processes on 127.0.0.1 instead,
for development machines without Docker [OUR CHOICE].
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
HISTORY = os.path.join(RESULTS, "history.csv")
HISTORY_FIELDS = ("run_id", "mode", "n", "ring_up_s", "calibrate_s", "state_bytes",
                  "chunk_bytes", "bandwidth_bps", "healthy_step_t", "warmup_steps")

FAILED = "Failed"
LEFT = "Left"


def node_name(i):
    return f"gpu{i}"


def node_index(name):
    return int(name[3:])


# --- Scenarios (section 8) -------------------------------------------------------
# Each fault: node offset from the default target (N // 2), type, delay, steps.
# "compute" and "network" faults are injected by the node itself (sleep on the
# compute or send path); "cpus" uses `docker update --cpus` and is Docker-only.
# A fault with "inject": False is ground truth only: the slowness comes from the
# scenario's per-node env instead (node_env, {node index: {VAR: value}}).

COMPUTE_SPREAD_MS = (0, 10, -5, 5, -10)   # per-node offsets, inside FALCON's 10% band


def per_node_compute_ms(n, base=100):
    """Every node gets its own processing time; the target is twice the base."""
    times = {i: base + COMPUTE_SPREAD_MS[i % len(COMPUTE_SPREAD_MS)] for i in range(n)}
    times[n // 2] = 2 * base
    return times


def scenarios(n, steps):
    target = n // 2
    start = max(40, steps // 5)
    compute_ms = per_node_compute_ms(n)
    return {
        "no_fault": {"faults": [], "env": {}},
        "compute_fault": {"faults": [
            {"node": target, "type": "compute", "delay_ms": 100, "start_step": start}]},
        "network_fault": {"faults": [
            {"node": target, "type": "network", "delay_ms": 25, "start_step": start}]},
        "jitter_only": {"faults": [], "env": {"JITTER_MS": "30"}},
        "slow_accuser": {"faults": [
            {"node": target, "type": "compute", "delay_ms": 150, "start_step": start},
            {"node": (target + 1) % n, "type": "compute", "delay_ms": 75, "start_step": start}]},
        "cpu_limit": {"docker_only": True, "faults": [
            {"node": target, "type": "cpus", "cpus": 0.25, "start_after_s": 15.0}]},
        "per_node_compute": {
            "node_env": {i: {"COMPUTE_MS": str(ms)} for i, ms in compute_ms.items()},
            "faults": [{"node": target, "type": "compute", "inject": False, "start_step": 0,
                        "delay_ms": compute_ms[target] - min(compute_ms.values())}]},
    }


def expected_class(fault_type):
    return "network" if fault_type == "network" else "compute"


# --- Running the cluster ---------------------------------------------------------

def node_env(scenario, i, n, run_id, steps, overrides):
    env = {"NODE_ID": str(i), "N": str(n), "RUN_ID": run_id, "STEPS": str(steps)}
    env.update(scenario.get("env", {}))
    env.update(scenario.get("node_env", {}).get(i, {}))
    # Plain overrides apply to every node; "GPU2.COMPUTE_MS" only to gpu2 and wins.
    for key, value in sorted(overrides.items(), key=lambda kv: "." in kv[0]):
        node, _, var = key.rpartition(".")
        if not node or node.lower() == node_name(i):
            env[var] = value
    for f in scenario["faults"]:
        if f["node"] == i and f["type"] in ("compute", "network") and f.get("inject", True):
            env.update({"FAULT_NODE": str(i), "FAULT_TYPE": f["type"],
                        "FAULT_DELAY_MS": str(f["delay_ms"]),
                        "FAULT_START_STEP": str(f["start_step"]),
                        "FAULT_END_STEP": str(f.get("end_step", -1))})
    return env


def run_local(scenario, n, run_id, steps, overrides, timeout):
    procs = []
    for i in range(n):
        # CPU limits only exist under Docker, and N busy processes sharing one
        # desktop add noise, so local nodes sleep through their compute.
        env = dict(os.environ, RESULTS_DIR=RESULTS, HOST_TEMPLATE="127.0.0.1",
                   PORT_STRIDE="10", RING_PORT="17000", PING_PORT="17001",
                   COMPUTE_MODE="sleep",
                   **node_env(scenario, i, n, run_id, steps, overrides))
        procs.append(subprocess.Popen([sys.executable, "-m", "src.node"], cwd=REPO, env=env))
    deadline = time.monotonic() + timeout
    codes = []
    for p in procs:
        try:
            codes.append(p.wait(max(1.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            p.kill()
            codes.append(None)
    return codes


def write_compose(scenario, n, run_id, steps, overrides, path):
    lines = ["name: swim-" + run_id.lower().replace("_", "-"), "services:"]
    results = RESULTS.replace("\\", "/")
    for i in range(n):
        env = node_env(scenario, i, n, run_id, steps, overrides)
        lines += [f"  {node_name(i)}:",
                  f"    build: {json.dumps(REPO.replace(chr(92), '/'))}",
                  "    image: swim-lifeguard:latest",
                  "    networks: [ring-net]",
                  f"    volumes: [{json.dumps(results + ':/results')}]",
                  "    environment:"]
        lines += [f"      {k}: {json.dumps(str(v))}" for k, v in sorted(env.items())]
    lines += ["networks:", "  ring-net:", "    driver: bridge", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def run_docker(scenario, n, run_id, steps, overrides, timeout, ground_truth):
    compose = os.path.join(RESULTS, run_id, "compose.generated.yaml")
    write_compose(scenario, n, run_id, steps, overrides, compose)
    base = ["docker", "compose", "-f", compose]
    cpu_faults = [f for f in scenario["faults"] if f["type"] == "cpus"]
    up = subprocess.Popen(base + ["up", "--build"], cwd=REPO)
    workers = [threading.Thread(target=apply_cpu_limit, args=(run_id, f, ground_truth), daemon=True)
               for f in cpu_faults]
    for w in workers:
        w.start()
    try:
        code = up.wait(timeout)
    except subprocess.TimeoutExpired:
        up.kill()
        code = None
    subprocess.run(base + ["down"], cwd=REPO)
    return [code]


def apply_cpu_limit(run_id, fault, ground_truth):
    """`docker update --cpus` on the target container after a delay."""
    import docker  # Docker SDK, only needed for Docker runs

    time.sleep(fault["start_after_s"])
    client = docker.from_env()
    project = "swim-" + run_id.lower().replace("_", "-")
    service = node_name(fault["node"])
    for container in client.containers.list(filters={
            "label": [f"com.docker.compose.project={project}",
                      f"com.docker.compose.service={service}"]}):
        # Same CFS quota that `docker update --cpus <x>` sets.
        container.update(cpu_period=100_000, cpu_quota=int(fault["cpus"] * 100_000))
    fault["start_ts"] = time.time()
    ground_truth["faults_applied"].append(dict(fault))


def run_scenario(name, n, steps, local, overrides, run_id=None, action_cost=None,
                 r_percentile=50):
    scenario = scenarios(n, steps)[name]
    if scenario.get("docker_only") and local:
        print(f"skipping {name}: needs Docker")
        return None
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{name}-n{n}"
    run_dir = os.path.join(RESULTS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ground_truth = {"run_id": run_id, "scenario": name, "n": n, "steps": steps,
                    "mode": "local" if local else "docker", "env": {**scenario.get("env", {}),
                                                                    **overrides},
                    "started_ts": time.time(),
                    "faults": [{**f, "node": node_name(f["node"])} for f in scenario["faults"]],
                    "node_env": {node_name(i): env for i, env in
                                 scenario.get("node_env", {}).items()},
                    "faults_applied": []}
    save_json(os.path.join(run_dir, "ground_truth.json"), ground_truth)
    timeout = steps * 1.0 + 120
    if local:
        codes = run_local(scenario, n, run_id, steps, overrides, timeout)
    else:
        codes = run_docker(scenario, n, run_id, steps, overrides, timeout, ground_truth)
    ground_truth["finished_ts"] = time.time()
    ground_truth["exit_codes"] = codes
    for f in ground_truth["faults_applied"]:
        for g in ground_truth["faults"]:
            if g["node"] == node_name(f["node"]) and g["type"] == f["type"]:
                g["start_ts"] = f["start_ts"]
    save_json(os.path.join(run_dir, "ground_truth.json"), ground_truth)
    return summarize(run_id, action_cost, r_percentile)


# --- Scoring (section 9) -----------------------------------------------------------

def load_run(run_dir, n):
    logs = {}
    for i in range(n):
        path = os.path.join(run_dir, f"{node_name(i)}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                logs[node_name(i)] = [json.loads(line) for line in fh if line.strip()]
    return logs


def fault_start(fault, logs):
    """(step, ts) at which a fault took effect, from the node's own log."""
    records = logs.get(fault["node"], [])
    if "start_ts" in fault:
        for r in records:
            if r["ts"] >= fault["start_ts"]:
                return r["step"], fault["start_ts"]
        return None, fault["start_ts"]
    for r in records:
        for e in r["events"]:
            if e["type"] == "FAULT_START":
                return r["step"], e["ts"]
    step = fault.get("start_step")
    if step is not None and step < len(records):
        # No FAULT_START event (slowness set through env): the step's start time.
        return step, records[step]["ts"] - records[step]["step_t"]
    return step, None


def first_event(logs, kind, about, origin_only=False):
    """Earliest (ts, step, event) of `kind` about `about` across all nodes."""
    best = None
    for node, records in logs.items():
        for r in records:
            for e in r["events"]:
                if e["type"] == kind and e["about"] == about:
                    if origin_only and e.get("from") != node:
                        continue
                    if best is None or e["ts"] < best[0]:
                        best = (e["ts"], r["step"], e)
    return best


def failed_views(logs):
    """{observer: {about: (first step, ts) the observer held `about` Failed}}."""
    out = {}
    for node, records in logs.items():
        seen = {}
        for r in records:
            for about, st in r["member_states"].items():
                if st["state"] == FAILED and about not in seen:
                    seen[about] = (r["step"], r["ts"])
        out[node] = seen
    return out


# --- Cost of a straggler ---------------------------------------------------------------
# The straggler's successor logs pred_lag: the time the whole ring lost to the
# straggler on that step. Their running sum since the slowdown began is the job
# time lost to it (slow_impact); GPU time lost is N times that. Given a one-off
# action cost R (drop or replace the node), acting pays off from the first step
# where slow_impact >= R: FALCON's ski-rental rule, never more than 2x the best
# choice made in hindsight. Measured after the run; nothing here acts.

def straggler_cost(logs, about, n, onset, confirm_step, action_cost):
    succ = node_name((node_index(about) + 1) % n)
    lags = sorted((r["step"], r["pred_lag"]) for r in logs.get(succ, [])
                  if onset is not None and r["step"] >= onset)
    cost = {"onset_step": onset, "cost_measured_by": succ, "cost_per_step_s": None,
            "slow_impact_at_confirm_s": None, "gpu_s_at_confirm": None,
            "slow_impact_total_s": None, "gpu_s_total": None,
            "action_cost_s": action_cost, "act_at_step": None}
    if not lags:
        return cost
    running, at_confirm = 0.0, None
    for step, lag in lags:
        running += lag
        if confirm_step is not None and step <= confirm_step:
            at_confirm = running
        if action_cost is not None and cost["act_at_step"] is None and running >= action_cost:
            cost["act_at_step"] = step
    cost["cost_per_step_s"] = round(statistics.median(lag for _, lag in lags), 4)
    cost["slow_impact_total_s"] = round(running, 3)
    cost["gpu_s_total"] = round(n * running, 1)
    if at_confirm is not None:
        cost["slow_impact_at_confirm_s"] = round(at_confirm, 3)
        cost["gpu_s_at_confirm"] = round(n * at_confirm, 1)
    return cost


# --- Estimating R from previous runs (--action-cost auto) --------------------------------
# R for "drop the straggler and re-form the ring" =
#     T_ring   past ring formation time at this mode and N (smallest RING_UP connect
#              time across a run's nodes: the last node up waits least, which is
#              closest to re-forming a ring whose members are all running)
#   + T_state  state bytes / this run's effective all-reduce bandwidth
#   + T_warmup warmup steps x this run's healthy step time
# Each scored run adds its row to results/history.csv. Rebalancing (R ~ 0) and
# checkpoint/restart (needs checkpoint size and disk speed) are not modelled.

def ring_profile(run_id, gt, logs, healthy_until):
    """This run's ring-formation numbers, or None if it predates RING_UP."""
    n = gt["n"]
    ups = [e for records in logs.values() if records
           for e in records[0]["events"] if e["type"] == "RING_UP"]
    if len(ups) < n:
        return None
    warmup = ups[0]["warmup_steps"]
    chunk_bytes = ups[0]["chunk_bytes"]
    healthy = [r for records in logs.values() for r in records
               if warmup <= r["step"] < healthy_until]
    step_t = statistics.median(r["step_t"] for r in healthy) if healthy else None
    comm = statistics.median(r["step_t"] - r["compute_t"] for r in healthy) if healthy else None
    # Each node sends and receives 2(N-1) chunks per step.
    bandwidth = 2 * (n - 1) * chunk_bytes / comm if comm else None
    return {"run_id": run_id, "mode": gt.get("mode"), "n": n,
            "ring_up_s": min(e["connect_s"] for e in ups),
            "calibrate_s": max(e["calibrate_s"] for e in ups),
            "state_bytes": ups[0]["state_bytes"], "chunk_bytes": chunk_bytes,
            "bandwidth_bps": round(bandwidth) if bandwidth else None,
            "healthy_step_t": round(step_t, 4) if step_t else None, "warmup_steps": warmup}


def load_history():
    if not os.path.exists(HISTORY):
        return []
    rows = []
    with open(HISTORY, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for key in HISTORY_FIELDS[2:]:
                if row[key] in ("", None):
                    row[key] = None
                elif key in ("n", "state_bytes", "chunk_bytes", "bandwidth_bps", "warmup_steps"):
                    row[key] = int(float(row[key]))
                else:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def save_history(profile):
    """Add this run's row, replacing an earlier row for the same run."""
    rows = [r for r in load_history() if r["run_id"] != profile["run_id"]] + [profile]
    with open(HISTORY, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in HISTORY_FIELDS} for r in rows)


def percentile(values, pct):
    """Nearest-rank percentile: the smallest value with at least pct% at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def estimate_action_cost(profile, history, pct):
    """R from previous runs of the same mode, plus this run's own figures."""
    past = [h for h in history if h["run_id"] != profile["run_id"]
            and h["mode"] == profile["mode"] and h["ring_up_s"] is not None]
    if past:
        nearest = min({h["n"] for h in past}, key=lambda m: (abs(m - profile["n"]), m))
        pool = [h for h in past if h["n"] == nearest]
        source = (f"{len(pool)} previous {profile['mode']} run(s) at N={nearest}"
                  + ("" if nearest == profile["n"] else " (nearest N)"))
    else:
        pool, source = [profile], "this run only (no history yet)"
    t_ring = percentile([h["ring_up_s"] for h in pool], pct)
    pool_bw = [h["bandwidth_bps"] for h in pool if h["bandwidth_bps"]]
    bandwidth = profile["bandwidth_bps"] or (statistics.median(pool_bw) if pool_bw else None)
    t_state = profile["state_bytes"] / bandwidth if bandwidth else 0.0
    pool_step = [h["healthy_step_t"] for h in pool if h["healthy_step_t"]]
    step_t = profile["healthy_step_t"] or (statistics.median(pool_step) if pool_step else 0.0)
    t_warmup = profile["warmup_steps"] * step_t
    return {"R_s": round(t_ring + t_state + t_warmup, 3), "source": source,
            "percentile": pct, "t_ring_s": round(t_ring, 4), "t_state_s": round(t_state, 4),
            "t_warmup_s": round(t_warmup, 4)}


def summarize(run_id, action_cost=None, r_percentile=50):
    run_dir = os.path.join(RESULTS, run_id)
    gt = load_json(os.path.join(run_dir, "ground_truth.json"))
    n = gt["n"]
    logs = load_run(run_dir, n)
    faulty = {f["node"] for f in gt["faults"]}
    healthy = [node_name(i) for i in range(n) if node_name(i) not in faulty]
    views = failed_views(logs)

    starts = [fault_start(f, logs)[0] for f in gt["faults"]]
    healthy_until = min((s for s in starts if s is not None), default=10 ** 9)
    profile = ring_profile(run_id, gt, logs, healthy_until)
    estimate = None
    if action_cost == "auto":
        if profile is None:
            print(f"{run_id}: no RING_UP events (older run), cannot estimate R")
            action_cost = None
        else:
            estimate = estimate_action_cost(profile, load_history(), r_percentile)
            action_cost = estimate["R_s"]
    if profile is not None:
        save_history(profile)

    detections = []
    for fault in gt["faults"]:
        about = fault["node"]
        start_step, start_ts = fault_start(fault, logs)
        raised = first_event(logs, "SUSPECT_RAISED", about, origin_only=True)
        confirm = first_event(logs, "CONFIRM", about, origin_only=True)
        others = [v for o, v in views.items() if o != about]
        held = [v[about] for v in others if about in v]
        full = max(held, key=lambda v: v[1]) if held and len(held) == len(others) else None
        row = {
            "node": about, "fault": fault["type"], "expected_class": expected_class(fault["type"]),
            "fault_start_step": start_step, "detected": confirm is not None,
            "detected_by": confirm[2]["from"] if confirm else None,
            "class": confirm[2].get("class") if confirm else None,
        }
        row["class_correct"] = row["class"] == row["expected_class"] if confirm else None
        for key, ev in (("suspect", raised), ("confirm", confirm)):
            row[f"{key}_latency_steps"] = ev[1] - start_step if ev and start_step is not None else None
            row[f"{key}_latency_s"] = round(ev[0] - start_ts, 3) if ev and start_ts else None
        row["full_dissemination_steps"] = full[0] - start_step if full and start_step is not None else None
        row["full_dissemination_s"] = round(full[1] - start_ts, 3) if full and start_ts else None
        # The slowdown began where the accuser's late streak began; if it was
        # never suspected, at the injected start (the cost of missing it).
        onset = raised[2]["window"][0] if raised and raised[2].get("window") else start_step
        row.update(straggler_cost(logs, about, n, onset, confirm[1] if confirm else None,
                                  action_cost))
        row["action_cost_source"] = (estimate["source"] if estimate
                                     else "given" if action_cost is not None else None)
        detections.append(row)

    fp = [(o, a) for o, v in views.items() for a in v if a in healthy]
    fp_minus = [(o, a) for o, a in fp if o in healthy]
    events = [e for records in logs.values() for r in records for e in r["events"]]

    def count(kind, about_healthy=None):
        return sum(1 for e in events if e["type"] == kind and (
            about_healthy is None or (e["about"] in healthy) == about_healthy))

    finals = {node: records[-1] for node, records in logs.items() if records}
    msgs_sent = sum(r["msgs_sent"] for r in finals.values())
    ping_reqs = sum(r["ping_req_sent"] for r in finals.values())
    first_fault = min((d["fault_start_step"] for d in detections
                       if d["fault_start_step"] is not None), default=None)
    warmup = 10

    def median_step(lo, hi):
        values = [r["step_t"] for records in logs.values() for r in records
                  if lo <= r["step"] < hi]
        return round(statistics.median(values), 4) if values else None

    summary = {
        "run_id": run_id, "scenario": gt["scenario"], "n": n, "mode": gt.get("mode"),
        "steps_logged": {node: len(records) for node, records in logs.items()},
        "faulty": sorted(faulty), "detections": detections,
        "false_positives": len(fp), "false_positives_at_healthy": len(fp_minus),
        "false_positive_pairs": [{"observer": o, "about": a} for o, a in fp],
        "suspicions_raised_about_healthy": count("SUSPECT_RAISED", True),
        "suspicions_dropped": count("SUSPICION_DROPPED"),
        "jitter_ignored": count("JITTER_IGNORED"),
        "accusations_held": count("ACCUSATION_HELD"),
        "refutations": count("REFUTE"),
        "ring_msgs_sent": msgs_sent, "ping_req_sent": ping_reqs,
        "extra_msg_fraction": round(ping_reqs / msgs_sent, 6) if msgs_sent else None,
        "median_step_t": median_step(warmup, 10 ** 9),
        "median_step_t_before_fault": median_step(warmup, first_fault) if first_fault else None,
        "median_step_t_during_fault": median_step(first_fault, 10 ** 9) if first_fault else None,
        "max_lhm": {node: max(r["lhm"] for r in records) for node, records in logs.items()},
        "action_cost": estimate or ({"R_s": action_cost, "source": "given"}
                                    if action_cost is not None else None),
        "ring_profile": profile,
    }
    save_json(os.path.join(run_dir, "summary.json"), summary)
    write_csv(os.path.join(run_dir, "summary.csv"), summary)
    return summary


def write_csv(path, summary):
    base = {k: summary[k] for k in ("run_id", "scenario", "n", "false_positives",
                                    "false_positives_at_healthy", "ping_req_sent",
                                    "extra_msg_fraction", "median_step_t")}
    rows = [{**base, **d} for d in summary["detections"]] or [base]
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overhead(n, steps, local, overrides):
    """Step time with detection on vs off on a healthy ring (section 9)."""
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    on = run_scenario("no_fault", n, steps, local, {**overrides, "DETECTION": "1"},
                      f"{stamp}-overhead-on-n{n}")
    off = run_scenario("no_fault", n, steps, local, {**overrides, "DETECTION": "0"},
                       f"{stamp}-overhead-off-n{n}")
    result = {
        "n": n, "steps": steps, "detection_on": on["run_id"], "detection_off": off["run_id"],
        "median_step_t_on": on["median_step_t"], "median_step_t_off": off["median_step_t"],
        "step_time_overhead": round(on["median_step_t"] / off["median_step_t"] - 1, 4),
        "extra_msgs_on": on["ping_req_sent"], "extra_msgs_off": off["ping_req_sent"],
    }
    out = os.path.join(RESULTS, f"{stamp}-overhead-n{n}.json")
    save_json(out, result)
    return result


# --- CLI ------------------------------------------------------------------------------

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def action_cost_arg(text):
    return "auto" if text.strip().lower() == "auto" else float(text)


def parse_overrides(pairs):
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        out[key.strip().upper()] = value.strip()
    return out


def print_summary(s):
    if s is None:
        return
    print(f"\n== {s['scenario']} (n={s['n']}, run {s['run_id']})")
    ac = s.get("action_cost")
    if ac and ac["source"] != "given":
        print(f"  R = {ac['R_s']} s = ring {ac['t_ring_s']} + state {ac['t_state_s']} "
              f"+ warmup {ac['t_warmup_s']} (ring time: p{ac['percentile']} of {ac['source']})")
    for d in s["detections"]:
        status = (f"detected by {d['detected_by']} as {d['class']} "
                  f"({'correct' if d['class_correct'] else 'WRONG'} class), "
                  f"confirm after {d['confirm_latency_steps']} steps / {d['confirm_latency_s']} s, "
                  f"everyone after {d['full_dissemination_s']} s") if d["detected"] else "NOT detected"
        print(f"  {d['node']} {d['fault']} fault: {status}")
        if d.get("slow_impact_total_s") is not None:
            at_confirm = (f"{d['slow_impact_at_confirm_s']} s ({d['gpu_s_at_confirm']} GPU-s) "
                          f"by confirm" if d["slow_impact_at_confirm_s"] is not None
                          else "never confirmed")
            line = (f"    cost: {d['cost_per_step_s']} s/step from step {d['onset_step']} "
                    f"(measured by {d['cost_measured_by']}); {at_confirm}, "
                    f"{d['slow_impact_total_s']} s ({d['gpu_s_total']} GPU-s) by the end")
            if d["action_cost_s"] is not None:
                line += (f"; with R = {d['action_cost_s']} s act at step {d['act_at_step']}"
                         if d["act_at_step"] is not None
                         else f"; with R = {d['action_cost_s']} s never act (waiting was cheaper)")
            print(line)
    print(f"  false positives: {s['false_positives']} (at healthy observers: "
          f"{s['false_positives_at_healthy']}), jitter streaks ignored: {s['jitter_ignored']}, "
          f"suspicions dropped: {s['suspicions_dropped']}")
    print(f"  extra messages: {s['ping_req_sent']} ping-req over {s['ring_msgs_sent']} ring frames, "
          f"median step {s['median_step_t']} s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for cmd in ("run", "all", "overhead"):
        p = sub.add_parser(cmd)
        if cmd == "run":
            p.add_argument("scenario")
            p.add_argument("--run-id")
        p.add_argument("--n", type=int, default=5)
        p.add_argument("--steps", type=int, default=300)
        p.add_argument("--local", action="store_true",
                       help="run nodes as local processes instead of Docker containers")
        p.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="node parameter override, e.g. --set ALPHA=2 --set BETA=2")
    p = sub.add_parser("summarize")
    p.add_argument("run_id")
    for name in ("run", "all", "summarize"):
        sub.choices[name].add_argument(
            "--action-cost", type=action_cost_arg, metavar="R",
            help="one-off cost of acting on a straggler: seconds, or 'auto' to estimate it "
                 "from results/history.csv; reports when acting pays off")
        sub.choices[name].add_argument(
            "--r-percentile", type=int, default=50, metavar="P",
            help="with --action-cost auto: percentile of past ring formation times (default 50)")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, sc in scenarios(5, 300).items():
            print(f"{name:15s} faults={sc['faults']} env={sc.get('env', {})}")
        return
    if args.cmd == "summarize":
        print_summary(summarize(args.run_id, args.action_cost, args.r_percentile))
        return
    if args.n < 3:
        parser.error("--n must be at least 3")
    overrides = parse_overrides(args.set)
    if args.cmd == "run":
        if args.scenario not in scenarios(args.n, args.steps):
            parser.error(f"unknown scenario {args.scenario!r}")
        print_summary(run_scenario(args.scenario, args.n, args.steps, args.local, overrides,
                                   args.run_id, args.action_cost, args.r_percentile))
    elif args.cmd == "all":
        for name in scenarios(args.n, args.steps):
            print_summary(run_scenario(name, args.n, args.steps, args.local, overrides,
                                       action_cost=args.action_cost,
                                       r_percentile=args.r_percentile))
    elif args.cmd == "overhead":
        print(json.dumps(overhead(args.n, args.steps, args.local, overrides), indent=2))


if __name__ == "__main__":
    main()
