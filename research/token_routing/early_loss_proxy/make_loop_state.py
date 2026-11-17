"""Assemble hub_tab/loop_state.json (the data of the hub's auto-research tab) from score.py boards + the hand-kept files.

    python3 make_loop_state.py --boards hub_tab/board_v2.json[:label] ... [--pilot hub_tab/board_s50.json]
"""
import argparse, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "hub_tab")
ap = argparse.ArgumentParser()
ap.add_argument("--boards", nargs="*", default=[])
ap.add_argument("--pilot", default="")
a = ap.parse_args()
ld = lambda n: json.load(open(os.path.join(T, n)))  # noqa: E731
arms = json.load(open(os.path.join(T, "arms.json"))) if os.path.exists(os.path.join(T, "arms.json")) else {}
state = {"updated": time.strftime("%Y-%m-%d %H:%M PT"), **ld("notes.json"), "ideas": ld("ideas.json"),
         "best_by_hour": ld("hourly.json"), "validation": ld("validation.json"), "board": [], "curves": [], "pilot": []}


def rows(path):
    d = json.load(open(path))
    out, cur = [], []
    for b in d["board"]:
        meta = arms.get(b["arm"], {})
        out.append({**{k: b.get(k) for k in ("tag", "arm", "seeds", "steps", "eval_size", "is_ref", "ce_last", "frac",
                                             "pf", "pf_per_step_vs_dense", "x", "xw", "peak_gib", "peak_vs_ref",
                                             "train_s", "d_vs_ref", "d_vs_ref_se")},
                    "idea": meta.get("idea", ""), "desc": meta.get("desc", "")})
    for r in d["runs"]:
        cur.append({"tag": r["tag"], "arm": r["arm"], "seed": r["seed"], "is_ref": r["is_ref"], "pts": r["curve"],
                    "by": r["by"]})
    return out, cur


for b in a.boards:
    o, c = rows(b)
    state["board"] += o
    state["curves"] += c
if a.pilot:
    state["pilot"] = rows(a.pilot)[0]
json.dump(state, open(os.path.join(T, "loop_state.json"), "w"), indent=1)
print("wrote loop_state.json:", len(state["board"]), "board rows,", len(state["curves"]), "curves,", len(state["ideas"]), "ideas")
