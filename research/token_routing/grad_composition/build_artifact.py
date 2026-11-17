"""Fill artifact_template.html with results/artifact_data.json, results/artifact_heat.json and the findings / setup
fragments -> results/grad_composition_artifact.html (publish that file).

    python research/token_routing/grad_composition/art_data.py && python research/token_routing/grad_composition/build_artifact.py
"""
import datetime
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    t = open(os.path.join(HERE, "artifact_template.html")).read()
    data = open(os.path.join(RES, "artifact_data.json")).read().replace("</", "<\\/")
    heat = open(os.path.join(RES, "artifact_heat.json")).read().replace("</", "<\\/")
    find = open(os.path.join(HERE, "artifact_findings.html")).read()
    setup = open(os.path.join(HERE, "artifact_setup.html")).read()
    out = (t.replace("__DATA__", data).replace("__HEAT__", heat).replace("__FINDINGS__", find).replace("__SETUP__", setup)
           .replace("__UPDATED__", datetime.date.today().isoformat())
           .replace("__STATUS__", open(os.path.join(HERE, "artifact_status.html")).read().strip()))
    cptp = os.path.join(RES, "cpt", "summary.json")
    cpt = open(cptp).read().replace("</", "<\\/") if os.path.exists(cptp) else "null"
    cptf = os.path.join(HERE, "artifact_cpt_findings.html")
    out = out.replace("__CPT__", cpt).replace("__CPT_FINDINGS__", open(cptf).read() if os.path.exists(cptf) else "")
    p = os.path.join(RES, "grad_composition_artifact.html")
    open(p, "w").write(out)
    print(f"wrote {p} ({os.path.getsize(p) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
