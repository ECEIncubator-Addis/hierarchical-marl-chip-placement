# ppa_evaluator.py
"""
Python Script for Extracting Real PPA Metrics from OpenROAD/TILOS Flow Reports

This script parses standard OpenROAD report files after running a full flow
(with your RL-exported macro placement DEF) to extract true PPA metrics:
- Power (total, internal, switching, leakage)
- Performance/Timing (WNS, TNS)
- Area (cell area, utilization)
- Wirelength (routed or HPWL)
- DRC/Routability (violations)

Usage:
  python ppa_evaluator.py --flow-dir /path/to/Flows/NanGate45/ariane136/results/final/

Assumes standard TILOS/OpenROAD report structure:
  - reports/power.rpt or similar for power
  - reports/timing.rpt for WNS/TNS
  - reports/area.rpt or summary for area
  - reports/route.dr or global_route.rpt for DRC/wirelength

Outputs a structured Markdown table + JSON for easy reporting/comparison.
"""

import argparse
import re
from pathlib import Path
import json

def parse_power_report(report_path: Path) -> dict:
    metrics = {"total_power_mW": None, "internal_mW": None, "switching_mW": None, "leakage_mW": None}
    if not report_path.exists():
        return metrics
    
    content = report_path.read_text()
    
    # Common patterns in OpenROAD power reports
    total_match = re.search(r'Total\s+([\d.E+-]+)\s+m?W', content, re.I)
    if total_match:
        metrics["total_power_mW"] = float(total_match.group(1))
    
    internal_match = re.search(r'Internal\s+([\d.E+-]+)', content)
    if internal_match:
        metrics["internal_mW"] = float(internal_match.group(1))
    
    switching_match = re.search(r'Switching\s+([\d.E+-]+)', content)
    if switching_match:
        metrics["switching_mW"] = float(switching_match.group(1))
    
    leakage_match = re.search(r'Leakage\s+([\d.E+-]+)', content)
    if leakage_match:
        metrics["leakage_mW"] = float(leakage_match.group(1))
    
    return metrics

def parse_timing_report(report_path: Path) -> dict:
    metrics = {"WNS_ns": None, "TNS_ns": None}
    if not report_path.exists():
        return metrics
    
    content = report_path.read_text()
    
    wns_match = re.search(r'WNS\s*:\s*([-]?\d+\.\d+)', content)
    if wns_match:
        metrics["WNS_ns"] = float(wns_match.group(1))
    
    tns_match = re.search(r'TNS\s*:\s*([-]?\d+\.\d+)', content)
    if tns_match:
        metrics["TNS_ns"] = float(tns_match.group(1))
    
    return metrics

def parse_area_report(report_path: Path) -> dict:
    metrics = {"cell_area_um2": None, "utilization_percent": None}
    if not report_path.exists():
        return metrics
    
    content = report_path.read_text()
    
    area_match = re.search(r'Cell Area\s*:\s*([\d.E+-]+)', content)
    if area_match:
        metrics["cell_area_um2"] = float(area_match.group(1))
    
    util_match = re.search(r'Utilization\s*:\s*([\d.]+)%', content)
    if util_match:
        metrics["utilization_percent"] = float(util_match.group(1))
    
    return metrics

def parse_route_report(report_path: Path) -> dict:
    metrics = {"routed_wirelength_um": None, "drc_violations": None}
    if not report_path.exists():
        return metrics
    
    content = report_path.read_text()
    
    wl_match = re.search(r'Wire Length\s*:\s*([\d.E+-]+)', content)
    if wl_match:
        metrics["routed_wirelength_um"] = float(wl_match.group(1))
    
    drc_match = re.search(r'Violations\s*:\s*(\d+)', content)
    if drc_match:
        metrics["drc_violations"] = int(drc_match.group(1))
    
    return metrics

def evaluate_ppa(flow_dir: Path):
    reports_dir = flow_dir / "reports"  # Common in TILOS/OpenROAD flows
    
    power_metrics = parse_power_report(reports_dir / "power.rpt")
    timing_metrics = parse_timing_report(reports_dir / "timing.rpt")
    area_metrics = parse_area_report(reports_dir / "area.rpt")
    route_metrics = parse_route_report(reports_dir / "route.rpt" or reports_dir / "detailed_route.dr")
    
    all_metrics = {
        "Power": power_metrics,
        "Timing": timing_metrics,
        "Area": area_metrics,
        "Routing": route_metrics
    }
    
    # Print Markdown table
    print("# PPA Evaluation Results")
    print(f"Flow Directory: {flow_dir}\n")
    print("| Category | Metric | Value |")
    print("|----------|--------|-------|")
    for cat, mets in all_metrics.items():
        for metric, value in mets.items():
            print(f"| {cat} | {metric.replace('_', ' ').title()} | {value if value is not None else 'N/A'} |")
    
    # Save JSON
    json_path = flow_dir / "ppa_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nJSON metrics saved to: {json_path}")
    
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Extract real PPA metrics from OpenROAD/TILOS flow reports")
    parser.add_argument("--flow-dir", type=str, required=True, help="Path to flow results directory (e.g., Flows/NanGate45/ariane136/results/final/)")
    args = parser.parse_args()
    
    flow_dir = Path(args.flow_dir)
    if not flow_dir.exists():
        raise FileNotFoundError(f"Directory not found: {flow_dir}")
    
    evaluate_ppa(flow_dir)

if __name__ == "__main__":
    main()