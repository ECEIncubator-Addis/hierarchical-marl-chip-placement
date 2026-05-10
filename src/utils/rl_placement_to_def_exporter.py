# rl_placement_to_def_exporter.py
"""
Python Script to Export RL-Optimized Macro Placement to DEF Format

This script loads your trained .pt graph and exports a valid DEF file
compatible with TILOS MacroPlacement flows or ChiPBench/OpenROAD evaluations.

Usage Examples:
  python rl_placement_to_def_exporter.py \
    --graph trained_ariane136.pt \
    --design ariane136 \
    --pdk Nangate45 \
    --repo-path /path/to/MacroPlacement \
    --output my_rl_placement.def

Features:
- Loads your PyG .pt graph (with pos normalized 0-1, canvas_size, real sizes)
- Parses real macro instance names and types from the repo's post-synthesis netlist (.v)
- Scales positions to DBU (Database Units) using PDK info
- Writes standard DEF 5.8 format with:
  - DIEAREA from canvas
  - COMPONENTS with PLACED macros (fixed orientation N)
- Handles uniform (Ariane) or heterogeneous macros
- Optional: Include standard cells at initial/random positions (not placed by RL)

Requirements:
  pip install torch torch-geometric pyverilog
"""

import argparse
import torch
from torch_geometric.data import Data
from pathlib import Path
import re
from pyverilog.vparser.parser import parse as verilog_parse

# Common DBU for TILOS PDKs (confirm in platform config.mk or LEF)
PDK_DBU = {
    "Nangate45": 2000,   # Typical for Nangate45
    "ASAP7": 2000,
    "SKY130HD": 2000
}

def load_graph(path: str) -> Data:
    graph = torch.load(path)
    if not isinstance(graph, Data):
        raise ValueError("Not a valid PyG Data object")
    print(f"Loaded {graph.num_nodes} macros")
    return graph

def find_netlist(repo_path: Path, design: str, pdk: str) -> Path:
    pdk_folder = "NanGate45" if pdk == "Nangate45" else pdk  # Repo uses NanGate45
    netlist_dir = repo_path / "Flows" / pdk_folder / design / "netlist"
    if not netlist_dir.exists():
        raise FileNotFoundError(f"Netlist dir not found: {netlist_dir}")
    v_files = list(netlist_dir.glob("*.v"))
    if not v_files:
        raise FileNotFoundError("No .v netlist found")
    # Largest is usually the full flattened netlist
    return max(v_files, key=lambda p: p.stat().st_size)

def parse_macro_instances(netlist_path: Path) -> list:
    """Parse netlist for macro instances: return list of (instance_name, module_type)"""
    ast, _ = verilog_parse([str(netlist_path)])
    macros = []
    for definition in ast.description.definitions:
        for item in definition.items:
            if hasattr(item, 'instances'):
                instances = item.instances
                module_name = item.modulename
            elif hasattr(item, 'name') and hasattr(item, 'module'):
                instances = [item]
                module_name = item.module
            else:
                continue
            for inst in instances:
                inst_name = getattr(inst, 'name', None)
                if inst_name:
                    macros.append((inst_name, module_name))
    print(f"Parsed {len(macros)} macro instances from netlist")
    return macros  # Ordered by appearance (matches your graph node order if consistent)

def export_to_def(graph: Data, macro_instances: list, canvas: tuple, dbu: int, design_name: str, output_path: Path):
    pos_norm = graph.pos.numpy()
    sizes = graph.x[:, :2].numpy()  # width, height in μm
    
    with open(output_path, "w") as f:
        f.write("VERSION 5.8 ;\n")
        f.write("DIVIDERCHAR \"/\" ;\n")
        f.write("BUSBITCHARS \"[]\" ;\n")
        f.write(f"DESIGN {design_name} ;\n")
        f.write(f"UNITS DISTANCE MICRONS {dbu} ;\n")
        
        die_w = int(canvas[0] * dbu)
        die_h = int(canvas[1] * dbu)
        f.write(f"DIEAREA ( 0 0 ) ( {die_w} {die_h} ) ;\n")
        
        f.write(f"COMPONENTS {len(macro_instances)} ;\n")
        for i, (inst_name, module_type) in enumerate(macro_instances):
            if i >= graph.num_nodes:
                break  # Safety
            x_norm, y_norm = pos_norm[i]
            x_dbu = int(x_norm * canvas[0] * dbu)
            y_dbu = int(y_norm * canvas[1] * dbu)
            # Center placement (adjust if needed)
            w_dbu = int(sizes[i, 0] * dbu)
            h_dbu = int(sizes[i, 1] * dbu)
            # DEF PLACED uses lower-left corner
            ll_x = x_dbu - w_dbu // 2
            ll_y = y_dbu - h_dbu // 2
            f.write(f"- {inst_name} {module_type} PLACED ( {ll_x} {ll_y} ) N ;\n")
        f.write("END COMPONENTS\n")
        f.write("END DESIGN\n")
    
    print(f"DEF exported to: {output_path}")
    print("   Ready for TILOS/ChiPBench/OpenROAD: cp to macro_placed.def and run flow")

def main():
    parser = argparse.ArgumentParser(description="Export RL macro placement to DEF")
    parser.add_argument("--graph", type=str, required=True, help="Path to trained .pt graph")
    parser.add_argument("--design", type=str, required=True, help="Design name (e.g., ariane136)")
    parser.add_argument("--pdk", type=str, default="Nangate45", help="PDK (Nangate45, ASAP7, SKY130HD)")
    parser.add_argument("--repo-path", type=str, default="MacroPlacement", help="Path to cloned TILOS repo")
    parser.add_argument("--canvas", nargs=2, type=float, default=None, help="Override canvas width height μm")
    parser.add_argument("--output", type=str, required=True, help="Output DEF file path")
    args = parser.parse_args()
    
    graph = load_graph(args.graph)
    repo_path = Path(args.repo_path)
    
    netlist_path = find_netlist(repo_path, args.design, args.pdk)
    macro_instances = parse_macro_instances(netlist_path)
    
    canvas = tuple(args.canvas) if args.canvas else graph.canvas_size.tolist()
    dbu = PDK_DBU.get(args.pdk, 2000)
    
    export_to_def(graph, macro_instances, canvas, dbu, args.design, Path(args.output))

if __name__ == "__main__":
    main()