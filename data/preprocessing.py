import os
import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import random
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Please install torch and torch-geometric: pip install torch torchvision torchaudio torch-geometric")

# Optional for real connectivity parsing
try:
    from pyverilog.vparser.parser import parse as verilog_parse
    from pyverilog.vparser.ast import Instance, PortArg, Identifier, InstanceList
    HAS_PYVERILOG = True
    print("PyVerilog detected: Real connectivity parsing enabled (pip install pyverilog if missing)")
except ImportError:
    HAS_PYVERILOG = False
    print("PyVerilog not found: Will use synthetic edges. Install with 'pip install pyverilog' for real macro connectivity")


# Hardcoded known information from TILOS repo README and papers (accurate as of repo state)
KNOWN_DESIGN_INFO = {
    "ariane136": {
        "macro_count": 136,
        "macro_type": "uniform 256x16-bit SRAM",
        "approx_macro_size_um": (50.0, 50.0),  # Approximate; actual depends on PDK layout (typical for similar SRAM in open PDKs)
        "canvas_size_um": (400.0, 400.0),  # Approximate die area to fit ~136 macros with spacing
        "stdcells_flops": "~20K",
        "note": "Matches Google Circuit Training benchmark; all macros identical"
    },
    "ariane133": {
        "macro_count": 133,
        "macro_type": "uniform 256x16-bit SRAM",
        "approx_macro_size_um": (50.0, 50.0),
        "canvas_size_um": (400.0, 400.0),
        "stdcells_flops": "~20K"
    },
    "MemPool_tile": {
        "macro_count": 20,
        "macro_type": "mixed 256x32-bit and 64x64-bit SRAMs",
        "approx_macro_size_um": (60.0, 60.0),  # Average approx
        "canvas_size_um": (300.0, 300.0),
        "stdcells_flops": "~18K"
    },
    "MemPool_group": {
        "macro_count": 324,
        "macro_type": "mixed varying SRAM sizes",
        "approx_macro_size_um": (60.0, 60.0),
        "canvas_size_um": (800.0, 800.0),
        "stdcells_flops": "~361K"
    },
    "NVDLA": {
        "macro_count": 128,
        "macro_type": "uniform 256x64-bit SRAM",
        "approx_macro_size_um": (70.0, 70.0),
        "canvas_size_um": (500.0, 500.0),
        "stdcells_flops": "~45K"
    },
    "BlackParrot": {
        "macro_count": 220,
        "macro_type": "diverse SRAM sizes",
        "approx_macro_size_um": (65.0, 65.0),  # Average
        "canvas_size_um": (600.0, 600.0),
        "stdcells_flops": "~214K"
    }
}
PDK_FOLDER_MAP = {
    "Nangate45": "NanGate45",   # Common user input → actual repo folder
    "ASAP7": "ASAP7",
    "SKY130HD": "SKY130HD"
}

def clone_repo(repo_root: Path):
    repo_url = "https://github.com/TILOS-AI-Institute/MacroPlacement.git"
    if not repo_root.exists():
        print(f"Cloning TILOS MacroPlacement repository to {repo_root}...")
        #subprocess.check_call(["git", "clone", repo_url, str(repo_root)])
    else:
        print(f"Repository already exists at {repo_root}. Pulling latest changes...")
        #subprocess.check_call(["git", "-C", str(repo_root), "pull"])

def collect_design_files(root: Path, design: str, pdk: str) -> Dict[str, List[str]]:
    pdk_folder = PDK_FOLDER_MAP.get(pdk, pdk)  # Map to actual repo folder name
    base = root.resolve()
    flows_base = base / "Flows" / pdk_folder / design
    
    files = {
        "rtl": [str(p) for p in (base / "Testcases" / design / "rtl").rglob("*.*v*")] if (base / "Testcases" / design / "rtl").exists() else [],
        "sv2v": [str(p) for p in (base / "Testcases" / design / "sv2v").rglob("*.*v*")] if (base / "Testcases" / design / "sv2v").exists() else [],
        "lef": [str(p) for p in (base / "Enablements" / pdk_folder / "lef").rglob("*.lef")],
        "lib": [str(p) for p in (base / "Enablements" / pdk_folder / "lib").rglob("*.lib")],
    }
    
    # Add potential netlist subdir explicitly
    netlist_dir = flows_base / "netlist"
    if netlist_dir.exists():
        files["netlist_candidates"] = [str(p) for p in netlist_dir.glob("*.v")]
    
    # Also scan full flows for any .v (fallback)
    files["flows_v"] = [str(p) for p in flows_base.rglob("*.v")]
    
    return {k: v for k, v in files.items() if v}

def parse_lef_for_macros(lef_paths: List[str]) -> Dict[str, Tuple[float, float]]:
    macro_sizes = {}
    for path in lef_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Fixed regex: Removed the unbalanced ')' after '.*?'
        # Changed to [\s\S]*? for reliable multiline non-greedy matching (safer than .*? with re.S)
        # Simplified name capture to [\w]+ (LEF macro names are typically alphanumeric/underscore)
        # Made number capture more robust ([0-9.E+-]+)
        pattern = r'MACRO\s+([\w]+)[\s\S]*?SIZE\s+([0-9.E+-]+)\s+BY\s+([0-9.E+-]+)[\s\S]*?END\s+\1'
        
        matches = re.finditer(pattern, content, re.I)  # re.I for case-insensitive (MACRO/end)
        
        for match in matches:
            name = match.group(1)
            try:
                w = float(match.group(2))
                h = float(match.group(3))
                area = w * h
                # Keep threshold for likely macros (large area)
                if area > 50:
                    macro_sizes[name] = (w, h)
                    print(f"  Parsed macro: {name} -> {w} x {h} μm (area: {area:.1f})")
            except ValueError:
                pass
    return macro_sizes
def find_netlist_and_def(file_dict: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """Find synthesized netlist (.v) and a DEF file with placements"""
    netlist_candidates = []
    # Priority 1: Explicit netlist/ subdir
    if "netlist_candidates" in file_dict:
        netlist_candidates.extend(file_dict["netlist_candidates"])
    # Priority 2: Any .v in flows
    if "flows_v" in file_dict:
        netlist_candidates.extend(file_dict["flows_v"])
    
    # for path in file_dict.get("flows", []) + file_dict.get("netlist", []):
    #     if path.endswith(".v") or path.endswith(".gv") and path not in netlist_candidates:
    #         netlist_candidates.append(path)

    # Pick largest (likely the full flattened post-synth netlist)
    netlist_path = max(netlist_candidates, key=os.path.getsize) if netlist_candidates else None
    def_candidates = [p for p in file_dict.get("flows", []) if p.endswith(".def")]
    def_path = def_candidates[0] if def_candidates else None
    if not netlist_candidates:
        return None, def_path
    print(f"Selected netlist: {netlist_path} (size: {os.path.getsize(netlist_path)/1e6:.2f} MB)")

    return netlist_path, def_path

def parse_real_connectivity_from_netlist(netlist_path: str, known_macro_count: int, approx_size: Tuple[float, float] = (50.0, 50.0)) -> Tuple[List[Tuple[float, float]], torch.Tensor, int]:
    """Enhanced parser: Auto-detect macro module by instance count closest to known_macro_count"""
    if not HAS_PYVERILOG:
        raise RuntimeError("PyVerilog required")

    ast, _ = verilog_parse([netlist_path])
    print(f"Parsed netlist AST from {netlist_path}")

    # First pass: Collect all instantiated module names and their instance counts
    module_instance_counts = defaultdict(int)
    all_instances = []

    for definition in ast.description.definitions:
        if not isinstance(definition, ModuleDef):
            continue
        for item in definition.items:
            if isinstance(item, InstanceList):
                module_name = item.modulename
                module_instance_counts[module_name] += len(item.instances)
                for inst in item.instances:
                    all_instances.append((module_name, inst.name if hasattr(inst, 'name') else "unnamed"))
            elif isinstance(item, Instance):
                module_name = item.module
                module_instance_counts[module_name] += 1
                all_instances.append((module_name, item.name if hasattr(item, 'name') else "unnamed"))

    # Debug: Print top modules by instance count
    print("\nTop instantiated modules by count (likely macro has high count matching known ~{}):".format(known_macro_count))
    sorted_modules = sorted(module_instance_counts.items(), key=lambda x: x[1], reverse=True)
    for module_name, count in sorted_modules[:10]:  # Top 10
        print(f"  {module_name}: {count} instances")

    # Auto-detect macro module: the one with count closest to known_macro_count (tolerance ±20%)
    candidates = []
    tolerance = known_macro_count * 0.2
    for module_name, count in module_instance_counts.items():
        if abs(count - known_macro_count) <= tolerance or count > known_macro_count * 0.5:  # Loose for safety
            candidates.append((module_name, count, abs(count - known_macro_count)))

    if not candidates:
        raise ValueError("No module found with instance count near known macro count (~{})".format(known_macro_count))

    # Pick best (closest count)
    candidates.sort(key=lambda x: x[2])
    macro_module_name, detected_count, _ = candidates[0]
    print(f"\nAuto-selected macro module: '{macro_module_name}' with {detected_count} instances (target: {known_macro_count})")

    # Second pass: Collect macro instances and connections
    macro_id = 0
    inst_to_id: Dict[str, int] = {}
    net_to_macros: defaultdict[str, List[int]] = defaultdict(list)

    for definition in ast.description.definitions:
        if not isinstance(definition, ModuleDef):
            continue
        for item in definition.items:
            if isinstance(item, InstanceList):
                if item.modulename == macro_module_name:
                    for instance in item.instances:
                        inst_name = instance.name
                        if inst_name:
                            inst_to_id[inst_name] = macro_id
                            macro_id += 1
                            # Connect ports
                            for portarg in instance.portlist:
                                if portarg.argname is not None:
                                    net_name = str(portarg.argname)
                                    net_to_macros[net_name].append(inst_to_id[inst_name])
            elif isinstance(item, Instance):
                if item.module == macro_module_name:
                    inst_name = item.name
                    if inst_name:
                        inst_to_id[inst_name] = macro_id
                        macro_id += 1
                        for portarg in item.portlist:
                            if portarg.argname is not None:
                                net_name = str(portarg.argname)
                                net_to_macros[net_name].append(inst_to_id[inst_name])

    # Build edges
    edge_index = torch.empty((2, 0), dtype=torch.long)
    total_edges = 0
    for macros_on_net in net_to_macros.values():
        if len(macros_on_net) > 1:
            for i in range(len(macros_on_net)):
                for j in range(i + 1, len(macros_on_net)):
                    a, b = macros_on_net[i], macros_on_net[j]
                    edge_index = torch.cat([edge_index, torch.tensor([[a, b], [b, a]])], dim=1)
                    total_edges += 1

    print(f"Built real macro graph: {macro_id} nodes, {total_edges} undirected edges")

    # Uniform sizes (since Ariane macros are identical)
    per_macro_sizes = [approx_size] * macro_id

    return per_macro_sizes, edge_index, macro_id

# In build_pyg_graph, update call:

def build_pyg_graph(design: str, design_info: dict, netlist_path: str) -> Data:
    # In build_pyg_graph, update call:
    use_real = True
    edge_index = None
    num_macros = design_info["macro_count"]
    if use_real:
        try:
            per_macro_sizes, edge_index, detected_num = parse_real_connectivity_from_netlist(
                netlist_path, design_info["macro_count"], design_info["approx_macro_size_um"]
            )
            num_macros = detected_num
            # Override known count if detection better
            if abs(detected_num - design_info["macro_count"]) < design_info["macro_count"] * 0.1:
                print("Real macro count matches known!")
        except Exception as e:
            print(f"Real parsing failed ({e}) - fallback to synthetic")
            use_real = False

    macro_w, macro_h = design_info["approx_macro_size_um"]
    canvas_w, canvas_h = design_info["canvas_size_um"]

    # Node features: [width, height, area, normalized_w, normalized_h]
    area = macro_w * macro_h
    node_features = torch.tensor([[macro_w, macro_h, area, macro_w / canvas_w, macro_h / canvas_h]] * num_macros, dtype=torch.float)

    # Initial random positions (normalized 0-1, later scale to canvas in RL env)
    pos = torch.rand(num_macros, 2)

    # Synthetic connectivity: Erdos-Renyi random graph (p=0.05 for sparse but connected)
    # Real connectivity requires full netlist parsing post-synthesis
    print("Edge index: ", edge_index)
    edge_index = torch.empty(2, 0, dtype=torch.long) if not edge_index else edge_index
    if num_macros > 1:
        prob = 0.05
        for i in range(num_macros):
            for j in range(i + 1, num_macros):
                if random.random() < prob:
                    edge_index = torch.cat([edge_index, torch.tensor([[i, j], [j, i]])], dim=1)

    # If no edges, add a minimal connected structure (chain)
    if edge_index.shape[1] == 0 and num_macros > 1:
        for i in range(num_macros - 1):
            edge_index = torch.cat([edge_index, torch.tensor([[i, i+1], [i+1, i]])], dim=1)

    graph = Data(
        x=node_features,
        edge_index=edge_index,
        pos=pos,  # normalized
        num_nodes=num_macros
    )

    return graph

def main():
    parser = argparse.ArgumentParser(description="Complete preprocessing for TILOS MacroPlacement dataset: metadata + PyG graph")
    parser.add_argument("--repo-root", type=str, default="MacroPlacement", help="Directory for cloned repo")
    parser.add_argument("--design", type=str, required=True, choices=list(KNOWN_DESIGN_INFO.keys()))
    parser.add_argument("--pdk", type=str, default="Nangate45", choices=["Nangate45", "ASAP7", "SKY130HD"])
    parser.add_argument("--output-dir", type=str, default="preprocessed_dataset")
    parser.add_argument("--build-graph", action="store_true", help="Build and save PyTorch Geometric graph (.pt)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    clone_repo(repo_root)

    print(f"\nProcessing design: {args.design} with PDK: {args.pdk}")

    file_dict = collect_design_files(repo_root, args.design, args.pdk)
    lef_macros = parse_lef_for_macros(file_dict.get("lef", []))
    netlist_path, def_path = find_netlist_and_def(file_dict)

    if netlist_path:
        print(f"Found potential netlist: {netlist_path}")
    else:
        print("No synthesized netlist found. Run Flows first for real connectivity.")

    design_info = KNOWN_DESIGN_INFO[args.design].copy()
    if lef_macros:
        print(f"Parsed {len(lef_macros)} potential macro(s) from LEF (large area):")
        for name, (w, h) in lef_macros.items():
            print(f"  {name}: {w} x {h} um")
        # Use first parsed macro size if available (override approx)
        if lef_macros:
            first_macro = next(iter(lef_macros.values()))
            design_info["approx_macro_size_um"] = first_macro

    # summary = {
    #     "design": args.design,
    #     "pdk": args.pdk,
    #     "known_info": design_info,
    #     "parsed_lef_macros": {name: {"width": w, "height": h} for name, (w, h) in lef_macros.items()},
    #     "files": file_dict,
    #     "note": "Graph is synthetic (random edges). For real connectivity, run repo Flows + clustering, then extend parser for clustered netlist."
    # }
    summary = {
        "design": args.design,
        "pdk": args.pdk,
        "known_info": design_info,
        "parsed_lef_macros": {name: {"width": w, "height": h} for name, (w, h) in lef_macros.items()},
        "source_files": {
            "primary_rtl": file_dict.get("rtl", [None])[0],  # Likely top-level (e.g., ariane.v)
            "rtl_paths": file_dict.get("rtl", []),          # Full list
            "sv2v_paths": file_dict.get("sv2v", []),         # Converted Verilog (critical for synthesis)
            "note": "Ariane designs often use sv2v-converted files for Yosys/OpenROAD compatibility."
        },
        "netlist_path": netlist_path,  # If available from flows
        "files": file_dict,  # Keep raw for completeness
        "note": "Graph is synthetic unless real netlist parsed. RTL/sv2v paths preserved for reproducibility."
    }

    out_dir = Path(args.output_dir) / args.design / args.pdk
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path =  f"{out_dir}/metadata-{args.design}-{args.pdk}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetadata saved to: {summary_path}")

    if args.build_graph:
        print("\nBuilding PyTorch Geometric graph (synthetic but usable for RL testing)...")
        graph = build_pyg_graph(args.design, design_info, netlist_path if netlist_path else "")
        graph_path = out_dir / f"{args.design}_{args.pdk}_graph.pt"
        save_data = {
            "graph": graph,
            "metadata": summary
        }
        torch.save(save_data, graph_path)
        print(f"GNN-ready graph saved to: {graph_path}")
        print(f"   Nodes (macros): {graph.num_nodes}")
        print(f"   Edges: {graph.num_edges // 2 if graph.edge_index.numel() > 0 else 0} (undirected)")
        print(f"   Feature dim: {graph.x.shape[1] if graph.num_nodes > 0 else 0}")

    print("\nDone! For real connectivity/hypergraph:")
    print("   1. Run the provided Flows in the repo to synthesize + generate DEF/netlist.")
    print("   2. Use CodeElements/Clustering (hMETIS) + FormatTranslators to get Circuit Training .pbtxt.")
    print("   3. Extend this script with a .pbtxt parser (or use google-research/circuit_training code).")

if __name__ == "__main__":
    main()