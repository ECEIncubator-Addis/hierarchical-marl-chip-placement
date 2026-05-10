import argparse
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch.serialization import safe_globals
from torch_geometric.utils import to_networkx
import networkx as nx
def visualize_tensor(tensor, name="tensor", batch_index=0):
    """
    Visualize a torch tensor based on its dimensionality.
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"Skipping {name}: not a tensor")
        return

    t = tensor.detach().cpu()

    # Handle batched tensors
    if t.ndim >= 3:
        t = t[batch_index]

    if t.ndim == 1:
        plt.figure()
        plt.plot(t.numpy())
        plt.title(f"{name} (1D)")
        plt.xlabel("Index")
        plt.ylabel("Value")

    elif t.ndim == 2:
        plt.figure()
        plt.imshow(t.numpy(), aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title(f"{name} (2D heatmap)")

    elif t.ndim == 3:
        # Assume (C, H, W) or (H, W, C)
        if t.shape[0] in (1, 3):
            img = t.permute(1, 2, 0).numpy()
        else:
            img = t.numpy()

        plt.figure()
        plt.imshow(img)
        plt.title(f"{name} (image)")
        plt.axis("off")

    else:
        print(f"Cannot visualize {name}: unsupported shape {t.shape}")
        return

    plt.tight_layout()
    plt.show()


def visualize_graph(graph: Data, name="graph"):
    assert isinstance(graph, Data)

    # Use physical macro positions if available
    pos = None
    if hasattr(graph, "pos") and graph.pos is not None:
        pos_arr = graph.pos.detach().cpu().numpy()
        pos = {i: pos_arr[i] for i in range(graph.num_nodes)}

    G = to_networkx(graph, to_undirected=True)

    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos=pos,
        node_size=100,
        alpha=0.8,
        with_labels=False,
    )

    plt.title(f"{name}: {graph.num_nodes} macros, {graph.num_edges} edges")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.axis("equal")
    plt.show()


def print_metadata(meta: dict, indent=0):
    pad = " " * indent
    for k, v in meta.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_metadata(v, indent + 2)
        else:
            print(f"{pad}{k}: {v}")


def visualize_pt(data):
    if not isinstance(data, dict):
        raise TypeError("Expected top-level dict")

    if "graph" in data:
        visualize_graph(data["graph"], name="Macro Placement Graph")

    if "metadata" in data:
        print("\nMetadata:")
        print_metadata(data["metadata"])


def visualize_pt_file(data, batch_index=0):

    if isinstance(data, torch.Tensor):
        visualize_tensor(data, name="data", batch_index=batch_index)

    elif isinstance(data, dict):
        for key, value in data.items():
            visualize_tensor(value, name=key, batch_index=batch_index)

    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            visualize_tensor(item, name=f"item_{i}", batch_index=batch_index)

    else:
        raise TypeError(f"Unsupported .pt content type: {type(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization of a built design graph")
    parser.add_argument("--path", type=str, default="", help="path to the .pt file to visualize")
    with safe_globals([Data]):
        data = torch.load(
            parser.parse_args().path,
            map_location="cpu",
            weights_only=False
        )
    print(type(data))
    print(data)
    visualize_pt(data)
    # for key, value in data.items():
    #     visualize_tensor(value, name=key)