from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def export_pipeline_graph(output_path: str = "docs/pipeline_graph.png") -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("Planner", "Retriever"),
            ("Retriever", "Analyst"),
            ("Analyst", "Critic"),
            ("Critic", "Writer"),
        ]
    )

    pos = {
        "Planner": (0, 0),
        "Retriever": (1, 0),
        "Analyst": (2, 0),
        "Critic": (3, 0),
        "Writer": (4, 0),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 3))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="#dbeafe",
        node_size=2500,
        arrows=True,
        arrowsize=20,
        font_size=11,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    export_pipeline_graph()
