import os
from pyvis.network import Network

NODES = {
    "Crops": [
        ("Wheat", "#22C55E"),
        ("Rice", "#22C55E"),
        ("Cotton", "#22C55E"),
        ("Sugarcane", "#22C55E"),
        ("Mustard", "#22C55E"),
        ("Soybean", "#22C55E"),
        ("Pulses", "#22C55E")
    ],
    "Pests": [
        ("Aphids", "#EF4444"),
        ("Stem Borer", "#EF4444"),
        ("Bollworm", "#EF4444"),
        ("Armyworm", "#EF4444"),
        ("Whitefly", "#EF4444")
    ],
    "Diseases": [
        ("Yellow Rust", "#F59E0B"),
        ("Blast", "#F59E0B"),
        ("Blight", "#F59E0B"),
        ("Powdery Mildew", "#F59E0B"),
        ("Wilt", "#F59E0B")
    ],
    "Nutrients": [
        ("Nitrogen", "#3B82F6"),
        ("Phosphorus", "#3B82F6"),
        ("Potassium", "#3B82F6"),
        ("Zinc", "#3B82F6"),
        ("Boron", "#3B82F6")
    ],
    "Schemes": [
        ("PM-KISAN", "#8B5CF6"),
        ("PMFBY", "#8B5CF6"),
        ("KCC", "#8B5CF6"),
        ("eNAM", "#8B5CF6")
    ],
    "Seasons": [
        ("Kharif", "#06B6D4"),
        ("Rabi", "#06B6D4"),
        ("Zaid", "#06B6D4")
    ]
}

EDGES = [
    ("Wheat", "Yellow Rust", "susceptible_to"),
    ("Wheat", "Nitrogen", "requires"),
    ("Wheat", "Phosphorus", "requires"),
    ("Wheat", "Potassium", "requires"),
    ("Wheat", "Zinc", "requires"),
    ("Wheat", "Rabi", "grown_in"),
    ("Rice", "Stem Borer", "susceptible_to"),
    ("Rice", "Blast", "susceptible_to"),
    ("Rice", "Blight", "susceptible_to"),
    ("Rice", "Nitrogen", "requires"),
    ("Rice", "Zinc", "requires"),
    ("Rice", "Kharif", "grown_in"),
    ("Cotton", "Bollworm", "susceptible_to"),
    ("Cotton", "Whitefly", "susceptible_to"),
    ("Cotton", "Aphids", "susceptible_to"),
    ("Cotton", "Potassium", "requires"),
    ("Cotton", "Kharif", "grown_in"),
    ("Sugarcane", "Stem Borer", "susceptible_to"),
    ("Sugarcane", "Wilt", "susceptible_to"),
    ("Sugarcane", "Nitrogen", "requires"),
    ("Mustard", "Aphids", "susceptible_to"),
    ("Mustard", "Powdery Mildew", "susceptible_to"),
    ("Mustard", "Rabi", "grown_in"),
    ("Soybean", "Armyworm", "susceptible_to"),
    ("Soybean", "Phosphorus", "requires"),
    ("Soybean", "Kharif", "grown_in"),
    ("Pulses", "Wilt", "susceptible_to"),
    ("Pulses", "Phosphorus", "requires"),
    ("PM-KISAN", "Wheat", "supports_farmers_of"),
    ("PM-KISAN", "Rice", "supports_farmers_of"),
    ("PM-KISAN", "Cotton", "supports_farmers_of"),
    ("PM-KISAN", "Pulses", "supports_farmers_of"),
    ("PMFBY", "Wheat", "insures"),
    ("PMFBY", "Rice", "insures"),
    ("PMFBY", "Cotton", "insures"),
    ("PMFBY", "Mustard", "insures"),
    ("KCC", "Wheat", "provides_credit_for"),
    ("KCC", "Rice", "provides_credit_for"),
    ("KCC", "Cotton", "provides_credit_for"),
    ("eNAM", "Wheat", "market_platform_for"),
    ("eNAM", "Mustard", "market_platform_for"),
    ("eNAM", "Cotton", "market_platform_for")
]


def build_graph() -> Network:
    """Build interactive pyvis agricultural knowledge graph."""
    net = Network(
        height="520px",
        width="100%",
        bgcolor="#0A0F1E",
        font_color="#F9FAFB",
        directed=True
    )

    # Configure physics for repulsion layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "color": {"color": "rgba(156, 163, 175, 0.4)", "highlight": "#22C55E"},
        "smooth": {"type": "continuous"}
      }
    }
    """)

    # Add Nodes
    for category, node_list in NODES.items():
        for name, color in node_list:
            net.add_node(
                name,
                label=name,
                color=color,
                size=22,
                title=f"{category}: {name}"
            )

    # Add Edges
    for src, dst, rel in EDGES:
        net.add_edge(src, dst, title=rel, label=rel)

    return net


def get_crop_connections(crop_name: str) -> dict:
    """Get all connected pests, diseases, nutrients, schemes, and seasons for a given crop."""
    susceptible = []
    nutrients = []
    seasons = []
    schemes = []

    for src, dst, rel in EDGES:
        if src == crop_name:
            if rel == "susceptible_to":
                susceptible.append(dst)
            elif rel == "requires":
                nutrients.append(dst)
            elif rel == "grown_in":
                seasons.append(dst)
        elif dst == crop_name:
            if rel in ["supports_farmers_of", "insures", "provides_credit_for", "market_platform_for"]:
                if src not in schemes:
                    schemes.append(src)

    return {
        "crop": crop_name,
        "susceptible": susceptible,
        "nutrients": nutrients,
        "seasons": seasons,
        "schemes": schemes
    }
