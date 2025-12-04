from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Union, Any
import uuid
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import networkx as nx
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

if TYPE_CHECKING:
    from .pipeline import Workflow
    from .model import Hierarchy, Node
    from .types import NumericType, Number, TFN, Crisp

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

def _check_pandas_availability():
    """Helper function to raise an error if pandas is not installed."""
    if not _PANDAS_AVAILABLE:
        raise ImportError("Excel/CSV export functionality requires the 'pandas' and 'openpyxl' libraries. "
                        "Please install them using: pip install pandas openpyxl")

# ==============================================================================
# 1. HTML HIERARCHY DIAGRAM
# ==============================================================================


def _render_expandable_node(
    node: Node,
    consistency_data: Dict[str, Any] | None = None,
    is_alt: bool = False,
    alt_obj=None,
    is_root: bool = False,
    consistency_method: str = "centroid"
) -> str:
    """
    Renders a single, generic, expandable node box, now including consistency info.
    """
    body_id = 'node-body-' + str(uuid.uuid4())[:8]
    details_html = "<ul>"

    if is_alt:
        score_val = alt_obj.overall_score
        score_str = f"{score_val.defuzzify(method=consistency_method):.4f}" if score_val is not None else 'Not Calculated'
        details_html += f"<li><b>Final Score:</b> {score_str}</li>"

        if alt_obj.performance_scores:
            details_html += "<li><b>Performance Scores:</b><ul>"
            for leaf_id, score in alt_obj.performance_scores.items():
                if hasattr(score, 'defuzzify'):
                    details_html += f"<li>{leaf_id}: {score.defuzzify(method=consistency_method):.3f}</li>"
                else:
                    details_html += f"<li>{leaf_id}: {score:.3f}</li>"
            details_html += "</ul></li>"

        header_text = alt_obj.name

    else: # Criterion or Goal Node
        if is_root:
            try:
                model = node.get_model()
                details_html += f"<li><b>Number Type:</b> {model.number_type.__name__}</li>"
                details_html += f"<li><b>Alternatives:</b> {len(model.alternatives)}</li>"
                details_html += f"<li><b>Weight Derivation:</b> {getattr(model, 'last_used_derivation_method', 'N/A')}</li>"
                if getattr(model, 'last_used_aggregation_method', 'N/A'):
                    details_html += f"<li><b>Aggregation Method:</b> {getattr(model, 'last_used_aggregation_method', 'N/A')}</li>"
                if getattr(model, 'last_used_ranking_defuzz_method', 'N/A'):
                    details_html += f"<li><b>Defuzzification Method:</b> {getattr(model, 'last_used_ranking_defuzz_method', 'N/A')}</li>"
            except (RuntimeError, AttributeError) as e:
                details_html += f"<li>Error getting model details: {e}</li>"

            header_text = f"Goal: {node.id}"
        else:
            local_w_str = f"{node.local_weight.defuzzify(method=consistency_method):.3f}" if node.local_weight else 'N/A'
            global_w_str = f"{node.global_weight.defuzzify(method=consistency_method):.4f}" if node.global_weight else 'N/A'
            parent_id_str = node.parent.id if node.parent else 'None'

            details_html += f"<li><b>Description:</b> {node.description or 'N/A'}</li>"
            details_html += f"<li><b>Parent:</b> {parent_id_str}</li>"
            details_html += f"<li><b>Local Weight:</b> {local_w_str}</li>"
            details_html += f"<li><b>Global Weight:</b> {global_w_str}</li>"

            if consistency_data and node.id in consistency_data:
                cons_info = consistency_data[node.id]
                details_html += "<li><b>Consistency:</b><ul>"
                for name, value in cons_info.items():
                    if 'status' in name or 'threshold' in name or name in ['matrix_size', 'is_consistent']:
                        continue
                    details_html += f"<li>{name.replace('_', ' ').title()}: {value}</li>"
                details_html += "</ul></li>"

            header_text = node.id

    details_html += "</ul>"

    return f"""
    <div class="ahp-node" onclick="toggleAHPNode('{body_id}', event)">
        <div class="ahp-node-header">{header_text}</div>
        <div class="ahp-node-body" id="{body_id}">
            {details_html}
        </div>
    </div>
    """

def _render_criterion_tree(node: 'Node', consistency_data: Dict[str, Any]) -> str:
    """Renders the tree for a single criterion, passing consistency data down."""
    if node.is_leaf:
        return _render_expandable_node(node, consistency_data)

    parent_html = f'<div class="tree-parent">{_render_expandable_node(node, consistency_data)}</div>'
    children_html = "".join([_render_expandable_node(child, consistency_data) for child in node.children])
    children_container_html = f'<div class="tree-children">{children_html}</div>'

    return f'<div class="criterion-tree-block">{parent_html}{children_container_html}</div>'

def display_tree_hierarchy(model: 'Hierarchy', filename: str | None = None, consistency_method: str = "centroid"):
    """
    Generates and displays/saves a static, tree-diagram-style HTML representation
    of the AHP hierarchy, matching the final photoshopped layout.
    """
    from IPython.display import display, HTML
    from .consistency import Consistency

    try:
        if model.root.children and model.root.children[0].local_weight is None:
             model.calculate_weights()
    except Exception as e:
         print(f"Warning: Could not auto-calculate weights for display. Run `model.calculate_weights()` first. Error: {e}")

    try:
        if model.root.global_weight is None: model.calculate_weights()
    except Exception as e: print(f"Warning: Could not auto-calculate weights for display. Error: {e}")

    consistency_results = Consistency.check_model_consistency(model)

    # --- CSS AND JAVASCRIPT ---
    css_js = """
    <style>
        .ahp-viz-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            /* Add overflow-x here to contain the entire visualization */
            overflow-x: auto;
            padding-bottom: 20px; /* Add some padding for the scrollbar */
        }
        .ahp-stage {
            border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.5em;
            margin: 20px 0; background-color: #f7f9fc;

            /* This makes the stage container size itself to its content */
            display: inline-block;
            min-width: 100%; /* Ensures it's at least as wide as the page */
            box-sizing: border-box; /* Includes padding and border in the width calculation */
        }
        .ahp-stage-title {
            text-align: center; font-weight: 600; font-size: 1.1em;
            color: #5f6368; margin-bottom: 25px; text-transform: uppercase; letter-spacing: 1px;
        }
        .ahp-connector { height: 30px; display: flex; justify-content: center; align-items: center; }
        .ahp-connector::after { content: ''; display: block; width: 2px; height: 100%; background-color: #bdc1c6; }

        /* Node Box Styling */
        .ahp-node {
            background: #fff; border: 1px solid #dadce0; border-radius: 6px;
            display: inline-block; min-width: 120px; text-align: center;
            box-shadow: 0 1px 2px rgba(60,64,67,0.1); cursor: pointer;
        }
        .ahp-node-header { padding: 8px 12px; font-weight: bold; color: #202124; }
        .ahp-node-body {
            font-size: 0.85em; color: #333; max-height: 0;
            overflow: hidden; transition: max-height 0.4s ease-in-out; text-align: left;
            border-top: 1px solid #f1f3f4;
        }
        .ahp-node-body ul { list-style-type: none; padding: 10px; margin: 0; }
        .ahp-node-body ul ul { padding-left: 15px; padding-top: 5px; }

        /* Criteria Stage Layout */
        .criteria-container {
            display: flex;
            justify-content: flex-start; /* Align to the start instead of space-around */
            align-items: flex-start;
            gap: 40px; /* More space between criteria trees */
            padding: 10px; /* Add some internal padding */
        }
        .criterion-tree-block { display: flex; flex-direction: column; align-items: center; }

        /* Tree Connecting Lines */
        .tree-parent { position: relative; padding-bottom: 20px; }
        .tree-parent::after {
            content: ''; position: absolute; bottom: 0; left: 50%;
            transform: translateX(-50%); width: 2px; height: 20px; background-color: #bdc1c6;
        }
        .tree-children {
            display: flex; justify-content: center; position: relative; padding-top: 20px;
            gap: 15px; /* Space between sibling sub-criteria */
        }
        .tree-children::before {
            content: ''; position: absolute; top: 0; left: 10%; right: 10%;
            height: 2px; background-color: #bdc1c6;
        }
        .tree-children .ahp-node { position: relative; }
        .tree-children .ahp-node::before {
            content: ''; position: absolute; top: -20px; left: 50%;
            transform: translateX(-50%); width: 2px; height: 20px; background-color: #bdc1c6;
        }

        /* Alternatives Stage Layout */
        .ahp-alternatives-container { display: flex; flex-direction: column; align-items: center; gap: 15px; }
    </style>
    <script>
        function toggleAHPNode(elementId, event) {
            event.stopPropagation();
            const body = document.getElementById(elementId);
            if (body.style.maxHeight && body.style.maxHeight !== '0px') {
                body.style.maxHeight = '0px';
            } else {
                body.style.maxHeight = body.scrollHeight + 'px';
            }
        }
    </script>
    """

    # --- HTML GENERATION ---

    # 1. Goal Stage
    goal_html = f"""
    <div class="ahp-stage">
        <div class="ahp-stage-title">Goal</div>
        <div style="display: flex; justify-content: center;">
            {_render_expandable_node(model.root, consistency_results, is_root=True)}
        </div>
    </div>
    """

    # 2. Criteria Stage
    criteria_trees_html = "".join([_render_criterion_tree(crit_node, consistency_results) for crit_node in model.root.children])
    criteria_html = f"""
    <div class="ahp-stage">
        <div class="ahp-stage-title">Criteria & Sub-Criteria</div>
        <div class="criteria-container">
            {criteria_trees_html}
        </div>
    </div>
    """

    # 3. Alternatives Stage
    alternatives_html = ""
    if model.alternatives:
        ranked_alts = sorted(model.alternatives, key=lambda alt: alt.overall_score.defuzzify(method=consistency_method) if alt.overall_score else -1, reverse=True)
        alt_html_parts = [_render_expandable_node(None, is_alt=True, alt_obj=alt) for alt in ranked_alts]

        alternatives_html = f"""
        <div class="ahp-stage">
            <div class="ahp-stage-title">Alternatives</div>
            <div class="ahp-alternatives-container">
                {''.join(alt_html_parts)}
            </div>
        </div>
        """

    connector = '<div class="ahp-connector"></div>'
    full_html_body = f'{goal_html}{connector}{criteria_html}{connector}{alternatives_html}' if model.alternatives else f'{goal_html}{connector}{criteria_html}'
    full_document = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>AHP Diagram</title>{css_js}</head>
    <body><div class="ahp-viz-container">{full_html_body}</div></body></html>
    """

    if filename:
        try:
            with open(filename, 'w', encoding='utf-8') as f: f.write(full_document)
            print(f"✅ Hierarchy visualization saved to: {filename}")
        except Exception as e: print(f"❌ Error saving file: {e}")
    else:
        try:
            from IPython.display import display, HTML
            display(HTML(full_document))
        except ImportError:
            print("Could not display inline. Provide a `filename` to save as an HTML file.")

def format_model_summary(model: Hierarchy, alternative_name: str) -> str:
    """
    Generates a structured string summary of the entire evaluation model
    for a single alternative, similar to the presentation slide.

    Args:
        model: The fully calculated Hierarchy instance.
        alternative_name: The name of the alternative to display scores for.

    Returns:
        A formatted string representing the model summary.
    """
    alt = next((a for a in model.alternatives if a.name == alternative_name), None)
    if alt is None: return f"Error: Alternative '{alternative_name}' not found."
    if alt.overall_score is None: return f"Error: Scores not calculated."

    lines = []
    header = f" HIERARCHICAL EVALUATION SUMMARY for: {alt.name} "
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    final_score = alt.overall_score
    final_val = final_score.defuzzify() if hasattr(final_score, 'defuzzify') else final_score
    lines.append(f"\nFINAL OVERALL SCORE: {final_val:.4f}\n")

    for crit_node in model.root.children:
        crit_weight = crit_node.local_weight.defuzzify() if crit_node.local_weight else 0.0
        crit_g_weight = crit_node.global_weight.defuzzify() if crit_node.global_weight else 0.0

        if crit_node.id in alt.node_scores:
            c_score_obj = alt.node_scores[crit_node.id]
            crit_score = c_score_obj.defuzzify() if hasattr(c_score_obj, 'defuzzify') else c_score_obj
        else:
            crit_score = 0.0

        crit_contribution = crit_weight * crit_score

        lines.append(f"--- Criterion: {crit_node.id} ({crit_node.description or ''}) ---")
        lines.append(f"  - Global Weight: {crit_g_weight:.4f}")
        lines.append(f"  - Aggregated Score: {crit_score:.4f}")
        lines.append(f"  - Contribution: {crit_contribution:.4f}\n")

        if not crit_node.is_leaf:
            lines.append("    Sub-criteria Breakdown:")
            for sub_crit_node in crit_node.children:
                sc_loc = sub_crit_node.local_weight.defuzzify()
                sc_glob = sub_crit_node.global_weight.defuzzify()

                raw_perf = alt.performance_scores.get(sub_crit_node.id)
                if raw_perf is None:
                    perf_score = 0.0
                elif hasattr(raw_perf, 'defuzzify'):
                    perf_score = raw_perf.defuzzify()
                else:
                    perf_score = float(raw_perf) # It's already a float

                lines.append(f"      - {sub_crit_node.id}:")
                lines.append(f"          Local W: {sc_loc:.3f} | Global W: {sc_glob:.4f}")
                lines.append(f"          Perf Score: {perf_score:.4f}")
        lines.append("-" * 40)

    return "\n".join(lines)


# ==============================================================================
# 2. MATPLOTLIB PLOTTING FUNCTIONS
# ==============================================================================

def _check_plotting_availability():
    """Helper function to raise an error if plotting libraries are not installed."""
    if not _PLOT_AVAILABLE:
        raise ImportError("Plotting functionality requires matplotlib and networkx. "
                          "Please install them using: pip install matplotlib networkx")

def plot_weights(model: Hierarchy, parent_node_id: str, figsize=(10, 6)) -> 'plt.Figure':
    """
    Plots the local weights of the children of a given parent node.

    Args:
        model: The Hierarchy (AHPModel) instance.
        parent_node_id: The ID of the parent node whose children's weights to plot.
        figsize: The size of the figure.

    Returns:
        The matplotlib Figure object.
    """
    _check_plotting_availability()

    parent_node = model._find_node(parent_node_id)
    if not parent_node or parent_node.is_leaf:
        raise ValueError(f"Node '{parent_node_id}' is not a valid parent or was not found.")

    children = parent_node.children
    labels = [child.id for child in children]
    weights = [child.local_weight for child in children]

    crisp_weights = [w.defuzzify() for w in weights]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, crisp_weights, color=plt.cm.viridis(np.linspace(0, 1, len(labels))))

    ax.set_ylabel('Weight (Defuzzified)')
    ax.set_title(f'Local Weight Distribution for Children of "{parent_node_id}"')
    ax.set_xticklabels(labels, rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    fig.tight_layout()
    return fig

def plot_final_rankings(model: Hierarchy, figsize=(10, 6)) -> 'plt.Figure':
    """
    Plots the final rankings of the alternatives with their crisp scores.

    Args:
        model: The fully calculated Hierarchy (AHPModel) instance.
        figsize: The size of the figure.

    Returns:
        The matplotlib Figure object.
    """
    _check_plotting_availability()

    rankings = model.get_rankings()

    alt_names = [r[0] for r in rankings]
    scores = [r[1] for r in rankings]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(alt_names, scores, color=plt.cm.plasma(np.linspace(0.4, 0.9, len(scores))))

    ax.set_xlabel('Final Score')
    ax.set_ylabel('Alternative')
    ax.set_title('Final Alternative Rankings')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.invert_yaxis()

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{scores[i]:.4f}', va='center')

    fig.tight_layout()
    return fig

def plot_sensitivity_analysis(
    model: Hierarchy,
    parent_node_id: str,
    criterion_to_vary_id: str,
    alternative_name: str,
    figsize=(12, 7)
) -> 'plt.Figure':
    """
    Performs a simple sensitivity analysis by varying the local weight of one criterion
    and observing the effect on a single alternative's final score.

    Args:
        model: The calculated Hierarchy (AHPModel) instance.
        parent_node_id: The parent of the criterion being varied.
        criterion_to_vary_id: The ID of the criterion whose weight will be varied.
        alternative_name: The name of the alternative to track.
        figsize: The size of the figure.

    Returns:
        The matplotlib Figure object.
    """
    _check_plotting_availability()

    parent_node = model._find_node(parent_node_id)
    if not parent_node: raise ValueError(f"Parent '{parent_node_id}' not found.")

    criterion_to_vary = model._find_node(criterion_to_vary_id)
    if not criterion_to_vary: raise ValueError(f"Criterion '{criterion_to_vary_id}' not found.")

    original_weights = {child.id: child.local_weight for child in parent_node.children}
    original_alt_scores = {a.name: a.overall_score for a in model.alternatives}

    orig_w_obj = original_weights[criterion_to_vary_id]
    original_crisp_weight = orig_w_obj.defuzzify() if hasattr(orig_w_obj, 'defuzzify') else orig_w_obj

    weight_range = np.linspace(0.01, 0.99, 50)
    final_scores = []

    try:
        for new_weight in weight_range:
            criterion_to_vary.local_weight = model.number_type.from_normalized(new_weight)

            siblings = [c for c in parent_node.children if c.id != criterion_to_vary_id]
            orig_sib_sum = sum(original_weights[s.id].defuzzify() for s in siblings)

            remaining_weight = 1.0 - new_weight

            if orig_sib_sum > 1e-9:
                for sib in siblings:
                    orig_val = original_weights[sib.id].defuzzify()
                    ratio = orig_val / orig_sib_sum
                    new_sib_val = ratio * remaining_weight
                    sib.local_weight = model.number_type.from_normalized(new_sib_val)
            else:
                for sib in siblings:
                    sib.local_weight = model.number_type.from_normalized(remaining_weight / len(siblings))

            model._recalculate_global_weights()
            model.calculate_alternative_scores()

            target_alt = next(a for a in model.alternatives if a.name == alternative_name)
            s_val = target_alt.overall_score
            val = s_val.defuzzify() if hasattr(s_val, 'defuzzify') else s_val
            final_scores.append(val)

    finally:
        for child in parent_node.children:
            child.local_weight = original_weights[child.id]

        model._recalculate_global_weights()
        model.calculate_alternative_scores()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(weight_range, final_scores, marker='.', label=f"Score for {alternative_name}")
    ax.axvline(x=original_crisp_weight, color='r', linestyle='--', label=f'Original Weight ({original_crisp_weight:.3f})')

    ax.set_xlabel(f"Local Weight of '{criterion_to_vary_id}'")
    ax.set_ylabel(f"Final Score for '{alternative_name}'")
    ax.set_title(f"Sensitivity Analysis for Criterion '{criterion_to_vary_id}'")
    ax.legend()
    ax.grid(True, alpha=0.5)

    return fig


# ==============================================================================
# 3. REPORTS
# ==============================================================================

def generate_matrix_report(
    node_with_matrix: Node,
    model: Hierarchy,
    derivation_method: str = 'geometric_mean',
    consistency_method: str = 'centroid'
) -> str:
    """
    Generates a detailed, publication-ready text report for a single
    pairwise comparison matrix.

    This report includes the formatted matrix, derived weights, and a full
    breakdown of consistency metrics (CI, CR, RI).

    Args:
        node_with_matrix: The parent Node whose children's comparison matrix
                          is being reported.
        model: The main Hierarchy object, needed to access the number_type.
        derivation_method: The method used to derive weights.
        consistency_method: The defuzzification method used for consistency checks.

    Returns:
        A formatted string containing the complete report for one matrix.
    """
    from .weight_derivation import derive_weights
    from .consistency import Consistency

    node = node_with_matrix
    matrix = node.comparison_matrix
    if matrix is None: return ""

    # If it's a leaf node, the items are Alternatives, otherwise Children
    if node.is_leaf and model.alternatives:
        items = [alt.name for alt in model.alternatives]
    else:
        items = [child.id for child in node.children]

    n = len(items)
    results = derive_weights(matrix, model.number_type, derivation_method, consistency_method=consistency_method)
    crisp_weights = results["crisp_weights"]

    # Consistency
    saaty_cr = Consistency.calculate_saaty_cr(matrix, consistency_method=consistency_method)

    # Local CI Calc for report
    ci, ri = 0.0, 0.0
    if n > 2:
        crisp_mat = np.eye(n)
        for r in range(n):
            for c in range(r+1, n):
                v = matrix[r,c].defuzzify(method=consistency_method)
                if v<=1e-9: v=1e-9
                crisp_mat[r,c]=v
                crisp_mat[c,r]=1.0/v
        try:
            evals = np.linalg.eigvals(crisp_mat)
            lmax = np.max(np.real(evals))
            ci = max(0.0, (lmax - n)/(n-1))
        except: ci = -1.0
        ri = Consistency._get_random_index(n)

    report_lines = []
    col_width = 25
    header = f"{'':<10}" + "".join([f"{item:<{col_width}}" for item in items])
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for i, item in enumerate(items):
        row_str = f"{item:<10}"
        for j in range(n):
            cell = matrix[i, j]
            if hasattr(cell, 'l'): c_str = f"({cell.l:.3f}, {cell.m:.3f}, {cell.u:.3f})"
            elif hasattr(cell, 'mu'): c_str = f"(μ:{cell.mu:.3f}, ν:{cell.nu:.3f})"
            elif hasattr(cell, 'value'): c_str = f"{cell.value:.3f}"
            else: c_str = str(cell)
            row_str += f"{c_str:<{col_width}}"
        report_lines.append(row_str)

    report_lines.append("-" * len(header))
    w_str = ", ".join([f"{w:.4f}" for w in crisp_weights])
    report_lines.append(f"Weights ({node.id}) = {{ {w_str} }}")
    report_lines.append(f"CR: {saaty_cr:.4f}, CI: {ci:.4f}, RI: {ri:.2f}")

    return "\n".join(report_lines)

def generate_full_report(
    model: Hierarchy,
    derivation_method: str = 'geometric_mean',
    consistency_method: str = 'centroid',
    filename: str | None = None
) -> str:
    """
    Generates a comprehensive text report for all matrices in the AHP model.

    Args:
        model: The fully defined and calculated Hierarchy object.
        derivation_method: The method to use for deriving weights.
        consistency_method: The defuzzification method for consistency.
        filename (optional): If provided, saves the report to a text file.

    Returns:
        The full report as a single string.
    """
    report_parts = []
    title = "Fuzzy Matrices Analysis Report"
    report_parts.append("=" * len(title))
    report_parts.append(title)
    report_parts.append("=" * len(title) + "\n")

    def _traverse(node):
        # Allow leaf nodes if they have matrices (Alternative comparisons)
        if node.comparison_matrix is not None:
            report_parts.append(f"\n--- Matrix for Node: {node.id} ---")
            report_parts.append(generate_matrix_report(node, model, derivation_method, consistency_method))

        for child in node.children: _traverse(child)

    _traverse(model.root)
    full_report = "\n".join(report_parts)

    if filename:
        try:
            with open(filename, 'w', encoding='utf-8') as f: f.write(full_report)
        except Exception as e: print(e)

    return full_report

def generate_performance_matrix(model, threshold_percent=0.60):
    """
    Generates a detailed breakdown of scores exactly like the Excel matrix.

    Rows: Criteria Weights, Sub-crit Weights, Companies, Threshold, Max.
    Columns: Hierarchical (Criterion -> Sub-criterion).
    """

    # 1. Setup Columns (MultiIndex) and Data Storage
    # Structure: (Criterion Name, Sub-Criterion Name)
    columns = []

    # We need to capture the weights for the header rows
    crit_weights_row = {}
    sub_weights_row = {}

    # To store company data: { 'Company A': { ('C1', 'S11'): 0.0386, ... } }
    company_rows = {alt.name: {} for alt in model.alternatives}

    # 2. Iterate through the Hierarchy
    for crit in model.root.children:
        crit_weight = crit.local_weight.defuzzify()

        # Determine effective children (if a criterion has no sub-criteria, it treats itself as one)
        children = crit.children if crit.children else [crit]

        criterion_total_scores = {alt.name: 0.0 for alt in model.alternatives}
        criterion_max_possible = 0.0

        for sub in children:
            # -- A. Get Weights --
            # Global weight of sub-criterion is what matters for the final sum
            global_w = sub.global_weight.defuzzify()

            col_key = (crit.id, sub.id)
            columns.append(col_key)

            crit_weights_row[col_key] = crit_weight # Repeat parent weight for display
            sub_weights_row[col_key] = global_w

            criterion_max_possible += global_w

            # -- B. Calculate Weighted Scores for Companies --
            for alt in model.alternatives:
                # Raw performance (0.0 to 1.0)
                raw_perf = alt.get_performance_score(sub.id)
                if raw_perf is None: raw_perf = 0.0

                # Weighted Contribution
                weighted_score = global_w * raw_perf

                company_rows[alt.name][col_key] = weighted_score
                criterion_total_scores[alt.name] += weighted_score

        # -- C. Add the "Total" Column for this Criterion --
        total_col_key = (crit.id, f"TOTAL ({crit.id})")
        columns.append(total_col_key)

        crit_weights_row[total_col_key] = np.nan # Spacer
        sub_weights_row[total_col_key] = np.nan  # Spacer

        # Store totals for companies
        for alt_name in company_rows:
            company_rows[alt_name][total_col_key] = criterion_total_scores[alt_name]

    # 3. Build the DataFrame
    # Create MultiIndex for columns
    midx = pd.MultiIndex.from_tuples(columns, names=["Criterion", "Sub-Criterion"])

    # Combine all rows
    df = pd.DataFrame(columns=midx)

    # Add Weight Rows
    df.loc["Criteria Weight"] = crit_weights_row
    df.loc["Global Sub-Weight"] = sub_weights_row

    # Add Company Rows
    for alt_name, scores in company_rows.items():
        df.loc[alt_name] = scores

    # 4. Add Stats Rows (Threshold & Max)
    # Theoretical Max is simply the Global Weight (assuming perf=1.0)
    # Threshold is Max * percentage

    theo_max_row = {}
    threshold_row = {}

    for col in columns:
        crit_id, sub_id = col

        if "TOTAL" in sub_id:
            # For total columns, sum the sub-weights of that criterion
            # (We have to find the matching sub-columns in the df to sum them)
            # Simplest way: Look at "Global Sub-Weight" row for this criterion group
            # But "Global Sub-Weight" is NaN for the total column.

            # Calculate sum of weights for this criterion group
            relevant_weights = [val for k, val in sub_weights_row.items() if k[0] == crit_id and not np.isnan(val)]
            group_max = sum(relevant_weights)

            theo_max_row[col] = group_max
            threshold_row[col] = group_max * threshold_percent
        else:
            # For specific sub-criteria
            g_weight = sub_weights_row[col]
            theo_max_row[col] = g_weight
            threshold_row[col] = g_weight * threshold_percent

    df.loc[f"Threshold ({int(threshold_percent*100)}%)"] = threshold_row
    df.loc["Theoretical Max"] = theo_max_row

    return df

def export_full_report(
    model: Hierarchy,
    target: str,
    output_format: str = 'excel',
    spreadsheet_id: str | None = None,
    *,  # Makes subsequent arguments keyword-only
    derivation_method: str = 'geometric_mean',
    consistency_method: str = 'centroid'
):
    """
    Generates a comprehensive analysis report and saves it to the specified format.

    Args:
        model: The fully calculated Hierarchy object.
        target: For 'excel' and 'csv', the path to the file.
                  For 'gsheet', the desired name of the new Google Spreadsheet.
        output_format (str, optional): The output format.
                                       Options: 'excel', 'csv', 'gsheet'. Defaults to 'excel'.
        derivation_method: The method for deriving weights.
        consistency_method: The defuzzification method for consistency checks.
    """
    _check_pandas_availability() # Required for all formats now

    # --- Dispatch to the correct export function ---
    if output_format.lower() == 'excel':
        _export_to_excel(model, target, derivation_method, consistency_method)
    elif output_format.lower() == 'csv':
        _export_to_csv(model, target, derivation_method, consistency_method)
    elif output_format.lower() == 'gsheet':
        if spreadsheet_id:
            _export_to_gsheet(model, spreadsheet_id, derivation_method, consistency_method, mode='open')
        else:
            # Otherwise, create a new one with the 'target' as its name
            _export_to_gsheet(model, target, derivation_method, consistency_method, mode='create')
    else:
        raise ValueError(f"Unsupported output_format: '{output_format}'. "
                         "Choose from 'excel', 'csv', or 'gsheet'.")

def _create_report_dataframe(node, model, derivation_method, consistency_method):
    from .weight_derivation import derive_weights
    from .consistency import Consistency

    matrix = node.comparison_matrix
    if node.is_leaf and model.alternatives:
        items = [alt.name for alt in model.alternatives]
    else:
        items = [child.id for child in node.children]

    n = len(items)
    res = derive_weights(matrix, model.number_type, derivation_method, consistency_method=consistency_method)

    saaty_cr = Consistency.calculate_saaty_cr(matrix, consistency_method=consistency_method)

    # Data rows
    data = []
    for i in range(n):
        row = {}
        for j in range(n):
            cell = matrix[i, j]
            if hasattr(cell, 'l'): s = f"{cell.l:.2f},{cell.m:.2f},{cell.u:.2f}"
            elif hasattr(cell, 'mu'): s = f"μ{cell.mu:.2f},ν{cell.nu:.2f}"
            elif hasattr(cell, 'value'): s = f"{cell.value:.2f}"
            else: s = str(cell)
            row[items[j]] = s
        data.append(row)

    df = pd.DataFrame(data, index=items)

    # Weights row
    w_row = pd.DataFrame({items[i]: f"{res['crisp_weights'][i]:.4f}" for i in range(n)}, index=["Weights"])
    df = pd.concat([df, pd.DataFrame([['']*n], index=[' '], columns=items), w_row])

    # Stats
    stats = pd.DataFrame({'Value': [f"{saaty_cr:.4f}"]}, index=['Saaty CR'])
    # Add empty columns to match
    stats = pd.concat([stats, pd.DataFrame('', index=stats.index, columns=items[1:])], axis=1)
    # Fix column name match for concat
    stats.columns = items

    df = pd.concat([df, stats])
    return df, None

def _create_report_dataframe_retired(
    node_with_matrix: Node,
    model: Hierarchy,
    derivation_method: str,
    consistency_method: str
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame for a single matrix report.
    This is a helper for exporting to structured formats.
    """
    from .weight_derivation import derive_weights
    from .consistency import Consistency

    matrix = node_with_matrix.comparison_matrix
    items = [child.id for child in node_with_matrix.children]
    n = len(items)

    # --- Step 1: Perform ALL calculations first ---
    results = derive_weights(matrix, model.number_type, derivation_method)
    crisp_weights = results["crisp_weights"]

    # Calculate all consistency metrics
    gci = Consistency.calculate_gci(matrix, consistency_method=consistency_method)

    # Mikhailov's Lambda is specific to TFNs and fuzzy programming
    mikhailov_lambda = "N/A"
    if isinstance(matrix[0,0], TFN):
        try:
            lambda_results = Consistency.calculate_mikhailov_lambda(matrix, TFN)
            mikhailov_lambda = f"{lambda_results:.4f}" if lambda_results != -1.0 else "Opt. Failed"
        except ImportError:
            mikhailov_lambda = "SciPy not installed"

    ci, ri, saaty_cr = 0.0, 0.0, 0.0
    if n > 2:
        # 1. Use the centralized robust logic from Consistency class
        saaty_cr = Consistency.calculate_saaty_cr(matrix, consistency_method=consistency_method)

        # 2. Re-calculate CI/RI locally strictly for display purposes
        # using the SAME robust matrix construction as Consistency.py
        crisp_matrix_for_eig = np.eye(n) # Force 1.0 diagonal
        for r in range(n):
            for c in range(r + 1, n):
                val = matrix[r, c].defuzzify(method=consistency_method)
                if val <= 1e-9: val = 1e-9
                crisp_matrix_for_eig[r, c] = val
                crisp_matrix_for_eig[c, r] = 1.0 / val

        try:
            # Check eigenvalues using the correct crisp matrix
            eigenvalues = np.linalg.eigvals(crisp_matrix_for_eig)
            lambda_max = np.max(np.real(eigenvalues))
            ci = (lambda_max - n) / (n - 1)
            if ci < 0: ci = 0.0 # Clamp negative zero
        except np.linalg.LinAlgError:
            ci = -1.0

        ri = Consistency._get_random_index(n)

    # --- Step 2: Build the report content as strings ---

    # 1. Main Matrix Data
    matrix_data_as_strings = []
    for i in range(n):
        row_data = {}
        for j in range(n):
            cell = matrix[i, j]
            if hasattr(cell, 'l'): # TFN-like
                 row_data[items[j]] = f"({cell.l:.3f}, {cell.m:.3f}, {cell.u:.3f})"
            elif hasattr(cell, 'a'): # TrFN-like
                 row_data[items[j]] = f"({cell.a:.3f}, {cell.b:.3f}, {cell.c:.3f}, {cell.d:.3f})"
            elif hasattr(cell, 'mu'): # IFN-like
                 row_data[items[j]] = f"(μ:{cell.mu:.3f}, ν:{cell.nu:.3f})"
            elif hasattr(cell, 'value'): # Crisp-like
                 row_data[items[j]] = f"{cell.value:.3f}"
            else: # Fallback
                 row_data[items[j]] = str(cell)
        matrix_data_as_strings.append(row_data)

    report_df = pd.DataFrame(matrix_data_as_strings, index=items)

    # 2. Add summary stats as new rows
    # Create an empty row for spacing
    spacer = pd.DataFrame([[''] * n], columns=items, index=[' '])
    report_df = pd.concat([report_df, spacer])

    # Add Weights row
    weights_data = {items[i]: f"{w:.4f}" for i, w in enumerate(crisp_weights)}
    weights_row = pd.DataFrame(weights_data, index=["Weights"])
    report_df = pd.concat([report_df, weights_row])

    # Add Consistency rows
    # For these, we create a row with a descriptive index and put the value in the first column
    cons_data = {
        'Saaty CR': f"{saaty_cr:.4f} (Threshold: {0.1})",
        'Consistency Index': f"{ci:.4f}",
        'Random Index': f"{ri:.2f}",
        'Geometric Consistency Index (GCI)': f"{gci:.4f} (Threshold: {Consistency._get_gci_threshold(n)})",
        "Mikhailov's Lambda (TFN only)": mikhailov_lambda,
    }

    # Create a DataFrame for the consistency metrics
    cons_df = pd.DataFrame.from_dict(cons_data, orient='index', columns=['Value'])
    cons_df.index.name = "Consistency Metrics"

    for desc, val in cons_data.items():
        # Create a row with NaN for all columns except the first one
        row_content = {col: '' for col in items}
        row_content[items[0]] = val
        row = pd.DataFrame(row_content, index=[desc])
        report_df = pd.concat([report_df, row])

    return report_df, cons_df

def format_matrix_as_table(
    matrix: np.ndarray,
    item_names: List[str],
    consistency_method: str | None = None
) -> 'pd.DataFrame':
    """
    Formats a single comparison matrix into a classic n x n table.

    Args:
        matrix: The single comparison matrix to format.
        item_names: A list of the names of the criteria/alternatives.
        consistency_method (optional): If provided, converts fuzzy numbers to crisp values.

    Returns:
        A pandas DataFrame representing the classic AHP matrix.
    """
    _check_pandas_availability()

    n = len(item_names)
    table_data = []

    for i in range(n):
        row_data = []
        for j in range(n):
            cell = matrix[i, j]
            if consistency_method and hasattr(cell, 'defuzzify'):
                row_data.append(f"{cell.defuzzify(method=consistency_method):.3f}")
            else:
                row_data.append(str(cell))
        table_data.append(row_data)

    df = pd.DataFrame(table_data, index=item_names, columns=item_names)
    return df

def format_group_judgments_as_table(
    matrices: List[np.ndarray],
    item_names: List[str],
    source_names: List[str] | None = None,
    consistency_method: str | None = None
) -> 'pd.DataFrame':
    """
    Formats a list of comparison matrices into a tabular, flattened pair format,
    ideal for reporting group judgments.

    Args:
        matrices: A list of comparison matrices (e.g., one per expert).
        item_names: A list of the names of the criteria/alternatives.
        source_names (optional): Names for each matrix source (e.g., ['Expert 1', 'Expert 2']).
        consistency_method (optional): If provided, converts fuzzy numbers to crisp values.

    Returns:
        A pandas DataFrame containing the formatted group judgments.
    """
    _check_pandas_availability()

    if not matrices:
        return pd.DataFrame()

    n = len(item_names)
    if source_names and len(source_names) != len(matrices):
        raise ValueError("Length of source_names must match the number of matrices.")
    elif not source_names:
        source_names = [f"Source {i+1}" for i in range(len(matrices))]

    table_data = []
    pair_labels = []

    # Iterate through the upper triangle to define the pairs
    for i in range(n):
        for j in range(i + 1, n):
            pair_label = f"({item_names[i]}, {item_names[j]})"
            pair_labels.append(pair_label)

            row_data = {}
            for k, matrix in enumerate(matrices):
                cell = matrix[i, j]
                source_name = source_names[k]

                if consistency_method and hasattr(cell, 'defuzzify'):
                    row_data[source_name] = f"{cell.defuzzify(method=consistency_method):.3f}"
                else:
                    row_data[source_name] = str(cell)

            table_data.append(row_data)

    df = pd.DataFrame(table_data, index=pd.Index(pair_labels, name="Pairs"))
    return df

def export_full_report(model, target, output_format='excel', spreadsheet_id=None, derivation_method='geometric_mean', consistency_method='centroid'):
    _check_pandas_availability()
    if output_format == 'excel': _export_to_excel(model, target, derivation_method, consistency_method)
    elif output_format == 'csv': _export_to_csv(model, target, derivation_method, consistency_method)
    elif output_format == 'gsheets': _export_to_gsheet(model, target, derivation_method, consistency_method)

def _export_to_excel(model, filename, derivation_method, consistency_method):
    if not filename.endswith('.xlsx'): filename += '.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        def _write(node):
            if node.comparison_matrix is not None:
                df, _ = _create_report_dataframe(node, model, derivation_method, consistency_method)
                sheet = node.id[:31].replace(':', '')
                df.to_excel(writer, sheet_name=sheet)
            for c in node.children: _write(c)
        _write(model.root)

def _export_to_csv(model, filename, derivation_method, consistency_method):
    if base_name.endswith('.csv'): base_name = base_name[:-4]
    def _write(node):
        if node.comparison_matrix is not None:
            df, _ = _create_report_dataframe(node, model, derivation_method, consistency_method)
            df.to_csv(f"{base_name}_{node.id}.csv")
        for c in node.children: _write(c)
    _write(model.root)

def _export_to_gsheet(
    model: Hierarchy,
    name_or_id: str,
    derivation_method: str,
    consistency_method: str,
    mode: str = 'create'
):
    print("Starting Google Sheets export...")
    try:
        import gspread
        from gspread_dataframe import set_with_dataframe
        from google.colab import auth
        from google.auth import default
    except ImportError:
        raise ImportError("Google Sheets export requires 'gspread', 'gspread-dataframe', and 'google-auth'. "
                        "Install with: pip install gspread gspread-dataframe google-auth")

    try:
        # Step 1: Authenticate
        print("  - Authenticating with Google...")
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        print("  - Authentication successful.")

        spreadsheet = None
        if mode == 'create':
            # Create a new spreadsheet
            spreadsheet = gc.create(name_or_id)
            print(f"  - Created new spreadsheet: '{name_or_id}'")
            # Share it back to the user
            user_email = gc.auth.service_account_email if hasattr(gc.auth, 'service_account_email') else gc.auth.user_info.get('email')
            if user_email:
                print(f"  - Sharing with {user_email}...")
                spreadsheet.share(user_email, perm_type='user', role='writer', notify=False)
            else:
                print("  - Warning: Could not determine user email to share sheet.")

        elif mode == 'open':
            # Open an existing spreadsheet by its ID/key
            try:
                spreadsheet = gc.open_by_key(name_or_id)
                print(f"  - Successfully opened existing spreadsheet: '{spreadsheet.title}'")
                # Clear all existing worksheets except the first one to start fresh
                for worksheet in spreadsheet.worksheets()[1:]:
                    spreadsheet.del_worksheet(worksheet)
                if spreadsheet.worksheets(): # If there's at least one sheet
                    spreadsheet.sheet1.clear()
                    spreadsheet.sheet1.update_title("TempSheet") # Rename to avoid name conflicts
            except gspread.exceptions.SpreadsheetNotFound:
                print(f"❌ Error: Spreadsheet with ID '{name_or_id}' not found.")
                return

        if not spreadsheet:
            return

        # Step 2: Write data to sheets
        def _traverse_and_upload(node: Node, first_sheet=True):
            """A nested helper function to recursively upload data."""
            if not node.is_leaf and node.comparison_matrix is not None:
                # Create a DataFrame for this node's matrix report
                df_report, df_matrices = _create_report_dataframe(node, model, derivation_method, consistency_method)

                # Sanitize sheet name
                sheet_name = node.id.replace(':', '').replace('\\', '').replace('/', '')[:31]

                if first_sheet and spreadsheet.worksheet("TempSheet"):
                    # Use and rename the first default sheet
                    worksheet = spreadsheet.sheet1
                    worksheet.update_title(sheet_name)
                else:
                    # Create a new sheet
                    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=20)

                # Use gspread-dataframe to upload the DataFrame
                set_with_dataframe(worksheet, df_report, include_index=True)
                print(f"    - Uploaded report for '{node.id}' to sheet '{sheet_name}'.")
                first_sheet = False # Ensure subsequent calls create new sheets

            for child in node.children:
                # Pass the updated 'first_sheet' status down the recursion
                first_sheet = _traverse_and_upload(child, first_sheet)

            return first_sheet # Return the final status

        _traverse_and_upload(model.root)

        # Clean up by deleting the temporary sheet if it still exists
        try:
            temp_sheet = spreadsheet.worksheet("TempSheet")
            if temp_sheet: spreadsheet.del_worksheet(temp_sheet)
        except gspread.exceptions.WorksheetNotFound:
            pass # It was correctly renamed, which is good.

        print(f"\n✅ Google Sheets report complete!")
        print(f"   URL: {spreadsheet.url}")

    except Exception as e:
        print(f"\n❌ An error occurred during Google Sheets export: {e}")
        print("   Please ensure you are running in a Google Colab environment and have granted permissions.")


class SensitivityAnalyzerRetired:
    """
    Performs One-at-a-Time (OAT) sensitivity analysis on a fitted AHP model.
    """
    def __init__(self, model: 'Hierarchy'):
        if model.get_criteria_weights is None:
            raise RuntimeError("The provided model has not been fully run. Call .fit_weights() and .score() first.")

        # Create a deep copy to avoid modifying the original model
        self.base_model = copy.deepcopy(model)
        self.results = {}

    def analyze(
        self,
        parent_node_id: str,
        steps: int = 50
    ) -> 'pd.DataFrame':
        """
        Analyzes the sensitivity of the final scores to changes in the weights
        of the children of a specified parent node.

        Args:
            parent_node_id (str): The ID of the parent node whose children's weights
                                  will be varied (e.g., 'COLOR Score' to vary C1, C2...).
            steps (int): The number of steps to simulate for each weight change.

        Returns:
            A pandas DataFrame containing the results, ready for plotting.
        """
        import pandas as pd # Local import

        parent_node = self.base_model._find_node(parent_node_id)
        if not parent_node or parent_node.is_leaf:
            raise ValueError(f"'{parent_node_id}' is not a valid parent node.")

        children_to_vary = parent_node.children
        original_local_weights = self.base_model.get_child_weights(parent_node_id, weight_type="local")

        analysis_data = []

        # Iterate through each criterion that we want to vary
        for child_node in children_to_vary:
            criterion_to_vary_id = child_node.id
            print(f"\n--- Analyzing sensitivity with respect to: '{criterion_to_vary_id}' ---")

            # Define the range of weights to test for this criterion
            weight_range = np.linspace(0.01, 0.99, steps)

            # Get the other "sibling" criteria
            siblings = [c for c in children_to_vary if c.id != criterion_to_vary_id]

            for new_weight in weight_range:
                # Create a temporary model for this single simulation step
                temp_model = copy.deepcopy(self.base_model)

                # Set the new weight for the criterion being varied
                node_to_change = temp_model._find_node(criterion_to_vary_id)
                node_to_change.local_weight = temp_model.number_type.from_normalized(new_weight)

                # Adjust the weights of the siblings proportionally
                remaining_weight = 1.0 - new_weight
                original_sibling_weight_sum = sum(original_local_weights[s.id] for s in siblings)

                if original_sibling_weight_sum > 1e-9:
                    for sibling in siblings:
                        proportion = original_local_weights[sibling.id] / original_sibling_weight_sum
                        adjusted_weight = proportion * remaining_weight
                        sibling_node_to_change = temp_model._find_node(sibling.id)
                        sibling_node_to_change.local_weight = temp_model.number_type.from_normalized(adjusted_weight)

                # With weights adjusted, recalculate the entire model
                temp_model._recalculate_global_weights()

                for alt in temp_model.alternatives:
                    alt.node_scores.clear()

                # Now, the scoring method will work correctly
                temp_model.score_alternatives_by_performance_ifwa()

                # Store the results for this step
                for alt in temp_model.alternatives:
                    analysis_data.append({
                        "varied_criterion": criterion_to_vary_id,
                        "criterion_weight": new_weight,
                        "alternative": alt.name,
                        "final_score": alt.overall_score.defuzzify()
                    })

        self.results_df = pd.DataFrame(analysis_data)
        return self.results_df

    def plot(self):
        """
        Generates a plot of the sensitivity analysis results.
        Requires the `analyze` method to have been run first.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise RuntimeError("No analysis results to plot. Run .analyze() first.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            raise ImportError("Plotting requires matplotlib and seaborn. Install them with: pip install matplotlib seaborn")

        varied_criteria = self.results_df['varied_criterion'].unique()
        num_plots = len(varied_criteria)

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes] # Ensure axes is always a list

        for ax, criterion in zip(axes, varied_criteria):
            subset_df = self.results_df[self.results_df['varied_criterion'] == criterion]

            sns.lineplot(
                data=subset_df,
                x='criterion_weight',
                y='final_score',
                hue='alternative',
                ax=ax,
                marker='o',
                markersize=4
            )
            ax.set_title(f"Sensitivity of Final Scores to Weight of '{criterion}'")
            ax.set_ylabel("Final Company Score")
            ax.legend(title="Company")

        axes[-1].set_xlabel("Criterion Weight")
        fig.tight_layout()
        plt.show()



class SensitivityAnalyzer:
    """
    Performs One-at-a-Time (OAT) sensitivity analysis on a fitted AHP model.
    """
    def __init__(self, pipeline: 'Workflow'):
        if pipeline.model.root.global_weight is None:
             raise RuntimeError("The provided pipeline's model has not been fully fitted. Run .fit_weights() and .score() on the pipeline first.")

        # We store the entire pipeline to have access to its configuration and fitted model
        self.base_pipeline = copy.deepcopy(pipeline)
        self.results_df = pd.DataFrame()

    def analyze(self, parent_node_id: str, steps: int = 30) -> pd.DataFrame:
        """
        Analyzes the sensitivity of the final scores to changes in the weights
        of the children of a specified parent node.
        """
        parent_node = self.base_pipeline.model._find_node(parent_node_id)
        if not parent_node or parent_node.is_leaf:
            raise ValueError(f"'{parent_node_id}' is not a valid parent node.")

        children_to_vary = parent_node.children
        original_local_weights = self.base_pipeline.model.get_child_weights(parent_node_id, weight_type="local")

        analysis_data = []

        for child_node in children_to_vary:
            criterion_to_vary_id = child_node.id
            print(f"\n--- Analyzing sensitivity for: '{criterion_to_vary_id}' ---")

            weight_range = np.linspace(0.01, 0.99, steps)
            siblings = [c for c in children_to_vary if c.id != criterion_to_vary_id]

            for new_weight in weight_range:
                # 1. Create a pristine, deep copy for this iteration.
                #    Our custom __deepcopy__ methods ensure it's a clean slate.
                temp_model = copy.deepcopy(self.base_pipeline.model)

                # 2. Apply the weight change
                node_to_change = temp_model._find_node(criterion_to_vary_id)
                node_to_change.local_weight = temp_model.number_type.from_normalized(new_weight)

                remaining_weight = 1.0 - new_weight
                original_sibling_weight_sum = sum(original_local_weights[s.id] for s in siblings)

                if original_sibling_weight_sum > 1e-9:
                    for sibling in siblings:
                        proportion = original_local_weights[sibling.id] / original_sibling_weight_sum
                        adjusted_weight = proportion * remaining_weight
                        sibling_node_to_change = temp_model._find_node(sibling.id)
                        sibling_node_to_change.local_weight = temp_model.number_type.from_normalized(adjusted_weight)

                # 3. Recalculate global weights based on the new local weights
                temp_model._recalculate_global_weights()

                # 4. Score the alternatives. Because the alternatives in temp_model have
                #    empty .node_scores caches, this will perform a full, fresh calculation.
                temp_model.score_alternatives_by_performance()

                # 5. Store results
                for alt in temp_model.alternatives:
                    analysis_data.append({
                        "varied_criterion": criterion_to_vary_id,
                        "criterion_weight": new_weight,
                        "alternative": alt.name,
                        "final_score": alt.overall_score.defuzzify()
                    })

        self.results_df = pd.DataFrame(analysis_data)
        return self.results_df

    def plot(self):
        """
        Generates a plot of the sensitivity analysis results.
        Requires the `analyze` method to have been run first.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise RuntimeError("No analysis results to plot. Run .analyze() first.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
        except ImportError:
            raise ImportError("Plotting requires matplotlib and seaborn. Install them with: pip install matplotlib seaborn")

        varied_criteria = self.results_df['varied_criterion'].unique()
        num_plots = len(varied_criteria)

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes] # Ensure axes is always a list

        for ax, criterion in zip(axes, varied_criteria):
            subset_df = self.results_df[self.results_df['varied_criterion'] == criterion]

            sns.lineplot(
                data=subset_df,
                x='criterion_weight',
                y='final_score',
                hue='alternative',
                ax=ax,
                marker='o',
                markersize=4
            )
            ax.set_title(f"Sensitivity of Final Scores to Weight of '{criterion}'")
            ax.set_ylabel("Final Company Score")
            ax.legend(title="Company")

        axes[-1].set_xlabel("Criterion Weight")
        fig.tight_layout()
        plt.show()
