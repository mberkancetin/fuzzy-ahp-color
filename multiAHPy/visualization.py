from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, Union
import uuid
import numpy as np
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
    from multiAHPy.model import Hierarchy, Node
    from multiAHPy.types import NumericType, Number, TFN, Crisp

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

def _render_expandable_node(node: 'Node', is_alt: bool = False, alt_obj=None) -> str:
    """
    Renders a single, generic, expandable node box for any element in the hierarchy.
    """
    body_id = 'node-body-' + str(uuid.uuid4())[:8]

    # --- Prepare details for the expandable body ---
    details_html = "<ul>"
    if is_alt: # Alternative Node
        score_str = f"{alt_obj.overall_score.defuzzify():.4f}" if alt_obj.overall_score else 'N/A'
        details_html += f"<li><b>Final Score:</b> {score_str}</li>"
        if alt_obj.performance_scores:
            details_html += "<li><b>Performance Scores:</b><ul>"
            for leaf_id, score in alt_obj.performance_scores.items():
                details_html += f"<li>{leaf_id}: {score.defuzzify():.3f}</li>"
            details_html += "</ul></li>"
    else: # Hierarchy Node (Goal, Criterion, Sub-criterion)
        local_w_str = f"{node.local_weight.defuzzify():.3f}" if node.local_weight else 'N/A'
        global_w_str = f"{node.global_weight.defuzzify():.4f}" if node.global_weight else 'N/A'
        parent_id_str = node.parent.id if node.parent else 'None'
        details_html += f"<li><b>Description:</b> {node.description or 'N/A'}</li>"
        details_html += f"<li><b>Parent:</b> {parent_id_str}</li>"
        details_html += f"<li><b>Local Weight:</b> {local_w_str}</li>"
        details_html += f"<li><b>Global Weight:</b> {global_w_str}</li>"
    details_html += "</ul>"

    header_text = alt_obj.name if is_alt else node.id

    return f"""
    <div class="ahp-node" onclick="toggleAHPNode('{body_id}', event)">
        <div class="ahp-node-header">{header_text}</div>
        <div class="ahp-node-body" id="{body_id}">
            {details_html}
        </div>
    </div>
    """

def _render_criterion_tree(node: 'Node') -> str:
    """Renders the tree for a single criterion and its children."""
    if node.is_leaf:
        return _render_expandable_node(node)

    parent_html = f'<div class="tree-parent">{_render_expandable_node(node)}</div>'
    children_html = "".join([_render_expandable_node(child) for child in node.children])
    children_container_html = f'<div class="tree-children">{children_html}</div>'

    return f'<div class="criterion-tree-block">{parent_html}{children_container_html}</div>'

def display_tree_hierarchy(model: 'Hierarchy', filename: str | None = None):
    """
    Generates and displays/saves a static, tree-diagram-style HTML representation
    of the AHP hierarchy, matching the final photoshopped layout.
    """
    from IPython.display import display, HTML

    # Ensure weights are calculated
    if not all(c.global_weight is not None for c in model.root.children):
        try: model.calculate_weights()
        except Exception as e: print(f"Warning: Could not auto-calculate weights for display. Run model.calculate_weights() first. Error: {e}")

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

            /* --- FIX IS HERE --- */
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
            {_render_expandable_node(model.root)}
        </div>
    </div>
    """

    # 2. Criteria Stage (Horizontal layout of vertical trees)
    criteria_trees_html = "".join([_render_criterion_tree(crit_node) for crit_node in model.root.children])
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
        # Sort alternatives by score before displaying
        ranked_alts = sorted(model.alternatives, key=lambda alt: alt.overall_score.defuzzify() if alt.overall_score else -1, reverse=True)
        alt_html_parts = [_render_expandable_node(None, is_alt=True, alt_obj=alt) for alt in ranked_alts]

        alternatives_html = f"""
        <div class="ahp-stage">
            <div class="ahp-stage-title">Alternatives</div>
            <div class="ahp-alternatives-container">
                {''.join(alt_html_parts)}
            </div>
        </div>
        """

    # Assemble the final document
    connector = '<div class="ahp-connector"></div>'
    full_html_body = f'{goal_html}{connector}{criteria_html}{connector}{alternatives_html}' if model.alternatives else f'{goal_html}{connector}{criteria_html}'
    full_document = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>AHP Diagram</title>{css_js}</head>
    <body><div class="ahp-viz-container">{full_html_body}</div></body></html>
    """

    # Final step: Save to file or display
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

def format_model_summary(model: 'Hierarchy', alternative_name: str) -> str:
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
    if alt is None:
        return f"Error: Alternative '{alternative_name}' not found in the model."
    if alt.overall_score is None:
        return f"Error: Scores for '{alternative_name}' have not been calculated yet."

    lines = []
    header = f" HIERARCHICAL EVALUATION SUMMARY for: {alt.name} "
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(f"\nFINAL OVERALL SCORE: {alt.overall_score.defuzzify():.4f}\n")

    # Iterate through the main criteria
    for crit_node in model.root.children:
        crit_weight = crit_node.local_weight.defuzzify()
        crit_score = alt.node_scores[crit_node.id].defuzzify()
        crit_contribution = crit_weight * crit_score

        lines.append(f"--- Criterion: {crit_node.id} ({crit_node.description}) ---")
        lines.append(f"  - Global Weight: {crit_node.global_weight.defuzzify():.4f}")
        lines.append(f"  - Aggregated Performance Score for this Criterion: {crit_score:.4f}")
        lines.append(f"  - Contribution to Final Score: {crit_contribution:.4f}\n")

        # Iterate through sub-criteria
        if not crit_node.is_leaf:
            lines.append("    Sub-criteria Breakdown:")
            for sub_crit_node in crit_node.children:
                sub_crit_local_weight = sub_crit_node.local_weight.defuzzify()
                sub_crit_global_weight = sub_crit_node.global_weight.defuzzify()
                perf_score = alt.performance_scores[sub_crit_node.id].defuzzify()

                lines.append(f"      - {sub_crit_node.id}:")
                lines.append(f"          Local Weight (within {crit_node.id}): {sub_crit_local_weight:.3f}")
                lines.append(f"          Global Weight: {sub_crit_global_weight:.4f}")
                lines.append(f"          Performance Score: {perf_score:.4f}")
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

    # Add value labels on top of bars
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

    rankings = model.get_rankings() # This already returns defuzzified, sorted results

    alt_names = [r[0] for r in rankings]
    scores = [r[1] for r in rankings]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(alt_names, scores, color=plt.cm.plasma(np.linspace(0.4, 0.9, len(scores))))

    ax.set_xlabel('Final Score')
    ax.set_ylabel('Alternative')
    ax.set_title('Final Alternative Rankings')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.invert_yaxis() # Display the top-ranked alternative at the top

    # Add value labels to the side of the bars
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{scores[i]:.4f}', va='center')

    fig.tight_layout()
    return fig

def plot_sensitivity_analysis(
    model: Hierarchy,  # <-- It accepts the model object
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
    if not parent_node:
        raise ValueError(f"Parent node '{parent_node_id}' not found.")

    criterion_to_vary = model._find_node(criterion_to_vary_id)
    if not criterion_to_vary or criterion_to_vary.parent != parent_node:
        raise ValueError(f"'{criterion_to_vary_id}' is not a direct child of '{parent_node_id}'.")

    alt_to_track = next((a for a in model.alternatives if a.name == alternative_name), None)
    if not alt_to_track:
        raise ValueError(f"Alternative '{alternative_name}' not found.")

    # --- IMPORTANT: Store original state ---
    # We need to save the local weights of all siblings to restore them later.
    original_sibling_weights = {sib.id: sib.local_weight for sib in parent_node.children}

    # Also get the original defuzzified weight for plotting
    original_crisp_weight = criterion_to_vary.local_weight.defuzzify()

    weight_range = np.linspace(0.01, 0.99, 50)
    final_scores = []

    for new_weight in weight_range:
        # Temporarily set the new weight for the criterion being varied
        criterion_to_vary.local_weight = model.number_type.from_crisp(new_weight)

        # Redistribute the remaining weight among the OTHER siblings proportionally
        siblings_to_adjust = [c for c in parent_node.children if c.id != criterion_to_vary_id]
        current_siblings_weight_sum = sum(original_sibling_weights[s.id].defuzzify() for s in siblings_to_adjust)
        remaining_weight = 1.0 - new_weight

        if current_siblings_weight_sum > 0:
            for sibling in siblings_to_adjust:
                proportion = original_sibling_weights[sibling.id].defuzzify() / current_siblings_weight_sum
                sibling.local_weight = model.number_type.from_crisp(remaining_weight * proportion)

        # Recalculate global weights and scores with the temporary weights
        model.calculate_weights()
        model.calculate_alternative_scores()

        # Find the target alternative in the newly calculated results
        target_alt_new = next(a for a in model.alternatives if a.name == alternative_name)
        final_scores.append(target_alt_new.overall_score.defuzzify())

    # --- RESTORE THE ORIGINAL WEIGHTS to avoid side effects ---
    for child in parent_node.children:
        child.local_weight = original_sibling_weights[child.id]
    # Fully recalculate everything to restore the model to its original state
    model.calculate_weights()
    model.calculate_alternative_scores()

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(weight_range, final_scores, marker='.', linestyle='-', label=f"Score for {alternative_name}")
    ax.axvline(x=original_crisp_weight, color='r', linestyle='--', label=f'Original Weight ({original_crisp_weight:.3f})')

    ax.set_xlabel(f"Local Weight of '{criterion_to_vary_id}'")
    ax.set_ylabel(f"Final Score for '{alternative_name}'")
    ax.set_title(f"Sensitivity Analysis for Criterion '{criterion_to_vary_id}'")
    ax.legend()
    ax.grid(True, alpha=0.5)

    fig.tight_layout()
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
    from multiAHPy.weight_derivation import derive_weights
    from multiAHPy.consistency import Consistency

    matrix = node_with_matrix.comparison_matrix
    if matrix is None:
        return f"--- No comparison matrix found for node '{node_with_matrix.id}' ---\n"

    items = [child.id for child in node_with_matrix.children]
    n = len(items)

    # --- Calculations ---
    results = derive_weights(matrix, model.number_type, derivation_method)
    crisp_weights = results["crisp_weights"]

    # Get consistency metrics
    cr = Consistency.calculate_consistency_ratio(matrix, defuzzify_method=consistency_method)
    # To get CI and RI, we can re-calculate parts of the CR logic
    ci = 0.0
    ri = 0.0
    if n > 2:
        crisp_matrix = np.array([[cell.defuzzify(method=consistency_method) for cell in row] for row in matrix])
        try:
            eigenvalues, _ = np.linalg.eig(crisp_matrix)
            lambda_max = np.max(np.real(eigenvalues))
            ci = (lambda_max - n) / (n - 1)
        except np.linalg.LinAlgError:
            ci = -1 # Indicate failure
        ri = Consistency._get_random_index(n)

    # --- Formatting ---
    report_lines = []

    # 1. Header and Matrix
    col_width = 25  # Width for each matrix column
    header = f"{'':<10}" + "".join([f"{item:<{col_width}}" for item in items])
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for i, item_name in enumerate(items):
        row_str = f"{item_name:<10}"
        for j in range(n):
            cell = matrix[i, j]
            # Format the cell based on its type
            if hasattr(cell, 'l'): # TFN or similar
                 cell_str = f"({cell.l:.3f}, {cell.m:.3f}, {cell.u:.3f})"
            elif hasattr(cell, 'value'): # Crisp
                 cell_str = f"{cell.value:.3f}"
            else:
                 cell_str = str(cell)
            row_str += f"{cell_str:<{col_width}}"
        report_lines.append(row_str)

    report_lines.append("-" * len(header))

    # 2. Weights
    weights_str = ", ".join([f"{w:.4f}" for w in crisp_weights])
    report_lines.append(f"w_{node_with_matrix.id} = {{ {weights_str} }}")

    # 3. Degrees of Probability (Specific to Extent Analysis)
    min_degrees = results.get("min_degrees")
    if min_degrees is not None:
        # Degrees of Probability Min Values Normalized are the final crisp weights
        report_lines.append(f"Degrees of Probability Min Values Normalized: [{weights_str}]")
    
    V_matrix = results.get("possibility_matrix")
    if V_matrix is not None:
        report_lines.append("\n--- Degrees of Probability Matrix (V) ---\n")
        v_header = f"{'':<10}" + "".join([f"{item:<10}" for item in items])
        report_lines.append(v_header)
        report_lines.append("-" * len(v_header))
        for i, item_name in enumerate(items):
            row_str = f"{item_name:<10}"
            for j in range(n):
                row_str += f"{V_matrix[i, j]:<10.4f}"
            report_lines.append(row_str)

    # 4. Consistency Metrics
    report_lines.append(f"CR, CI, RI: ({cr:.6f}, {ci:.6f}, {ri:.2f})")
    
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

    def _traverse_and_report(node: Node):
        if not node.is_leaf and node.comparison_matrix is not None:
            # Add a title for this section
            report_parts.append(f"\n--- Comparisons for Children of: {node.id} ---\n")
            # Generate the report for this node's matrix
            report_parts.append(generate_matrix_report(node, model, derivation_method, consistency_method))
            report_parts.append("\n")

        for child in node.children:
            _traverse_and_report(child)

    # Start the recursive report generation from the root
    _traverse_and_report(model.root)

    full_report = "\n".join(report_parts)

    if filename:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"✅ Full analysis report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving report file: {e}")

    return full_report

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
        _export_to_excel(model, filename, derivation_method, consistency_method)
    elif output_format.lower() == 'csv':
        _export_to_csv(model, filename, derivation_method, consistency_method)
    elif output_format.lower() == 'gsheet':
        if spreadsheet_id:
            _export_to_gsheet(model, spreadsheet_id, derivation_method, consistency_method, mode='open')
        else:
            # Otherwise, create a new one with the 'target' as its name
            _export_to_gsheet(model, target, derivation_method, consistency_method, mode='create')
    else:
        raise ValueError(f"Unsupported output_format: '{output_format}'. "
                         "Choose from 'excel', 'csv', or 'gsheet'.")

def _create_report_dataframe(
    node_with_matrix: Node,
    model: Hierarchy,
    derivation_method: str,
    consistency_method: str
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame for a single matrix report.
    This is a helper for exporting to structured formats.
    """
    from multiAHPy.weight_derivation import derive_weights
    from multiAHPy.consistency import Consistency

    matrix = node_with_matrix.comparison_matrix
    items = [child.id for child in node_with_matrix.children]
    n = len(items)

    # --- Calculations ---
    weights = derive_weights(matrix, model.number_type, derivation_method)
    crisp_weights = [w.defuzzify(method=consistency_method) for w in weights]

    ci, ri, cr = 0.0, 0.0, 0.0
    if n > 2:
        cr = Consistency.calculate_consistency_ratio(matrix, defuzzify_method=consistency_method)
        crisp_matrix = np.array([[cell.defuzzify() for cell in row] for row in matrix])
        try:
            lambda_max = np.max(np.real(np.linalg.eig(crisp_matrix)[0]))
            ci = (lambda_max - n) / (n - 1)
        except np.linalg.LinAlgError: ci = -1
        ri = Consistency._get_random_index(n)

    # --- Build DataFrame ---
    # 1. Matrix Data
    data = []
    for i in range(n):
        row_data = {}
        for j in range(n):
            cell = matrix[i, j]
            # Format the cell based on its type
            if hasattr(cell, 'l'): # TFN or similar
                 cell_str = f"({cell.l:.3f}, {cell.m:.3f}, {cell.u:.3f})"
            elif hasattr(cell, 'value'): # Crisp
                 cell_str = f"{cell.value:.3f}"
            else: cell_str = str(cell)
            row_data[items[j]] = cell_str
        data.append(row_data)

    df = pd.DataFrame(data, index=items)

    # 2. Add empty rows for spacing, then add summary stats
    df.loc[''] = '' # Spacer
    df.loc['Weights'] = {items[i]: f"{w:.4f}" for i, w in enumerate(crisp_weights)}
    df.loc['CR'] = {'': 'Consistency Ratio', items[0]: f"{cr:.4f}"}
    df.loc['CI'] = {'': 'Consistency Index', items[0]: f"{ci:.4f}"}
    df.loc['RI'] = {'': 'Random Index', items[0]: f"{ri:.2f}"}

    return df

def _export_to_excel(model, filename, derivation_method, consistency_method):
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            def _traverse_and_write(node: Node):
                if not node.is_leaf and node.comparison_matrix is not None:
                    df_report = _create_report_dataframe(node, model, derivation_method, consistency_method)
                    sheet_name = node.id.replace(':', '').replace('\\', '').replace('/', '')[:31]
                    df_report.to_excel(writer, sheet_name=sheet_name)
                for child in node.children:
                    _traverse_and_write(child)
            _traverse_and_write(model.root)
        print(f"✅ Excel report successfully saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving Excel report: {e}")

def _export_to_csv(model, filename, derivation_method, consistency_method):
    base_filename = filename[:-4] if filename.endswith('.csv') else filename
    try:
        def _traverse_and_write_csv(node: Node):
            if not node.is_leaf and node.comparison_matrix is not None:
                df_report = _create_report_dataframe(node, model, derivation_method, consistency_method)
                csv_filename = f"{base_filename}_matrix_{node.id}.csv"
                df_report.to_csv(csv_filename)
                print(f"  - Saved {csv_filename}")
            for child in node.children:
                _traverse_and_write_csv(child)
        print(f"✅ Exporting CSV reports with base name '{base_filename}':")
        _traverse_and_write_csv(model.root)
        print("CSV export complete.")
    except Exception as e:
        print(f"❌ Error saving CSV reports: {e}")

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
                df_report = _create_report_dataframe(node, model, derivation_method, consistency_method)
                
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
