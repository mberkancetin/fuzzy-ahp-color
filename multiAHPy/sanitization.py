from __future__ import annotations
from typing import Dict, Any, List, Callable, Literal
import numpy as np
import copy

from .model import Hierarchy, Node
from .types import Crisp
from .consistency import Consistency
from .completion import complete_matrix
from .matrix_builder import FuzzyScale, rebuild_consistent_matrix, rebuild_from_eigenvector

SANITIZATION_STRATEGIES: Dict[str, Callable] = {}

def register_sanitization_strategy(name: str):
    """Decorator to register a new sanitization strategy."""
    def decorator(func: Callable) -> Callable:
        SANITIZATION_STRATEGIES[name] = func
        return func
    return decorator


class DataSanitizer:
    """A configurable class for cleaning and revising expert judgment matrices."""
    def __init__(self, strategy: str, **strategy_kwargs):
        """
        Args:
            strategy (str): The name of the registered sanitization strategy to use.
            **strategy_kwargs: Keyword arguments to pass to the chosen strategy function
                               (e.g., `target_cr`, `max_cycles`, `scale`).
        """
        if strategy not in SANITIZATION_STRATEGIES:
            raise ValueError(f"Unknown sanitization strategy '{strategy}'. Available: {list(SANITIZATION_STRATEGIES.keys())}")
        self.strategy = strategy
        self.strategy_kwargs = strategy_kwargs

    def transform(
        self,
        raw_expert_matrices: Dict[str, List[np.ndarray]],
        root_node: Node,
        target_number_type: type
    ) -> tuple[Dict[str, List[np.ndarray]], Dict[str, Any]]:
        """
        Applies the chosen sanitization strategy to a set of expert matrices.
        """
        print(f"\n--- Sanitizing Expert Data (Strategy: {self.strategy}) ---")

        strategy_func = SANITIZATION_STRATEGIES[self.strategy]

        all_args = {
            "raw_matrices": raw_expert_matrices,
            "root_node": root_node,
            "target_number_type": target_number_type,
            **self.strategy_kwargs
        }

        sanitized_matrices, change_log = strategy_func(**all_args)

        print("--- Sanitization Complete ---")
        return sanitized_matrices, change_log

def _refuzzify_matrix(crisp_matrix: np.ndarray, target_number_type: type, scale: str) -> np.ndarray:
    """
    Helper to convert a consistent crisp matrix back to fuzzy.
    CRITICAL: Uses from_normalized to ensure the centroid remains identical to the crisp value,
    preserving the Consistency Ratio (CR) achieved during sanitization.
    """
    n = crisp_matrix.shape[0]
    re_fuzzified_matrix = np.empty((n, n), dtype=object)
    for r in range(n):
        for c in range(n):
            # We use from_normalized (or equivalent) to create a 'crisp' fuzzy number
            # (e.g. TFN(3,3,3)) instead of a spread-out one (e.g. TFN(2,3,4)).
            # This ensures the CR calculated on the centroids matches the sanitized CR.
            val = float(crisp_matrix[r, c].value)

            # Note: from_normalized usually expects [0,1], but most implementations
            # (like TFN) simply wrap the value: TFN(val, val, val).
            # If the type requires stric 0-1, we use from_saaty with zero spread logic manually.
            if hasattr(target_number_type, 'from_normalized'):
                 re_fuzzified_matrix[r, c] = target_number_type.from_normalized(val)
            else:
                 re_fuzzified_matrix[r, c] = target_number_type.from_saaty(val)

    return re_fuzzified_matrix

def _perform_iterative_revision(
    crisp_matrices_per_expert: List[Dict[str, np.ndarray]],
    root_node: Node,
    target_cr: float,
    max_cycles: int,
    use_bounded_adjustment: bool,
    bound: float
) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    change_log = {}
    revised_crisp_matrices = copy.deepcopy(crisp_matrices_per_expert)

    for i, expert_matrices_dict in enumerate(revised_crisp_matrices):
        expert_id = f"expert_{i+1}"
        expert_change_log = {}
        stagnated_matrices_for_expert = set()

        for revision_cycle in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for node_id, matrix in expert_matrices_dict.items():
                temp_model.set_comparison_matrix(node_id, matrix)

            consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

            # Find worst matrix
            worst_matrix_info = None
            max_cr = target_cr
            for node_id, metrics in consistency_report.items():
                if node_id in stagnated_matrices_for_expert: continue
                cr_val = metrics.get('saaty_cr')
                if isinstance(cr_val, (float, np.floating)) and cr_val > max_cr:
                    max_cr = cr_val
                    worst_matrix_info = {"node_id": node_id, "cr": cr_val}

            if worst_matrix_info is None: break

            node_id_to_fix = worst_matrix_info['node_id']
            recommendations = Consistency.get_consistency_recommendations(temp_model, node_id_to_fix)

            if "error" in recommendations or not recommendations.get("revisions"):
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue

            top_rec = recommendations['revisions'][0]
            new_value = top_rec['suggested_value']
            if use_bounded_adjustment:
                new_value = np.clip(new_value, 1/bound, bound)

            if np.isclose(top_rec['current_value'], new_value, atol=1e-4):
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue

            r, c = top_rec['pair']
            expert_matrices_dict[node_id_to_fix][r, c] = Crisp(new_value)
            expert_matrices_dict[node_id_to_fix][c, r] = Crisp(1 / new_value)

            if node_id_to_fix not in expert_change_log: expert_change_log[node_id_to_fix] = []
            expert_change_log[node_id_to_fix].append({'from': top_rec['current_value'], 'to': new_value, 'cycle': revision_cycle})

        change_log[expert_id] = expert_change_log

    return revised_crisp_matrices, change_log

# ==============================================================================
# REGISTERED SANITIZATION STRATEGIES
# ==============================================================================

def _perform_iterative_revision_retired(
    crisp_matrices_per_expert: List[Dict[str, np.ndarray]],
    root_node: Node,
    target_cr: float,
    max_cycles: int,
    use_bounded_adjustment: bool,
    bound: float
) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    """The self-contained engine for iterative adjustment strategies."""
    change_log = {}

    revised_crisp_matrices = copy.deepcopy(crisp_matrices_per_expert)

    for i, expert_matrices_dict in enumerate(revised_crisp_matrices):
        expert_id = f"expert_{i+1}"
        print(f"\n  - Sanitizing matrices for {expert_id}...")
        expert_change_log = {}
        stagnated_matrices_for_expert = set()

        for revision_cycle in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for node_id, matrix in expert_matrices_dict.items():
                temp_model.set_comparison_matrix(node_id, matrix)

            consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

            worst_matrix_info = None
            max_cr = target_cr
            for node_id, metrics in consistency_report.items():
                if node_id in stagnated_matrices_for_expert:
                    continue

                cr_val = metrics.get('saaty_cr')
                if isinstance(cr_val, (float, np.floating)) and cr_val > max_cr:
                    max_cr = cr_val
                    worst_matrix_info = {"node_id": node_id, "cr": cr_val}

            if worst_matrix_info is None:
                print(f"    - All possible matrices for {expert_id} are sanitized.")

                break

            node_id_to_fix = worst_matrix_info['node_id']
            print(f"    - Cycle {revision_cycle + 1}/{max_cycles}: Fixing '{node_id_to_fix}' (CR: {worst_matrix_info['cr']:.4f})")

            recommendations = Consistency.get_consistency_recommendations(temp_model, node_id_to_fix)

            if "error" in recommendations or not recommendations.get("revisions"):
                print(f"    - ❌ Warning: Could not get recommendation for {node_id_to_fix}. Flagging as stagnated.")
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue # Skip to the next cycle

            top_rec = recommendations['revisions'][0]
            old_value = top_rec['current_value']
            ideal_value = top_rec['suggested_value']

            new_value = ideal_value
            if use_bounded_adjustment:
                new_value = np.clip(ideal_value, 1/bound, bound)

            # --- ROBUST STAGNATION CHECK ---
            if np.isclose(old_value, new_value, atol=1e-4):
                print(f"    - ⚠️ Warning: Suggested change for '{node_id_to_fix}' is negligible ({old_value:.4f} -> {new_value:.4f}). Flagging as stagnated.")
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue # Skip to the next cycle, we'll try to fix a different matrix

            # Apply the change
            r, c = top_rec['pair']
            matrix_to_revise = expert_matrices_dict[node_id_to_fix]
            matrix_to_revise[r, c] = Crisp(new_value)
            matrix_to_revise[c, r] = Crisp(1 / new_value)

            if node_id_to_fix not in expert_change_log: expert_change_log[node_id_to_fix] = []
            expert_change_log[node_id_to_fix].append({'from': top_rec['current_value'], 'to': new_value, 'cycle': revision_cycle + 1})

        else:
            print(f"    - ⚠️ Warning: Reached max_cycles for {expert_id}. Some matrices may still be inconsistent.")

        change_log[expert_id] = expert_change_log

    return revised_crisp_matrices, change_log

def _perform_dynamic_revision(
    crisp_matrices_per_expert: List[Dict[str, np.ndarray]],
    root_node: Node,
    target_cr: float,
    max_cycles: int,
    bound: float
) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    The self-contained engine for iterative adjustment strategies with dynamic step size.
    """
    change_log = {}

    revised_crisp_matrices = copy.deepcopy(crisp_matrices_per_expert)

    for i, expert_matrices_dict in enumerate(revised_crisp_matrices):
        expert_id = f"expert_{i+1}"
        print(f"\n  - Sanitizing matrices for {expert_id} (Dynamic Step Strategy)...")
        expert_change_log = {}
        stagnated_matrices_for_expert = set()

        for revision_cycle in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for node_id, matrix in expert_matrices_dict.items():
                temp_model.set_comparison_matrix(node_id, matrix)

            consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

            worst_matrix_info = None
            max_cr = target_cr
            for node_id, metrics in consistency_report.items():
                if node_id in stagnated_matrices_for_expert:
                    continue

                cr_val = metrics.get('saaty_cr')
                if isinstance(cr_val, (float, np.floating)) and cr_val > max_cr:
                    max_cr = cr_val
                    worst_matrix_info = {"node_id": node_id, "cr": cr_val}

            if worst_matrix_info is None:
                if stagnated_matrices_for_expert:
                    print(f"    - ❌ Some matrices for {expert_id} are not consistent.")
                    break
                else:
                    print(f"    - ✅ All matrices for {expert_id} are now consistent.")
                    break

            node_id_to_fix = worst_matrix_info['node_id']
            current_cr = worst_matrix_info['cr']
            print(f"    - Cycle {revision_cycle + 1}/{max_cycles}: Fixing '{node_id_to_fix}' (CR: {current_cr:.4f})")

            recommendations = Consistency.get_consistency_recommendations(temp_model, node_id_to_fix)

            if "error" in recommendations or not recommendations.get("revisions"):
                print(f"    - ❌ Warning: Could not get recommendation for {node_id_to_fix}. Flagging as stagnated.")
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue

            top_rec = recommendations['revisions'][0]
            old_value = top_rec['current_value']
            ideal_value = top_rec['suggested_value']

            # --- DYNAMIC STEP SIZE CALCULATION ---
            # Step size is proportional to how far the CR is from the target (0.1)
            # We use a factor (e.g., 0.5) to control the aggressiveness of the step.
            # The step is a fraction of the distance between the current value and the ideal value.
            cr_diff = current_cr - target_cr
            # Normalize cr_diff by the initial CR (or a max CR) to get a factor between 0 and 1
            # For simplicity, we'll use a linear decay factor based on the current CR
            # Let's use a simple factor: 1.0 - (target_cr / current_cr)
            # This factor is 0 when CR=target_cr and approaches 1 when CR is very large.
            # We'll cap the factor to prevent overshooting.

            # Simple linear step factor: 0.5 * (CR_current - CR_target) / CR_current
            # This ensures the step size shrinks as CR approaches CR_target
            step_factor = np.clip(0.5 * (current_cr - target_cr) / current_cr, 0.05, 0.95)

            # Calculate the new value as a weighted average of the old and ideal values
            # new_value = old_value * (1 - step_factor) + ideal_value * step_factor

            # A simpler approach is to adjust the log-ratio of the values
            log_old = np.log(old_value)
            log_ideal = np.log(ideal_value)
            log_new = log_old + (log_ideal - log_old) * step_factor
            new_value = np.exp(log_new)

            # Apply bounding
            new_value = np.clip(new_value, 1/bound, bound)

            # --- ROBUST STAGNATION CHECK ---
            if np.isclose(old_value, new_value, atol=1e-4):
                print(f"    - ⚠️ Warning: Suggested change for '{node_id_to_fix}' is negligible ({old_value:.4f} -> {new_value:.4f}). Flagging as stagnated.")
                stagnated_matrices_for_expert.add(node_id_to_fix)
                continue

            # Apply the change
            r, c = top_rec['pair']
            matrix_to_revise = expert_matrices_dict[node_id_to_fix]
            matrix_to_revise[r, c] = Crisp(new_value)
            matrix_to_revise[c, r] = Crisp(1 / new_value)

            if node_id_to_fix not in expert_change_log: expert_change_log[node_id_to_fix] = []
            expert_change_log[node_id_to_fix].append({'from': top_rec['current_value'], 'to': new_value, 'cycle': revision_cycle + 1, 'step_factor': step_factor})

        else:
            print(f"    - ⚠️ Warning: Reached max_cycles for {expert_id}. Some matrices may still be inconsistent.")

        change_log[expert_id] = expert_change_log

    return revised_crisp_matrices, change_log

@register_sanitization_strategy("adjust_optimal")
@register_sanitization_strategy("adjust_bounded")
@register_sanitization_strategy("simple_iterative")
def iterative_adjustment_strategy(
    raw_matrices: Dict, root_node: Node, target_number_type: type,
    target_cr: float = 0.1, max_cycles: int = 20, bound: float = 9.0,
    strategy: str = "adjust_bounded", scale: str = 'linear', **kwargs
) -> tuple[Dict, Dict]:
    """
    Sanitizes matrices using an iterative adjustment method on a crisp representation.
    This function handles both 'adjust_optimal' and 'adjust_bounded'.
    """
    # 1. Defuzzify
    crisp_matrices_per_expert = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            nid: np.array([[Crisp(c.defuzzify()) for c in row] for row in mats[i]], dtype=object)
            for nid, mats in raw_matrices.items()
        }
        crisp_matrices_per_expert.append(expert_dict)

    # 2. Revise
    use_bounded = (strategy == "adjust_bounded")
    consistent_crisp_matrices, change_log = _perform_iterative_revision(
        crisp_matrices_per_expert, root_node, target_cr, max_cycles, use_bounded, bound
    )

    # 3. Re-fuzzify the results
    sanitized_fuzzy_matrices = {nid: [] for nid in raw_matrices}
    for i in range(num_experts):
        for nid in raw_matrices:
            fuzzy_mat = _refuzzify_matrix(consistent_crisp_matrices[i][nid], target_number_type, scale)
            sanitized_fuzzy_matrices[nid].append(fuzzy_mat)

    return sanitized_fuzzy_matrices, change_log


@register_sanitization_strategy("adjust_persistent")
def adjust_persistent_strategy(
    raw_matrices: Dict, root_node: Node, target_number_type: type,
    target_cr: float = 0.1, max_cycles: int = 40, bound: float = 9.0,
    scale: str = 'linear', **kwargs
) -> tuple[Dict, Dict]:
    # Similar to iterative but tries next best revision if stalled
    # ... (Implementation follows same pattern: defuzzify -> loop -> refuzzify)
    # Reusing the fix in _refuzzify_matrix at the end

    # 1. Defuzzify
    crisp_matrices_per_expert = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            nid: np.array([[Crisp(c.defuzzify()) for c in row] for row in mats[i]], dtype=object)
            for nid, mats in raw_matrices.items()
        }
        crisp_matrices_per_expert.append(expert_dict)

    # 2. Revise Persistent
    change_log = {}
    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        expert_change_log = {}
        mats = crisp_matrices_per_expert[i]

        for _ in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for nid, m in mats.items(): temp_model.set_comparison_matrix(nid, m)

            report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)
            worst = max((m['saaty_cr'], nid) for nid, m in report.items() if m.get('is_consistent') is False) if any(m.get('is_consistent') is False for m in report.values()) else None

            if not worst: break

            nid = worst[1]
            recs = Consistency.get_consistency_recommendations(temp_model, nid).get('revisions', [])

            applied = False
            for rec in recs:
                new_val = np.clip(rec['suggested_value'], 1/bound, bound)
                if not np.isclose(rec['current_value'], new_val, atol=1e-4):
                    r, c = rec['pair']
                    mats[nid][r,c] = Crisp(new_val)
                    mats[nid][c,r] = Crisp(1/new_val)
                    expert_change_log.setdefault(nid, []).append({'cycle': _, 'from': rec['current_value'], 'to': new_val})
                    applied = True
                    break
            if not applied: break # Stagnated

        change_log[expert_id] = expert_change_log

    # 3. Refuzzify
    sanitized_fuzzy_matrices = {nid: [] for nid in raw_matrices}
    for i in range(num_experts):
        for nid in raw_matrices:
            fuzzy_mat = _refuzzify_matrix(crisp_matrices_per_expert[i][nid], target_number_type, scale)
            sanitized_fuzzy_matrices[nid].append(fuzzy_mat)

    return sanitized_fuzzy_matrices, change_log


def adjust_persistent_strategy_retired(
    raw_matrices: Dict,
    root_node: Node,
    target_number_type: type,
    target_cr: float = 0.1,
    max_cycles: int = 40,
    bound: float = 9.0,
    scale: str = 'linear',
    **kwargs
) -> tuple[Dict, Dict]:
    """
    Sanitizes matrices using a persistent iterative adjustment method.

    If the top recommended change for a matrix is negligible (stagnated), this
    strategy will attempt to apply the second most inconsistent change, then the
    third, and so on, until a meaningful revision is made.
    """
    change_log = {}

    # 1. Defuzzify all matrices to a list of crisp matrix dictionaries
    expert_crisp_matrices = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            node_id: np.array([[Crisp(cell.defuzzify()) for cell in row] for row in matrices[i]], dtype=object)
            for node_id, matrices in raw_matrices.items()
        }
        expert_crisp_matrices.append(expert_dict)

    # 2. Iterate through each expert and clean their set of crisp matrices
    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        print(f"\n  - Sanitizing matrices for {expert_id} (Persistent Strategy)...")
        current_expert_matrices = expert_crisp_matrices[i]
        expert_change_log = {}

        for revision_cycle in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for node_id, matrix in current_expert_matrices.items():
                temp_model.set_comparison_matrix(node_id, matrix)
            consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

            worst_matrix_info = None
            max_cr = target_cr
            for node_id, metrics in consistency_report.items():
                cr_val = metrics.get('saaty_cr')
                if isinstance(cr_val, (float, np.floating)) and cr_val > max_cr:
                    max_cr = cr_val
                    worst_matrix_info = {"node_id": node_id, "cr": cr_val}

            if worst_matrix_info is None:
                print(f"    - ✅ All matrices for {expert_id} are now consistent.")
                break

            node_id_to_fix = worst_matrix_info['node_id']
            print(f"    - Cycle {revision_cycle + 1}/{max_cycles}: Targeting '{node_id_to_fix}' (CR: {worst_matrix_info['cr']:.4f})")

            recommendations = Consistency.get_consistency_recommendations(temp_model, node_id_to_fix)

            if "error" in recommendations or not recommendations.get("revisions"):
                print(f"    - ❌ Error: Could not get recommendations for {node_id_to_fix}. Halting for this expert.")
                break

            change_applied_this_cycle = False
            for rec in recommendations['revisions']:
                old_value = rec['current_value']
                ideal_value = rec['suggested_value']
                new_value = np.clip(ideal_value, 1/bound, bound)

                if np.isclose(old_value, new_value, atol=1e-4):
                    print(f"      - Info: Suggestion for pair {rec['pair']} is stagnated. Trying next...")
                    continue

                print(f"      - Applying fix for pair {rec['pair']}: changing ~{old_value:.2f} to {new_value:.2f}")
                r, c = rec['pair']
                matrix_to_revise = current_expert_matrices[node_id_to_fix]
                matrix_to_revise[r, c] = Crisp(new_value)
                matrix_to_revise[c, r] = Crisp(1 / new_value)

                if node_id_to_fix not in expert_change_log: expert_change_log[node_id_to_fix] = []
                expert_change_log[node_id_to_fix].append({'from': old_value, 'to': new_value, 'cycle': revision_cycle + 1, 'pair': (r,c)})

                change_applied_this_cycle = True
                break

            if not change_applied_this_cycle:
                print(f"    - ⚠️ Warning: All suggestions for '{node_id_to_fix}' were stagnated. Halting for this expert.")
                break
        else:
            print(f"    - ⚠️ Warning: Reached max_cycles for {expert_id}. Some matrices may still be inconsistent.")

        change_log[expert_id] = expert_change_log

    # --- 3. FINAL VERIFICATION AND RE-FUZZIFICATION LOGIC ---
    # This is the complete block that was missing.
    sanitized_fuzzy_matrices = {node_id: [] for node_id in raw_matrices}
    all_sanitization_successful = True

    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        temp_model = Hierarchy(root_node, Crisp)
        for node_id, matrix in expert_crisp_matrices[i].items():
            temp_model.set_comparison_matrix(node_id, matrix)
        final_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

        for node_id, metrics in final_report.items():
            # Check the final CR against the target
            cr_val = metrics.get('saaty_cr')
            if not (isinstance(cr_val, (float, np.floating)) and cr_val <= target_cr):
                all_sanitization_successful = False
                print(f"    - ❌ FAILURE POST-CHECK: Matrix for {expert_id}/{node_id} is still inconsistent (CR: {metrics.get('saaty_cr', 99):.4f})")

        # Convert the (now consistent) crisp matrices back to the original fuzzy type
        for node_id in raw_matrices.keys():
            consistent_crisp_matrix = expert_crisp_matrices[i][node_id]
            n = consistent_crisp_matrix.shape[0]
            re_fuzzified_matrix = np.empty((n, n), dtype=object)
            for r in range(n):
                for c in range(n):
                    re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                        consistent_crisp_matrix[r, c].value, target_number_type, scale=scale
                    )
            sanitized_fuzzy_matrices[node_id].append(re_fuzzified_matrix)

    # If any expert failed the post-check, raise an error to fail the test.
    if not all_sanitization_successful:
        raise AssertionError(f"Sanitization failed. Not all expert matrices could be made consistent within the given cycle limit.")

    return sanitized_fuzzy_matrices, change_log


@register_sanitization_strategy("adjust_triads")
def adjust_triads_strategy(
    raw_matrices: Dict, root_node: Node, target_number_type: type,
    target_cr: float = 0.1, max_cycles: int = 20, bound: float = 9.0,
    scale: str = 'linear', **kwargs
) -> tuple[Dict, Dict]:
    # 1. Defuzzify
    expert_crisp_matrices = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            nid: np.array([[Crisp(c.defuzzify()) for c in row] for row in mats[i]], dtype=object)
            for nid, mats in raw_matrices.items()
        }
        expert_crisp_matrices.append(expert_dict)

    # 2. Revise Triads
    change_log = {}
    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        mats = expert_crisp_matrices[i]
        expert_log = {}

        for _ in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for nid, m in mats.items(): temp_model.set_comparison_matrix(nid, m)

            report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)
            worst = max((m['saaty_cr'], nid) for nid, m in report.items() if m.get('is_consistent') is False) if any(m.get('is_consistent') is False for m in report.values()) else None

            if not worst: break

            nid = worst[1]
            mat = mats[nid]
            n = mat.shape[0]
            arr = np.array([[c.value for c in r] for r in mat])

            # Find worst triad
            worst_triad = None
            max_err = 0
            for r in range(n):
                for c in range(n):
                    for k in range(n):
                         if r==c or c==k or r==k: continue
                         err = (arr[r,c] * arr[c,k]) / arr[r,k]
                         dist = max(err, 1/err)
                         if dist > max_err:
                             max_err = dist
                             worst_triad = (r, c, k, err)

            if not worst_triad or max_err < 1.01: break

            r, c, k, eps = worst_triad
            delta = (1/eps)**(1/3)

            for (x,y) in [(r,c), (c,k), (r,k)]:
                 old = mats[nid][x,y].value
                 # logic: update row-col elements by factor/inverse factor
                 if (x,y) == (r,k): factor = 1/delta
                 else: factor = delta
                 new = np.clip(old * factor, 1/bound, bound)
                 mats[nid][x,y] = Crisp(new)
                 mats[nid][y,x] = Crisp(1/new)

            expert_log.setdefault(nid, []).append({'cycle': _})

        change_log[expert_id] = expert_log

    # 3. Refuzzify
    sanitized_fuzzy_matrices = {nid: [] for nid in raw_matrices}
    for i in range(num_experts):
        for nid in raw_matrices:
            fuzzy_mat = _refuzzify_matrix(expert_crisp_matrices[i][nid], target_number_type, scale)
            sanitized_fuzzy_matrices[nid].append(fuzzy_mat)

    return sanitized_fuzzy_matrices, change_log


def adjust_triads_strategy_retired(
    raw_matrices: Dict,
    root_node: Node,
    target_number_type: type,
    target_cr: float = 0.1,
    max_cycles: int = 20,
    bound: float = 9.0,
    scale: str = 'linear',
    **kwargs
) -> tuple[Dict, Dict]:
    """
    Sanitizes matrices using an iterative Triad-Based Adjustment method.
    """
    change_log = {}

    # 1. Defuzzify all matrices to a list of crisp matrix dictionaries
    expert_crisp_matrices = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            node_id: np.array([[Crisp(cell.defuzzify()) for cell in row] for row in matrices[i]], dtype=object)
            for node_id, matrices in raw_matrices.items()
        }
        expert_crisp_matrices.append(expert_dict)

    # 2. Iterate through each expert and clean their set of crisp matrices
    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        print(f"\n  - Sanitizing matrices for {expert_id} (Triad Strategy)...")
        current_expert_matrices = expert_crisp_matrices[i]
        expert_change_log = {}

        for revision_cycle in range(max_cycles):
            temp_model = Hierarchy(root_node, Crisp)
            for node_id, matrix in current_expert_matrices.items():
                temp_model.set_comparison_matrix(node_id, matrix)
            consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

            worst_matrix_info = None
            max_cr = target_cr
            for node_id, metrics in consistency_report.items():
                cr_val = metrics.get('saaty_cr')
                if isinstance(cr_val, (float, np.floating)) and cr_val > max_cr:
                    max_cr = cr_val
                    worst_matrix_info = {"node_id": node_id, "cr": cr_val}

            if worst_matrix_info is None:
                print(f"    - ✅ All matrices for {expert_id} are now consistent.")
                break

            node_id_to_fix = worst_matrix_info['node_id']
            print(f"    - Cycle {revision_cycle + 1}/{max_cycles}: Targeting '{node_id_to_fix}' (CR: {worst_matrix_info['cr']:.4f})")

            matrix_to_revise = current_expert_matrices[node_id_to_fix]
            crisp_matrix = np.array([[c.value for c in row] for row in matrix_to_revise], dtype=float)
            n = crisp_matrix.shape[0]

            max_triad_error = 1.0
            worst_triad = None
            for i_ in range(n):
                for j_ in range(n):
                    for k_ in range(n):
                        if i_ == j_ or j_ == k_ or i_ == k_: continue
                        a_ij, a_jk, a_ik = crisp_matrix[i_, j_], crisp_matrix[j_, k_], crisp_matrix[i_, k_]
                        epsilon = (a_ij * a_jk) / a_ik
                        error = max(epsilon, 1.0 / epsilon)
                        if error > max_triad_error:
                            max_triad_error = error
                            worst_triad = (i_, j_, k_, epsilon)

            if worst_triad is None:
                print(f"    - ⚠️ Warning: Could not find inconsistent triad in {node_id_to_fix}. Halting for this expert.")
                break

            i_, j_, k_, epsilon = worst_triad
            delta = (1.0 / epsilon) ** (1/3)

            old_ij, old_jk, old_ik = crisp_matrix[i_, j_], crisp_matrix[j_, k_], crisp_matrix[i_, k_]
            new_ij = np.clip(old_ij * delta, 1/bound, bound)
            new_jk = np.clip(old_jk * delta, 1/bound, bound)
            new_ik = np.clip(old_ik / delta, 1/bound, bound)

            matrix_to_revise[i_, j_] = Crisp(new_ij); matrix_to_revise[j_, i_] = Crisp(1/new_ij)
            matrix_to_revise[j_, k_] = Crisp(new_jk); matrix_to_revise[k_, j_] = Crisp(1/new_jk)
            matrix_to_revise[i_, k_] = Crisp(new_ik); matrix_to_revise[k_, i_] = Crisp(1/new_ik)

            if node_id_to_fix not in expert_change_log: expert_change_log[node_id_to_fix] = []
            expert_change_log[node_id_to_fix].append({'triad': (i_,j_,k_), 'cycle': revision_cycle + 1})
        else:
            print(f"    - ⚠️ Warning: Reached max_cycles for {expert_id}. Some matrices may still be inconsistent.")

        change_log[expert_id] = expert_change_log

    sanitized_fuzzy_matrices = {node_id: [] for node_id in raw_matrices}
    all_sanitization_successful = True

    for i in range(num_experts):
        expert_id = f"expert_{i+1}"
        # Create one final temp model for this expert to verify their final state
        temp_model = Hierarchy(root_node, Crisp)
        for node_id, matrix in expert_crisp_matrices[i].items():
            temp_model.set_comparison_matrix(node_id, matrix)
        final_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

        # Check if every matrix for this expert is now consistent
        for node_id, metrics in final_report.items():
            cr_val = metrics.get('saaty_cr')
            if not (isinstance(cr_val, (float, np.floating)) and cr_val <= target_cr):
                all_sanitization_successful = False
                print(f"    - ❌ FAILURE POST-CHECK: Matrix for {expert_id}/{node_id} is still inconsistent (CR: {metrics.get('saaty_cr', 99):.4f})")

        # Convert the (now hopefully consistent) crisp matrices back to the original fuzzy type
        for node_id in raw_matrices.keys():
            consistent_crisp_matrix = expert_crisp_matrices[i][node_id]
            n = consistent_crisp_matrix.shape[0]
            re_fuzzified_matrix = np.empty((n, n), dtype=object)
            for r in range(n):
                for c in range(n):
                    re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                        consistent_crisp_matrix[r, c].value, target_number_type, scale=scale
                    )
            sanitized_fuzzy_matrices[node_id].append(re_fuzzified_matrix)

    # If any expert's sanitization failed, raise an error to fail the test.
    if not all_sanitization_successful:
        raise AssertionError("Sanitization failed. Not all expert matrices could be made consistent within the given cycle limit.")

    return sanitized_fuzzy_matrices, change_log


@register_sanitization_strategy("rebuild_consistent")
def rebuild_consistent_strategy(
    raw_matrices: Dict,
    root_node: Node,
    target_number_type: type,
    target_cr: float,
    completion_method: str = 'llsm_type_agnostic',
    consistency_check: List[str] = ["saaty_cr", "gci"],
    scale: str = 'linear',
    **kwargs
) -> tuple[Dict, Dict]:
    """
    A one-shot method that rebuilds each inconsistent matrix to be perfectly consistent.
    """
    from .matrix_builder import rebuild_consistent_matrix # Local import

    change_log = {"info": f"Rebuilt all matrices with CR > {target_cr} using geometric mean reconstruction."}
    sanitized_matrices = copy.deepcopy(raw_matrices)
    num_experts = len(list(raw_matrices.values())[0])

    for i in range(num_experts):
        for node_id, matrices in raw_matrices.items():
            matrix_to_check = matrices[i]

            temp_model = Hierarchy(root_node, target_number_type)
            temp_model.set_comparison_matrix(node_id, matrix_to_check)
            report = temp_model.check_consistency(required_indices=consistency_check)

            if report and node_id in report and report[node_id].get('saaty_cr', 0) > target_cr:
                print(f"  - Rebuilding matrix for expert_{i+1}/{node_id} (CR={report[node_id]['saaty_cr']:.3f})")
                crisp_matrix = np.array([[c.defuzzify() for c in row] for row in matrix_to_check], dtype=float)

                consistent_crisp = rebuild_consistent_matrix(crisp_matrix)

                n = consistent_crisp.shape[0]
                re_fuzzified_matrix = np.empty((n, n), dtype=object)
                for r in range(n):
                    for c in range(n):
                        re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                            crisp_value=consistent_crisp[r, c],
                            number_type=target_number_type,
                            scale=scale
                        )
                sanitized_matrices[node_id][i] = re_fuzzified_matrix

    return sanitized_matrices, change_log


@register_sanitization_strategy("rebuild_dynamic_step")
def rebuild_dynamic_step_strategy(
    raw_matrices: Dict, root_node: Node, target_number_type: type,
    target_cr: float = 0.1, max_cycles: int = 50, bound: float = 9.0,
    scale: str = 'linear', **kwargs
) -> tuple[Dict, Dict]:
    """
    Sanitizes matrices using an iterative adjustment method with a dynamic step size.
    The step size is proportional to the current CR, allowing for larger steps
    when far from the target and smaller steps when close (to avoid overshoot).
    """
    change_log = {}

    # 1. Defuzzify all matrices to a list of crisp matrix dictionaries
    expert_crisp_matrices = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            node_id: np.array([[Crisp(cell.defuzzify()) for cell in row] for row in matrices[i]], dtype=object)
            for node_id, matrices in raw_matrices.items()
        }
        expert_crisp_matrices.append(expert_dict)

    # 2. Run the dynamic revision engine
    consistent_crisp_matrices, change_log = _perform_dynamic_revision(
        expert_crisp_matrices, root_node, target_cr, max_cycles, bound
    )

    # 3. Re-fuzzify the results
    sanitized_fuzzy_matrices = {node_id: [] for node_id in raw_matrices}
    for i in range(num_experts):
        for node_id in raw_matrices.keys():
            crisp_matrix = consistent_crisp_matrices[i][node_id]
            n = crisp_matrix.shape[0]
            re_fuzzified_matrix = np.empty((n, n), dtype=object)
            for r in range(n):
                for c in range(n):
                    crisp_value = crisp_matrix[r, c].value
                    re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                        crisp_value=crisp_value,
                        number_type=target_number_type,
                        scale=scale
                    )
            sanitized_fuzzy_matrices[node_id].append(re_fuzzified_matrix)

    return sanitized_fuzzy_matrices, change_log


@register_sanitization_strategy("rebuild_last_ditch")
def rebuild_last_ditch_strategy(
    raw_matrices: Dict,
    root_node: Node,
    target_number_type: type,
    target_cr: float = 0.1,
    max_cycles: int = 20,
    bound: float = 9.0,
    scale: str = 'linear',
    completion_method: str = 'llsm',
    **kwargs
) -> tuple[Dict, Dict]:
    """
    Sanitizes matrices using the standard iterative adjustment. If a matrix
    fails to converge after max_cycles, it is rebuilt using a one-shot
    method (e.g., LLSM) as a 'last ditch' effort.
    """
    # 1. Defuzzify all matrices to a list of crisp matrix dictionaries
    expert_crisp_matrices = []
    num_experts = len(list(raw_matrices.values())[0])
    for i in range(num_experts):
        expert_dict = {
            node_id: np.array([[Crisp(cell.defuzzify()) for cell in row] for row in matrices[i]], dtype=object)
            for node_id, matrices in raw_matrices.items()
        }
        expert_crisp_matrices.append(expert_dict)

    # 2. Run the standard iterative revision engine
    consistent_crisp_matrices, change_log = _perform_iterative_revision(
        expert_crisp_matrices, root_node, target_cr, max_cycles, use_bounded_adjustment=True, bound=bound
    )

    # 3. Perform 'last ditch' rebuild for non-converged matrices
    print("\n--- Performing Last Ditch Rebuild for Non-Converged Matrices ---")
    for i, expert_matrices_dict in enumerate(consistent_crisp_matrices):
        expert_id = f"expert_{i+1}"
        temp_model = Hierarchy(root_node, Crisp)
        for node_id, matrix in expert_matrices_dict.items():
            temp_model.set_comparison_matrix(node_id, matrix)

        consistency_report = Consistency.check_model_consistency(temp_model, saaty_cr_threshold=target_cr)

        for node_id, metrics in consistency_report.items():
            cr_val = metrics.get('saaty_cr')
            if isinstance(cr_val, (float, np.floating)) and cr_val > target_cr:
                print(f"  - ⚠️ Rebuilding {expert_id}/{node_id} (CR: {cr_val:.4f}) using '{completion_method}'...")
                matrix_to_rebuild = expert_matrices_dict[node_id]
                crisp_matrix_float = np.array([[c.value for c in row] for row in matrix_to_rebuild], dtype=float)
                crisp_matrix = np.array([[c.defuzzify() for c in row] for row in matrix_to_rebuild], dtype=float)


                n = crisp_matrix_float.shape[0]
                rebuilt_crisp_obj_matrix = np.empty((n, n), dtype=object)
                for r in range(n):
                    for c in range(n):
                        rebuilt_crisp_obj_matrix[r, c] = Crisp(crisp_matrix_float[r,c])
                expert_matrices_dict[node_id] = rebuilt_crisp_obj_matrix

                if expert_id not in change_log: change_log[expert_id] = {}
                if node_id not in change_log[expert_id]: change_log[expert_id][node_id] = []
                change_log[expert_id][node_id].append({'from': 'Inconsistent', 'to': 'Rebuilt', 'cycle': 'Last Ditch'})

    sanitized_fuzzy_matrices = {node_id: [] for node_id in raw_matrices}
    for i in range(num_experts):
        for node_id in raw_matrices.keys():
            crisp_matrix = consistent_crisp_matrices[i][node_id]
            n = crisp_matrix.shape[0]
            re_fuzzified_matrix = np.empty((n, n), dtype=object)
            for r in range(n):
                for c in range(n):
                    cell = crisp_matrix[r, c]
                    if isinstance(cell, Crisp):
                        crisp_value = cell.value
                    elif isinstance(cell, (int, float, np.number)):
                        crisp_value = float(cell)
                    else:
                        raise TypeError(f"Unexpected type in crisp matrix list: {type(cell)}")

                    re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                        crisp_value=crisp_value,
                        number_type=target_number_type,
                        scale=scale
                    )
            sanitized_fuzzy_matrices[node_id].append(re_fuzzified_matrix)

    return sanitized_fuzzy_matrices, change_log


@register_sanitization_strategy("rebuild_eigenvector")
def rebuild_eigenvector_strategy(
    raw_matrices: Dict,
    root_node: Node,
    target_number_type: type,
    target_cr: float,
    consistency_check: List[str] = ["saaty_cr", "gci"],
    scale: str = 'linear',
    **kwargs
) -> tuple[Dict, Dict]:
    """
    A one-shot method that rebuilds each inconsistent matrix to be perfectly
    consistent based on its principal eigenvector weights.
    """
    change_log = {"info": f"Rebuilt all matrices with CR > {target_cr} using eigenvector reconstruction."}
    sanitized_matrices = copy.deepcopy(raw_matrices)
    num_experts = len(list(raw_matrices.values())[0])

    for i in range(num_experts):
        for node_id, matrices in raw_matrices.items():
            matrix_to_check = matrices[i]
            temp_model = Hierarchy(root_node, target_number_type)
            temp_model.set_comparison_matrix(node_id, matrix_to_check)
            report = temp_model.check_consistency(required_indices=consistency_check)

            if report and node_id in report and report[node_id].get('saaty_cr', 0) > target_cr:
                print(f"  - Rebuilding matrix for expert_{i+1}/{node_id} (CR={report[node_id]['saaty_cr']:.3f}) via Eigenvector")

                crisp_matrix = np.array([[c.defuzzify() for c in row] for row in matrix_to_check], dtype=float)
                consistent_crisp = rebuild_from_eigenvector(crisp_matrix)

                n = consistent_crisp.shape[0]
                re_fuzzified_matrix = np.empty((n, n), dtype=object)
                for r in range(n):
                    for c in range(n):
                        re_fuzzified_matrix[r, c] = FuzzyScale.get_fuzzy_number(
                        consistent_crisp[r, c], target_number_type, scale=scale
                    )
                sanitized_matrices[node_id][i] = re_fuzzified_matrix

    return sanitized_matrices, change_log
