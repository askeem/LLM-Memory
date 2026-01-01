"""
Deterministic verifier (ground truth) for all task types.

This is used to score the LLM's answer attempts; the verifier is *not* exposed to the LLM.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple, List

import numpy as np


def npv(rate: float, cfs: List[float]) -> float:
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cfs)))

def irr_from_cfs(cfs: List[float]) -> float:
    coeffs = list(cfs)
    roots = np.roots(coeffs)
    real_roots = []
    for z in roots:
        if abs(z.imag) < 1e-8:
            r = float(z.real - 1.0)
            if r > -0.999999:
                real_roots.append(r)
    if real_roots:
        best = min(real_roots, key=lambda r: abs(npv(r, cfs)))
        return float(best)
    # Newton fallback
    r = 0.1
    for _ in range(100):
        f = npv(r, cfs)
        df = sum(-t * cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cfs) if t > 0)
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        if abs(r_new - r) < 1e-12:
            break
        r = r_new
    return float(r)

def capm_ke(rf: float, beta: float, market_premium: float) -> float:
    return float(rf + beta * market_premium)

def wacc(ke: float, kd: float, tax: float, w_d: float, w_e: float) -> float:
    return float(w_e * ke + w_d * kd * (1 - tax))

def unlever_beta(beta_l: float, tax: float, d_to_e: float) -> float:
    return float(beta_l / (1 + (1 - tax) * d_to_e))

def relever_beta(beta_u: float, tax: float, d_to_e: float) -> float:
    return float(beta_u * (1 + (1 - tax) * d_to_e))

def dcf_5y(inp: Dict[str, Any], years: int = 5) -> Dict[str, float]:
    fcf1 = inp["ebit"] * (1 - inp["tax"]) + inp["da"] - inp["capex"] - inp["delta_nwc"]
    fcfs = [fcf1 * ((1 + inp["fcf_growth"]) ** (t - 1)) for t in range(1, years + 1)]
    pv = sum(fcfs[t - 1] / ((1 + inp["wacc"]) ** t) for t in range(1, years + 1))
    fcf5 = fcfs[-1]
    tv = fcf5 * (1 + inp["g_terminal"]) / (inp["wacc"] - inp["g_terminal"])
    pv_tv = tv / ((1 + inp["wacc"]) ** years)
    ev = pv + pv_tv
    eqv = ev + inp["cash"] - inp["debt"]
    pps = eqv / inp["shares"]
    return {
        "fcf1": float(fcf1),
        "enterprise_value": float(ev),
        "equity_value": float(eqv),
        "price_per_share": float(pps),
        "terminal_value_t5": float(tv),
        "pv_terminal_value": float(pv_tv),
    }

def break_even_units(price: float, variable_cost: float, fixed_cost: float) -> float:
    return float(fixed_cost / (price - variable_cost))

def lease_npv(lease_payments: List[float], discount: float, tax: float) -> float:
    return float(sum((-p * (1 - tax)) / ((1 + discount) ** t) for t, p in enumerate(lease_payments, start=1)))

def buy_npv(buy_price: float, discount: float, tax: float, dep_years: int, salvage: float) -> float:
    dep = buy_price / dep_years
    shield = dep * tax
    pv_shields = sum(shield / ((1 + discount) ** t) for t in range(1, dep_years + 1))
    pv_salv = (salvage * (1 - tax)) / ((1 + discount) ** dep_years)
    return float(-buy_price + pv_shields + pv_salv)

def discounted_payback(cfs: List[float], r: float) -> float:
    cum = 0.0
    for t, cf in enumerate(cfs):
        pv = cf / ((1 + r) ** t)
        cum += pv
        if cum >= 0 and t > 0:
            prev = cum - pv
            frac = (0 - prev) / pv if pv != 0 else 0.0
            return float((t - 1) + frac)
    return float("inf")

def mirr(cfs: List[float], finance_rate: float, reinvest_rate: float) -> float:
    n = len(cfs) - 1
    pv_neg = sum(cf / ((1 + finance_rate) ** t) for t, cf in enumerate(cfs) if cf < 0)
    fv_pos = sum(cf * ((1 + reinvest_rate) ** (n - t)) for t, cf in enumerate(cfs) if cf > 0)
    return float((fv_pos / -pv_neg) ** (1 / n) - 1)

def verify(task: Dict[str, Any], model_answer: Dict[str, Any], task_index: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (ok, details).
    details includes expected, got, per-key errors.
    """
    ttype = task["type"]
    exp = task["expected"]
    tol = task["tolerance"]
    got = {}
    errors = {}

    def num(x):
        return float(x)

    try:
        if ttype == "npv_irr":
            got["npv"] = num(model_answer.get("npv"))
            got["irr"] = num(model_answer.get("irr"))
        elif ttype == "wacc":
            got["ke"] = num(model_answer.get("ke"))
            got["wacc"] = num(model_answer.get("wacc"))
        elif ttype == "beta_recap":
            got["beta_u"] = num(model_answer.get("beta_u"))
            got["beta_l_new"] = num(model_answer.get("beta_l_new"))
        elif ttype == "dcf":
            got["fcf1"] = num(model_answer.get("fcf1"))
            got["enterprise_value"] = num(model_answer.get("enterprise_value"))
            got["equity_value"] = num(model_answer.get("equity_value"))
            got["price_per_share"] = num(model_answer.get("price_per_share"))
        elif ttype == "break_even":
            got["break_even_units"] = num(model_answer.get("break_even_units"))
        elif ttype == "lease_vs_buy":
            got["npv_buy"] = num(model_answer.get("npv_buy"))
            got["npv_lease"] = num(model_answer.get("npv_lease"))
            got["preferred"] = str(model_answer.get("preferred")).strip().lower()
        # Meta types:
        elif ttype in ("npv_from_refs", "npv_compare_by_ref"):
            # handled by exp
            for k in task["answer_keys"]:
                if k == "preferred":
                    got[k] = str(model_answer.get(k)).strip()
                else:
                    got[k] = num(model_answer.get(k))
        elif ttype in ("recall_and_transform", "capm_with_relevered_beta", "wacc_mix_sources", "beta_chain", "npv_at_irr",
                       "discounted_payback", "mirr", "dcf_sensitivity", "dcf_with_new_wacc", "break_even_mix",
                       "lease_tax_swap", "lease_discount_sensitivity", "ev_fcf_multiple", "terminal_value_year5",
                       "pv_terminal_value", "tv_share_of_ev", "profitability_index", "eaa",
                       "contribution_margin_ratio", "npv_advantage", "beta_pct_change"):
            for k in task["answer_keys"]:
                if k == "preferred":
                    got[k] = str(model_answer.get(k)).strip().lower()
                else:
                    got[k] = num(model_answer.get(k))

        elif ttype == "policy_learning":
            for k in task["answer_keys"]:
                got[k] = num(model_answer.get(k))

        elif ttype == "banned_method":
            for k in task["answer_keys"]:
                got[k] = num(model_answer.get(k))

        elif ttype == "complex_procedure":
            for k in task["answer_keys"]:
                got[k] = num(model_answer.get(k))

        else:
            return False, {"error": f"Unknown task type: {ttype}"}
    except Exception as e:
        return False, {"error": f"Parse error: {e}", "expected": exp}

    ok = True
    for k in task["answer_keys"]:
        if isinstance(exp.get(k), str):
            if str(got.get(k)).strip().lower() != str(exp[k]).strip().lower():
                ok = False
                errors[k] = {"expected": exp[k], "got": got.get(k)}
        else:
            err = abs(float(got.get(k)) - float(exp[k]))
            errors[k] = {"expected": exp[k], "got": got.get(k), "abs_err": err, "tol": tol[k]}
            if err > float(tol[k]):
                ok = False

    return ok, {"expected": exp, "got": got, "errors": errors}


def compute_expected_for_ref_task(task: Dict[str, Any], base_tasks_by_id: Dict[str, Any], base_answers_by_id: Dict[str, Any]) -> Dict[str, Any]:
    """
    OPTIONAL helper if you want to generate more tasks programmatically.
    Not used by the verifier during runs (we rely on task['expected']).
    """
    raise NotImplementedError
