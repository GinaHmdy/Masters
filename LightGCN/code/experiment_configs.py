"""
Per-dataset experiment configuration -- the single source of truth.

Extracted from run_all_remaining_datasets.py so that driver scripts can import the
hyperparameters WITHOUT pulling in `world`, which parses sys.argv at import time and
would otherwise hijack their own command-line arguments.

This module must stay free of project imports for that reason: pure data only.
"""

DATASET_CONFIGS = {
    "Yahoo": {
        "layer": 2,
        "recdim": 256,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "beta_ppd": 0.07,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.10,
            "astd_topo_gate_init": 0.75,
            "astd_topo_pop_damping": 0.1,
            "astd_highfreq_gamma": 0.1,
            "astd_neg_uniform_ratio": 0.5,
            "astd_neg_matched_ratio": 0.4,
            "astd_neg_hard_ratio": 0.1,
            "astd_fuse_lambda_max": 0.5,
            "astd_tail_phi": 1.0
        }
    },
    "Coat": {
        "layer": 2,
        "recdim": 64,
        "epochs": 50,
        "bpr_batch": 512,
        "lr": 0.001,
        "beta_ppd": 0.20,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.60", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.10,
            "astd_topo_gate_init": 0.60,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.10,
            "astd_neg_matched_ratio": 0.40,
            "astd_neg_uniform_ratio": 0.50,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.30,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    },
    "Gowalla": {
        "layer": 3,
        "recdim": 64,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "beta_ppd": 0.20,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.40,
            "astd_topo_gate_init": 0.40,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.15,
            "astd_neg_matched_ratio": 0.40,
            "astd_neg_uniform_ratio": 0.45,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.35,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    },
    "KuaiRec": {
        "layer": 2,
        "recdim": 256,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "beta_ppd": 2.00,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.35,
            "astd_topo_gate_init": 0.50,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.15,
            "astd_neg_matched_ratio": 0.45,
            "astd_neg_uniform_ratio": 0.40,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.35,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    }
}

# ---------------------------------------------------------------------------------------
# Validation-selected hyperparameters (Phase 5, remediation item 5.7, audit finding AH).
#
# DATASET_CONFIGS above holds the SHIPPED configurations, and they stay there on purpose:
# every recorded result in results/ -- Phase 1 baselines, the Phase 2 ablation, the Phase 3
# gate transfer -- was produced with them, and changing the base would silently invalidate
# what those files mean. The shipped values were, however, test-influenced: they came from
# refine_yahoo_ppd_v6.py, whose score_fn sorts candidates by seed-2024 TEST metrics.
#
# run_phase5_hpo.py re-selected them over 96 configurations reading the [Best Valid Results]
# block only. The winners are below, with the shipped values beside them. Retrained at 5
# seeds (results/phase5_final_*.json) every one lands within 0.4% of the shipped config on
# test and no conclusion changes -- so this records a fixed protocol, not a new result.
#
# To report under validation-selected hyperparameters, use run_phase5_final.py, which reads
# the sweep winner directly. Do not hand-copy these into DATASET_CONFIGS without re-running
# Phases 1-3, or the results/ tree and this file will disagree.
# ---------------------------------------------------------------------------------------
PPD_V6_VALIDATION_SELECTED = {
    "Yahoo": {
        "selected": {"astd_alpha_base": 0.05, "astd_fuse_lambda_max": 0.5, "astd_tail_phi": 1.5, "astd_topo_pop_damping": 0.05},
        "shipped":  {"astd_alpha_base": 0.1, "astd_fuse_lambda_max": 0.5, "astd_tail_phi": 1.0, "astd_topo_pop_damping": 0.1},
        "valid_hypervolume": 0.00540228,
    },
    "Coat": {
        "selected": {"astd_alpha_base": 0.05, "astd_fuse_lambda_max": 0.3, "astd_tail_phi": 2.0, "astd_topo_pop_damping": 0.1},
        "shipped":  {"astd_alpha_base": 0.1, "astd_fuse_lambda_max": 0.3, "astd_tail_phi": 1.5, "astd_topo_pop_damping": 0.05},
        "valid_hypervolume": 0.00005639,
    },
    "Gowalla": {
        "selected": {"astd_alpha_base": 0.6, "astd_tail_phi": 1.5, "astd_topo_pop_damping": 0.1},
        "shipped":  {"astd_alpha_base": 0.4, "astd_tail_phi": 1.5, "astd_topo_pop_damping": 0.05},
        "valid_hypervolume": 0.00050521,
    },
    "KuaiRec": {
        "selected": {"astd_alpha_base": 0.2, "astd_tail_phi": 1.0, "astd_topo_pop_damping": 0.05},
        "shipped":  {"astd_alpha_base": 0.35, "astd_tail_phi": 1.5, "astd_topo_pop_damping": 0.05},
        "valid_hypervolume": 0.00293828,
    },
}
