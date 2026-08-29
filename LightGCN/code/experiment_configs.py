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
