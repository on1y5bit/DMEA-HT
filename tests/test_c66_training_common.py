import unittest

import pandas as pd
import torch
from torch import nn

from scripts import c66_training_common as common


class RouteModel(nn.Module):
    def __init__(self, source_trainable: bool) -> None:
        super().__init__()
        self.sources = nn.Module()
        self.sources.image_encoder = nn.Linear(1, 1)
        self.sources.text_encoder = nn.Linear(1, 1)
        self.sources.bio_encoder = nn.Linear(1, 1)
        self.sources.image_projector = nn.Linear(1, 1)
        self.sources.text_projector = nn.Linear(1, 1)
        self.sources.bio_projector = nn.Linear(1, 1)
        self.source_evidence_stack = nn.Linear(1, 1)
        self.multimodal_encoder = nn.Linear(1, 1)
        self.continuous_bio_encoder = nn.Linear(1, 1)
        self.joint_instance_encoder = nn.Linear(1, 1)
        self.patient_readout = nn.Linear(1, 1)
        self.classifier = nn.Linear(1, 1)
        for name, parameter in self.named_parameters():
            if name.startswith(("sources.", "source_evidence_stack.")):
                parameter.requires_grad_(source_trainable)


class C66TrainingCommonTests(unittest.TestCase):
    def test_route_optimizer_reads_route_specific_factors(self) -> None:
        config = {
            "route_training": {
                "route_f": {
                    "learning_rate_factors": {
                        "image_text_encoders": 0.0,
                        "bio_source_encoder": 0.0,
                        "evidence_projectors": 0.0,
                        "cbpi_task_path": 1.0,
                    }
                },
                "route_e": {
                    "learning_rate_factors": {
                        "image_text_encoders": 0.02,
                        "bio_source_encoder": 0.05,
                        "evidence_projectors": 0.10,
                        "cbpi_task_path": 1.0,
                    }
                },
                "base_lr": 1.0e-4,
                "weight_decay": 5.0e-4,
            }
        }
        optimizer, inventory, _ = common.optimizer_and_inventory(RouteModel(False), config, "route", "F")
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(set(inventory.loc[inventory["requires_grad"], "optimizer_group"]), {"cbpi_task_path"})

    def test_group_level_health_accepts_partial_parameter_updates(self) -> None:
        groups = sorted(common.expected_groups("source"))
        gradient = pd.DataFrame(
            [
                {"epoch": 3, "optimizer_group": group, "max_norm": 0.25}
                for group in groups
            ]
        )
        updates = pd.DataFrame(
            [
                {
                    "kind": "module_summary",
                    "optimizer_group": group,
                    "delta_l2": 0.1,
                    "updated": False,
                    "finite": True,
                    "requires_grad": True,
                }
                for group in groups
            ]
        )
        details = common.training_health_details(gradient, updates, "source", 3)
        self.assertTrue(bool(details["training_health_pass"].all()))
        self.assertTrue(common.training_health_pass(gradient, updates, "source", 3))


if __name__ == "__main__":
    unittest.main()
