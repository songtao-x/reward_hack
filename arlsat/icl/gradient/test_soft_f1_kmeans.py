import importlib.util
import sys
import types
import unittest

import numpy as np
import torch


ANALYSIS_PATH = "/home/songtaow/projects/aip-xiye17/songtaow/reward_hack/arlsat/icl/gradient/analysis.py"


def _install_stubs():
    """Install lightweight stubs so analysis.py can be imported in isolation."""
    if "datasets" not in sys.modules:
        datasets_mod = types.ModuleType("datasets")
        datasets_mod.load_dataset = lambda *args, **kwargs: None
        sys.modules["datasets"] = datasets_mod

    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoTokenizer = object
        transformers_mod.AutoModelForCausalLM = object
        transformers_mod.BitsAndBytesConfig = object
        sys.modules["transformers"] = transformers_mod

    if "peft" not in sys.modules:
        peft_mod = types.ModuleType("peft")
        peft_mod.LoraConfig = object
        peft_mod.get_peft_model = lambda *args, **kwargs: None
        peft_mod.prepare_model_for_kbit_training = lambda *args, **kwargs: None
        sys.modules["peft"] = peft_mod

    if "utils_" not in sys.modules:
        utils_mod = types.ModuleType("utils_")
        utils_mod.result_processer = None
        sys.modules["utils_"] = utils_mod

    if "icl" not in sys.modules:
        sys.modules["icl"] = types.ModuleType("icl")
    if "icl.gradient" not in sys.modules:
        sys.modules["icl.gradient"] = types.ModuleType("icl.gradient")
    if "icl.gradient.gradient_h" not in sys.modules:
        grad_h_mod = types.ModuleType("icl.gradient.gradient_h")
        grad_h_mod.get_gradients_over_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("stub")
        )
        grad_h_mod.load_model_and_tokenizer = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("stub")
        )
        grad_h_mod.layer_selection = lambda *args, **kwargs: []
        sys.modules["icl.gradient.gradient_h"] = grad_h_mod


def _load_analysis_module():
    _install_stubs()
    spec = importlib.util.spec_from_file_location("analysis_under_test", ANALYSIS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSoftF1KMeansAvailability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = _load_analysis_module()

    def setUp(self):
        self.analyzer = self.analysis.GradientAnalyzer(args=None)

    def _make_blobs(self, n_per_class=40, seed=224):
        rng = np.random.default_rng(seed)
        false_blob = rng.normal(loc=-1.5, scale=0.4, size=(n_per_class, 4)).astype(np.float32)
        true_blob = rng.normal(loc=1.5, scale=0.4, size=(n_per_class, 4)).astype(np.float32)
        z = np.concatenate([false_blob, true_blob], axis=0)
        y = np.concatenate([
            np.zeros(n_per_class, dtype=np.int64),
            np.ones(n_per_class, dtype=np.int64),
        ])
        return z, y, false_blob, true_blob

    def test_method_available(self):
        self.assertTrue(hasattr(self.analyzer, "soft_f1_kmeans"))
        self.assertTrue(callable(self.analyzer.soft_f1_kmeans))

    def test_soft_f1_kmeans_output_contract(self):
        z, y, _, _ = self._make_blobs()
        out = self.analyzer.soft_f1_kmeans(z, y, max_iter=60, lr=3e-2, temp=1.0)

        required = {
            "soft_f1",
            "soft_precision",
            "soft_recall",
            "cluster_probs",
            "cluster_labels",
            "binary_pred",
            "class1_cluster",
            "centers",
        }
        self.assertTrue(required.issubset(set(out.keys())))

        self.assertEqual(out["cluster_probs"].shape, (z.shape[0], 2))
        self.assertEqual(out["cluster_labels"].shape, (z.shape[0],))
        self.assertEqual(out["binary_pred"].shape, (z.shape[0],))
        self.assertEqual(out["centers"].shape, (2, z.shape[1]))
        self.assertIn(out["class1_cluster"], (0, 1))

        self.assertGreaterEqual(out["soft_f1"], 0.0)
        self.assertLessEqual(out["soft_f1"], 1.0)
        self.assertGreaterEqual(out["soft_precision"], 0.0)
        self.assertLessEqual(out["soft_precision"], 1.0)
        self.assertGreaterEqual(out["soft_recall"], 0.0)
        self.assertLessEqual(out["soft_recall"], 1.0)

    def test_cluster_analysis_soft_mode_availability(self):
        _, _, false_blob, true_blob = self._make_blobs()
        self.analyzer.false_gradients = torch.tensor(false_blob, dtype=torch.float32)
        self.analyzer.true_gradients = torch.tensor(true_blob, dtype=torch.float32)

        print(self.analyzer.false_gradients)
        print(self.analyzer.true_gradients)

        res = self.analyzer.cluster_analysis(
            use_soft_f1_kmeans=True,
            normalized=True,
            use_pca=False,
            use_svd=False,
            use_t_sne=False,
            do_plot=False,
            soft_f1_max_iter=40,
            soft_f1_lr=3e-2,
            soft_f1_temp=1.0,
        )

        self.assertIn("K-means", res)
        self.assertIn("GMM", res)
        self.assertIn("Aggolomerative", res)
        self.assertIn("Soft-F1-KMeans", res)
        self.assertIn("soft_f1", res)
        self.assertIn("soft_precision", res)
        self.assertIn("soft_recall", res)
        self.assertIn("cluster_probs", res)
        self.assertIn("cluster_labels", res)
        self.assertEqual(res["cluster_probs"].shape[1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
