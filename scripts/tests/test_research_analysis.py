import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]


class ResearchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ezkl_analysis_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_script(self, name, *args):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPTS / name), *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    @staticmethod
    def load_script(name):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def analyze(self, runs_by_config):
        paths = []
        for i, runs in enumerate(runs_by_config):
            path = self.root / f"config_{i}.json"
            path.write_text(
                json.dumps(
                    {
                        "config": {"d_model": 16 * (i + 1), "t_max": 16, "vocab": 32},
                        "runs": runs,
                    }
                )
            )
            paths.append(path)
        out = self.root / "summary.json"
        stdout = self.run_script("analyze_inner_cols_2d.py", *paths, "--out", out)
        return json.loads(out.read_text()), stdout

    @staticmethod
    def run_record(n, k, seconds, status="OK"):
        return {
            "num_inner_cols": n,
            "logrows": k,
            "num_rows": (1 << k) - 8,
            "prove_s": seconds,
            "prove_rss_mb": (1 << k) / 16,
            "status": status,
        }

    def test_failed_prove_and_verify_do_not_win_or_enter_fits(self):
        summary, stdout = self.analyze(
            [
                [
                    self.run_record(2, 16, 50),
                    self.run_record(4, 16, 2, "FAILED at prove"),
                ],
                [
                    self.run_record(2, 17, 100),
                    self.run_record(4, 17, 1, "FAILED at verify"),
                ],
            ]
        )
        self.assertEqual(set(summary["best_n_by_config"].values()), {2})
        self.assertEqual(
            summary["fixed_n_fit_worst_resid_pct"],
            {
                "N=2/prove_s": 0.0,
                "N=2/prove_gb": 0.0,
            },
        )
        self.assertIn("FAILED at prove", stdout)
        self.assertIn("FAILED at verify", stdout)
        self.assertEqual(len(summary["records"]), 4)

    def test_all_failed_samples_leave_selection_and_fits_empty(self):
        summary, stdout = self.analyze(
            [
                [self.run_record(2, 16, 2, "FAILED at prove")],
                [self.run_record(2, 17, 1, "FAILED at prove")],
            ]
        )
        self.assertEqual(summary["best_n_by_config"], {})
        self.assertEqual(summary["fixed_n_fit_worst_resid_pct"], {})
        self.assertIn("无法比较最优 N", stdout)
        self.assertIn("无法拟合", stdout)

    def test_all_time_fits_above_threshold_do_not_crash(self):
        first = self.run_record(2, 16, 50)
        second = self.run_record(2, 17, 50)
        first.pop("prove_rss_mb")
        second.pop("prove_rss_mb")
        summary, stdout = self.analyze([[first], [second]])
        self.assertGreaterEqual(
            summary["fixed_n_fit_worst_resid_pct"]["N=2/prove_s"], 15
        )
        self.assertIn("达到或超过 15%", stdout)
        self.assertNotIn("N ≤", stdout)

    def test_circuit_limit_partition_and_unclamped_tiers(self):
        tiers = {"within": (25, 24), "enter": (26, 25), "above": (27, 26)}
        paths = []
        for side in (0, 1):
            path = self.root / f"grid_{side}.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "name",
                        "status",
                        "num_rows",
                        "d_model",
                        "seq_len",
                        "n_layers",
                    ],
                )
                writer.writeheader()
                for name, pair in tiers.items():
                    writer.writerow(
                        {
                            "name": name,
                            "status": "OK",
                            "num_rows": (1 << (pair[side] - 1)) + 1,
                            "d_model": 16,
                            "seq_len": 16,
                            "n_layers": 1,
                        }
                    )
            paths.append(path)
        out = self.root / "grid.md"
        stdout = self.run_script(
            "compare_grids.py",
            "--baseline",
            paths[0],
            "--optimized",
            paths[1],
            "--out",
            out,
        )
        self.assertIn(
            "1 within the circuit limit, 1 entering it, 1 remaining above", stdout
        )
        report = out.read_text()
        self.assertIn("public SRS supports `k <= 26`", report)
        self.assertEqual(report.count("N/A (not representable at k="), 3)
        with out.with_suffix(".csv").open() as f:
            rows = {r["name"]: r for r in csv.DictReader(f)}
        for name, (base, opt) in tiers.items():
            self.assertEqual(int(rows[name]["logrows_base"]), base)
            self.assertEqual(int(rows[name]["logrows_opt"]), opt)
            self.assertEqual(int(rows[name]["tiers_dropped"]), 1)

    def test_scale_probe_uses_existing_system_temp_directory(self):
        module = self.load_script("probe_scale_window")
        out = self.root / "scale.json"
        workdirs = []

        def probe(model, data, scale, fusion, wd):
            self.assertTrue(wd.is_dir())
            workdirs.append(wd)
            return {
                "scale": scale,
                "fusion": fusion,
                "stage": "gen-witness",
                "ok": True,
            }

        argv = [
            "probe_scale_window.py",
            "--model-dir",
            "models/test",
            "--scales",
            "7",
            "--out",
            str(out),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(module, "probe", probe),
        ):
            module.main()
        self.assertEqual(json.loads(out.read_text())["usable_window"], [7])
        self.assertTrue(workdirs)
        self.assertTrue(all(not wd.exists() for wd in workdirs))

    def test_gas_runner_preserves_temp_parent_and_cleans_only_its_project(self):
        module = self.load_script("run_t2_onchain_gas")
        parent = self.root / "builds"
        parent.mkdir()
        marker = parent / "keep.txt"
        marker.write_text("existing file")
        workdirs = []

        def run_one(hardfork, args, work):
            self.assertEqual(work.parent, parent)
            self.assertTrue((work / "src" / module.CONTRACT.name).is_file())
            workdirs.append(work)
            return {}

        argv = [
            "run_t2_onchain_gas.py",
            "--temp-dir",
            str(parent),
            "--out",
            str(self.root / "gas"),
            "--hardforks",
            "cancun,prague",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(module.shutil, "which", return_value="anvil"),
            mock.patch.object(module, "run_one", run_one),
            mock.patch.object(module, "write_out"),
        ):
            module.main()
        self.assertEqual(marker.read_text(), "existing file")
        self.assertEqual(len(workdirs), 2)
        self.assertTrue(all(not wd.exists() for wd in workdirs))


if __name__ == "__main__":
    unittest.main()
