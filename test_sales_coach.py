"""Offline tests: parsing, roles, quote verification and the full mock pipeline. Run: python3 -m unittest"""
import argparse
import os
import shutil
import tempfile
import unittest

import backends
import sales_coach as sc

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "examples")


class Parsing(unittest.TestCase):
    def test_txt(self):
        rows = sc.parse_file(os.path.join(EX, "2026-03-04_summit-ridge-roofing.txt"))
        self.assertEqual({n for n, _ in rows}, {"Jordan Blake", "Dale Whitaker"})
        self.assertFalse(rows[0][1].startswith("["))

    def test_vtt(self):
        rows = sc.parse_file(os.path.join(EX, "2026-03-11_brightside-dental.vtt"))
        self.assertEqual({n for n, _ in rows}, {"Priya Shah", "Maya Okafor", "Lena Ortiz"})

    def test_srt(self):
        rows = sc.parse_file(os.path.join(EX, "2026-03-18_harbor-line-hvac.srt"))
        self.assertEqual({n for n, _ in rows}, {"Jordan Blake", "Marcus Reyes"})

    def test_otter_and_continuation(self):
        rows = sc.parse_text("Ana Lopez  0:05\nWe need more leads.\nSam: Got it.\nThat makes sense.")
        self.assertEqual(rows, [["Ana Lopez", "We need more leads."], ["Sam", "Got it. That makes sense."]])

    def test_roles(self):
        r = sc.Roles({"reps": ["Jordan Blake"], "team_domain": "seller.com"})
        self.assertEqual(r.role("Jordan"), "REP")
        self.assertEqual(r.role("jordan blake"), "REP")
        self.assertEqual(r.role("sam@seller.com"), "REP")
        self.assertEqual(r.role("Dale Whitaker"), "PROSPECT")
        self.assertEqual(r.role("Unknown"), "UNKNOWN")


class Verification(unittest.TestCase):
    pros = [sc.norm("Honestly it's mostly referrals and word of mouth. I'm on the roof all day, so I'm calling people back at eight at night.")]

    def test_exact(self):
        self.assertEqual(sc.quote_score("mostly referrals and word of mouth", self.pros, 4), 1.0)

    def test_near_verbatim(self):
        self.assertGreaterEqual(sc.quote_score("I'm on the roof all day so I am calling people back at eight at night", self.pros, 4), 0.85)

    def test_invented(self):
        self.assertLess(sc.quote_score(backends.INVENTED, self.pros, 4), 0.85)

    def test_too_short(self):
        self.assertEqual(sc.quote_score("word of mouth", self.pros, 4), 0.0)

    def test_ellipsis_parts_must_all_match(self):
        self.assertLess(sc.quote_score("mostly referrals and word of mouth ... we hate our current agency deeply", self.pros, 4), 0.85)


class MockPipeline(unittest.TestCase):
    def test_end_to_end(self):
        tmp = tempfile.mkdtemp()
        try:
            cfg = sc.load_config(os.path.join(HERE, "coach.config.json"))
            cfg["cache"] = os.path.join(tmp, "cache.sqlite")
            cfg["reports_dir"] = tmp
            be = backends.make("mock", cfg["model"])
            db = sc.open_cache(cfg)
            paths = sc.list_transcripts(EX)
            keys, failed = sc.analyze_files(paths, cfg, be, db, sc.Roles(cfg), quiet=True)
            self.assertEqual(failed, [])
            calls = [sc.score_call(r, cfg) for r in sc.load_rows(db, hashes=set(keys.values()))]
            self.assertEqual(len(calls), 3)
            # the mock invents one quote per call; the verifier must drop it
            self.assertTrue(all(c["rejected"] for c in calls))
            self.assertTrue(all(backends.INVENTED not in str(c["pain"]) + str(c["obj"]) for c in calls))
            harbor = next(c for c in calls if "harbor" in c["file"])
            self.assertIn("price_too_high", harbor["obj"])
            args = argparse.Namespace(folder=EX, top=10, out=os.path.join(tmp, "report.md"))
            sc.cmd_report(args, cfg, db, be, keys)
            with open(args.out) as f:
                self.assertIn("## Coaching priorities", f.read())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
