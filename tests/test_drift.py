"""端到端测试：现场校准漂移分析（drift audit，v1.6）。

覆盖：
- 基本计算：锚点按本地时间相邻配对，逐段给出首尾锚点、本地/UTC 跨度、
  UTC/本地速率与漂移 ppm；整体给出首末锚点、总跨度与总漂移；
- 阈值判定：|漂移| 严格大于 max_drift_ppm 才标异常；等于阈值不标；
  正漂移（本地钟偏慢）/负漂移（本地钟偏快）的异常原因与方向；
  整数/浮点/零阈值均可；
- 错误路径：锚点少于 2 个 -> 409 insufficient_anchors（0/1 个锚点）；
  阈值非法（负数/布尔/字符串/NaN/无穷大）-> 400 invalid_threshold；
  缺字段/非对象 -> 400；version_no 非法 -> 400 validation_error；
  设备不存在 -> 404 device_not_found；周期未登记 -> 404 cycle_not_found；
  版本不存在 -> 404 version_not_found；冻结快照中该周期无锚点同样
  409 insufficient_anchors；
- 周期独立：只分析指定周期的锚点，其他周期锚点不参与；
- 历史版本：指定 version_no 只使用发布时冻结的锚点，之后锚点更正不改动
  历史结果；旧版（format_rev=1）快照缺失周期归属按 default 处理；
- 只读：分析后锚点/周期/版本数量与内容均不变；
- 并发：与锚点更正/发布并发时，每次分析响应都是一致快照。

运行：python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import db, server  # noqa: E402


def request(base, method, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


class DriftAuditTestCase(unittest.TestCase):
    server = None
    base = None
    tmpdir = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.server = server.make_server("127.0.0.1", 0, ":memory:")
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmpdir.cleanup()

    def setUp(self):
        self.db_path = os.path.join(self.tmpdir.name, f"{self.id().split('.')[-1]}.db")
        db.init_db(self.db_path)
        self.server.ctx.db_path = self.db_path

    def req(self, method, path, body=None):
        return request(self.base, method, path, body)

    # ------------------------------------------------------------ 工具

    def make_device(self, device_id="D1", name=None):
        payload = {"device_id": device_id}
        if name is not None:
            payload["name"] = name
        return self.req("POST", "/devices", payload)

    def add_anchor(self, device_id, local_ms, utc_ms, cycle_id=None):
        payload = {"local_ms": local_ms, "utc_ms": utc_ms}
        if cycle_id is not None:
            payload["cycle_id"] = cycle_id
        return self.req("POST", f"/devices/{device_id}/anchors", payload)

    def register_cycle(self, device_id, cycle_id, note=None):
        payload = {"cycle_id": cycle_id}
        if note is not None:
            payload["note"] = note
        return self.req("POST", f"/devices/{device_id}/cycles", payload)

    def drift(self, device_id, cycle_id, max_ppm, version_no=None):
        payload = {"max_drift_ppm": max_ppm}
        if version_no is not None:
            payload["version_no"] = version_no
        return self.req(
            "POST", f"/devices/{device_id}/cycles/{cycle_id}/drift-audit", payload)

    # ------------------------------------------------------------ 基本计算

    def test_basic_intervals_and_overall(self):
        """三个锚点配成两段：逐段跨度/速率/漂移 + 首末锚点与整体总漂移。"""
        self.make_device("D1")
        # 第1段：本地 100000ms，UTC 100100ms（本地慢 1000ppm）
        # 第2段：本地 100000ms，UTC 99990ms（本地快 100ppm）
        self.add_anchor("D1", 0, 1700000000000)
        self.add_anchor("D1", 100000, 1700000100100)
        self.add_anchor("D1", 200000, 1700000200090)

        status, body = self.drift("D1", "default", 5000)
        self.assertEqual(status, 200)
        self.assertEqual(body["source"], "current")
        self.assertIsNone(body["version_no"])
        self.assertEqual(body["device_id"], "D1")
        self.assertEqual(body["cycle_id"], "default")
        self.assertEqual(body["max_drift_ppm"], 5000.0)
        self.assertEqual(body["anchor_count"], 3)
        self.assertEqual(body["interval_count"], 2)
        self.assertTrue(body["ok"])
        self.assertEqual(body["abnormal_count"], 0)
        self.assertEqual(body["abnormal_intervals"], [])

        i0, i1 = body["intervals"]
        self.assertEqual(i0["index"], 0)
        self.assertEqual(i0["from_anchor"],
                         {"local_ms": 0, "utc_ms": 1700000000000})
        self.assertEqual(i0["to_anchor"],
                         {"local_ms": 100000, "utc_ms": 1700000100100})
        self.assertEqual(i0["local_span_ms"], 100000)
        self.assertEqual(i0["utc_span_ms"], 100100)
        self.assertAlmostEqual(i0["utc_rate_per_local"], 1.001)
        self.assertEqual(i0["drift_ppm"], 1000.0)
        self.assertFalse(i0["exceeds_threshold"])
        self.assertIsNone(i0["reason"])

        self.assertEqual(i1["index"], 1)
        self.assertEqual(i1["local_span_ms"], 100000)
        self.assertEqual(i1["utc_span_ms"], 99990)
        self.assertAlmostEqual(i1["utc_rate_per_local"], 0.9999)
        self.assertEqual(i1["drift_ppm"], -100.0)
        self.assertFalse(i1["exceeds_threshold"])

        overall = body["overall"]
        self.assertEqual(overall["first_anchor"],
                         {"local_ms": 0, "utc_ms": 1700000000000})
        self.assertEqual(overall["last_anchor"],
                         {"local_ms": 200000, "utc_ms": 1700000200090})
        self.assertEqual(overall["local_span_ms"], 200000)
        self.assertEqual(overall["utc_span_ms"], 200090)
        self.assertAlmostEqual(overall["utc_rate_per_local"], 1.00045)
        self.assertEqual(overall["drift_ppm"], 450.0)
        self.assertFalse(overall["exceeds_threshold"])

    def test_exceeds_threshold_flags_and_slow_fast_reasons(self):
        """超阈值区间标出异常原因：正漂移=本地偏慢，负漂移=本地偏快。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100100)      # +1000 ppm（慢）
        self.add_anchor("D1", 200000, 200040)      # 第二段 UTC 99940 -> -600 ppm（快）

        status, body = self.drift("D1", "default", 500)
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertEqual(body["abnormal_count"], 2)

        i0, i1 = body["intervals"]
        self.assertTrue(i0["exceeds_threshold"])
        reason = i0["reason"]
        self.assertEqual(reason["code"], "drift_exceeds_threshold")
        self.assertEqual(reason["direction"], "local_clock_runs_slow")
        self.assertEqual(reason["limit_ppm"], 500.0)
        self.assertEqual(reason["excess_ppm"], 500.0)
        self.assertIn("偏慢", reason["message"])
        self.assertIn("1000", reason["message"])

        self.assertTrue(i1["exceeds_threshold"])
        self.assertEqual(i1["drift_ppm"], -600.0)
        self.assertEqual(i1["reason"]["direction"], "local_clock_runs_fast")
        self.assertEqual(i1["reason"]["excess_ppm"], 100.0)
        self.assertIn("偏快", i1["reason"]["message"])

        self.assertEqual(body["abnormal_intervals"], [i0, i1])
        # 整体总漂移 +200ppm，未超阈值
        self.assertEqual(body["overall"]["drift_ppm"], 200.0)
        self.assertFalse(body["overall"]["exceeds_threshold"])

    def test_threshold_boundary_strictly_greater(self):
        """|漂移| 恰好等于阈值不标异常；再大 1 个最小单位才标。"""
        self.make_device("D1")
        # 100000ms 本地 vs 100010ms UTC -> 恰好 100 ppm
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100010)
        status, body = self.drift("D1", "default", 100)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertFalse(body["intervals"][0]["exceeds_threshold"])
        self.assertIsNone(body["intervals"][0]["reason"])
        self.assertFalse(body["overall"]["exceeds_threshold"])

        status, body = self.drift("D1", "default", 99.999)
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertTrue(body["intervals"][0]["exceeds_threshold"])

    def test_float_and_zero_threshold(self):
        """浮点阈值正常比较；零阈值下任何非零漂移都标异常，精确 0 漂移不标。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 1000000, 1000001)   # +1 ppm
        self.add_anchor("D1", 2000000, 2000002)   # +1 ppm

        status, body = self.drift("D1", "default", 1.5)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        status, body = self.drift("D1", "default", 0)
        self.assertEqual(status, 200)
        self.assertEqual(body["max_drift_ppm"], 0.0)
        self.assertFalse(body["ok"])
        self.assertEqual(body["abnormal_count"], 2)

        # 恒等映射设备：零阈值下也无异常
        self.make_device("D2")
        self.add_anchor("D2", 0, 5000)
        self.add_anchor("D2", 60000, 65000)
        status, body = self.drift("D2", "default", 0)
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["intervals"][0]["drift_ppm"], 0.0)
        self.assertEqual(body["intervals"][0]["utc_rate_per_local"], 1.0)

    def test_ppm_computed_on_non_equal_spans(self):
        """本地/UTC 跨度不相等的锚点对也按比例计算（ppm 与具体跨度无关）。"""
        self.make_device("D1")
        self.add_anchor("D1", 1000, 1700000000000)
        self.add_anchor("D1", 61000, 1700000060500)  # 本地 60000，UTC 60500
        status, body = self.drift("D1", "default", 10000)
        self.assertEqual(status, 200)
        iv = body["intervals"][0]
        self.assertEqual(iv["local_span_ms"], 60000)
        self.assertEqual(iv["utc_span_ms"], 60500)
        # (500/60000)*1e6 = 8333.333... ppm，速率 60500/60000
        self.assertAlmostEqual(iv["drift_ppm"], 8333.333333)
        self.assertAlmostEqual(iv["utc_rate_per_local"], 60500 / 60000)
        self.assertFalse(iv["exceeds_threshold"])

    # ------------------------------------------------------------ 错误路径

    def test_insufficient_anchors(self):
        """0 个或 1 个锚点 -> 409 insufficient_anchors，details 带锚点数。"""
        self.make_device("D1")
        status, body = self.drift("D1", "default", 100)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "insufficient_anchors")
        self.assertEqual(body["error"]["details"]["anchor_count"], 0)

        self.add_anchor("D1", 0, 1000)
        status, body = self.drift("D1", "default", 100)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "insufficient_anchors")
        self.assertEqual(body["error"]["details"]["anchor_count"], 1)

    def test_invalid_threshold(self):
        """阈值非法（负数/布尔/字符串/None/NaN/无穷大/对象）-> 400 invalid_threshold。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 1000, 1000)
        for bad in (-1, -0.5, True, False, "100", None, [], {}, float("nan"),
                    float("inf"), float("-inf")):
            status, body = self.drift("D1", "default", bad)
            self.assertEqual(status, 400, f"阈值 {bad!r} 应被拒绝")
            self.assertEqual(body["error"]["code"], "invalid_threshold",
                             f"阈值 {bad!r}")

    def test_missing_field_and_bad_payload(self):
        """缺少 max_drift_ppm -> validation_error；非对象请求体 -> validation_error。"""
        self.make_device("D1")
        status, body = self.req("POST", "/devices/D1/cycles/default/drift-audit",
                                {"version_no": 1})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "validation_error")
        self.assertEqual(body["error"]["details"]["missing"], ["max_drift_ppm"])

        status, body = self.req("POST", "/devices/D1/cycles/default/drift-audit",
                                [1, 2, 3])
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "validation_error")

    def test_invalid_version_no(self):
        """version_no 必须是正整数。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 1000, 1000)
        for bad in (0, -1, 1.5, "1", True):
            status, body = self.drift("D1", "default", 100, version_no=bad)
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["code"], "validation_error")

    def test_device_and_cycle_not_found(self):
        """设备不存在 -> 404 device_not_found；周期未登记 -> 404 cycle_not_found。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 1000, 1000)

        status, body = self.drift("NOPE", "default", 100)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "device_not_found")

        status, body = self.drift("D1", "boot-9", 100)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "cycle_not_found")

        # 路径编号非法（'.bad' 不以字母数字开头）-> 400 validation_error（路由层校验）
        status, body = self.req(
            "POST", "/devices/D1/cycles/.bad/drift-audit",
            {"max_drift_ppm": 100})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "validation_error")

    def test_version_not_found(self):
        """版本不存在 -> 404 version_not_found（设备/周期存在也先报版本）。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 1000, 1000)
        status, body = self.drift("D1", "default", 100, version_no=999)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "version_not_found")
        self.assertEqual(body["error"]["details"]["version_no"], 999)

    # ------------------------------------------------------------ 周期独立

    def test_only_named_cycle_anchors_used(self):
        """漂移分析只使用指定周期的锚点：boot-2 单独有 2 锚点即可分析，
        与 default 是否有锚点无关；不同周期相同 local_ms 不互相影响。"""
        self.make_device("D1")
        self.register_cycle("D1", "boot-2", note="第二次开机")
        self.add_anchor("D1", 0, 0)  # default 仅 1 个锚点
        # boot-2：两个锚点，50000ms 本地 vs 49995ms UTC（快 100ppm）
        self.add_anchor("D1", 0, 1700010000000, cycle_id="boot-2")
        self.add_anchor("D1", 50000, 1700010049995, cycle_id="boot-2")

        status, body = self.drift("D1", "boot-2", 200)
        self.assertEqual(status, 200)
        self.assertEqual(body["cycle_id"], "boot-2")
        self.assertEqual(body["anchor_count"], 2)
        self.assertEqual(body["intervals"][0]["drift_ppm"], -100.0)
        self.assertTrue(body["ok"])

        # default 仍只有 1 个锚点 -> 409
        status, body = self.drift("D1", "default", 200)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["details"]["anchor_count"], 1)

    # ------------------------------------------------------------ 历史版本

    def test_version_audit_uses_frozen_anchors(self):
        """指定版本：历史结果只依据发布时冻结的锚点，后续更正不改动历史结果。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100100)   # +1000 ppm
        self.assertEqual(self.req("POST", "/versions")[0], 201)

        # 当前分析（v1 冻结时也是这组锚点）
        status, v1 = self.drift("D1", "default", 500, version_no=1)
        self.assertEqual(status, 200)
        self.assertEqual(v1["source"], "version")
        self.assertEqual(v1["version_no"], 1)
        self.assertEqual(v1["intervals"][0]["drift_ppm"], 1000.0)
        self.assertFalse(v1["ok"])

        # 锚点更正为 +2000 ppm，再发 v2
        status, _ = self.req("PUT", "/devices/D1/anchors",
                             {"anchors": [{"local_ms": 0, "utc_ms": 0},
                                          {"local_ms": 100000, "utc_ms": 100200}]})
        self.assertEqual(status, 200)
        self.assertEqual(self.req("POST", "/versions")[0], 201)

        # v1 历史结果不变；当前与 v2 反映新锚点
        status, v1_again = self.drift("D1", "default", 500, version_no=1)
        self.assertEqual(status, 200)
        self.assertEqual(v1_again["intervals"], v1["intervals"])
        self.assertEqual(v1_again["overall"], v1["overall"])

        status, v2 = self.drift("D1", "default", 500, version_no=2)
        self.assertEqual(status, 200)
        self.assertEqual(v2["intervals"][0]["drift_ppm"], 2000.0)

        status, cur = self.drift("D1", "default", 500)
        self.assertEqual(status, 200)
        self.assertEqual(cur["intervals"][0]["drift_ppm"], 2000.0)

    def test_version_cycle_without_frozen_anchors_is_insufficient(self):
        """冻结快照中指定周期没有锚点（如版本发布后才登记的周期）-> 409。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100100)
        self.assertEqual(self.req("POST", "/versions")[0], 201)

        self.register_cycle("D1", "boot-2")
        self.add_anchor("D1", 0, 9000, cycle_id="boot-2")
        self.add_anchor("D1", 1000, 10001, cycle_id="boot-2")

        # 当前可分析；v1 快照中 boot-2 无锚点 -> 409 insufficient_anchors
        status, body = self.drift("D1", "boot-2", 500)
        self.assertEqual(status, 200)
        status, body = self.drift("D1", "boot-2", 500, version_no=1)
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "insufficient_anchors")
        self.assertEqual(body["error"]["details"]["anchor_count"], 0)
        self.assertEqual(body["error"]["details"]["version_no"], 1)

    # ------------------------------------------------------------ 只读

    def test_audit_is_read_only(self):
        """分析不改任何数据：锚点、周期、声段、版本数量与内容前后一致。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100100)
        self.req("POST", "/versions")
        before = {
            "anchors": self.req("GET", "/devices/D1/anchors")[1],
            "cycles": self.req("GET", "/devices/D1/cycles")[1],
            "versions": self.req("GET", "/versions")[1],
        }
        for body in ({"max_drift_ppm": 500},
                     {"max_drift_ppm": 500, "version_no": 1},
                     {"max_drift_ppm": -1},
                     {"max_drift_ppm": 500, "version_no": 999}):
            self.req("POST", "/devices/D1/cycles/default/drift-audit", body)
        after = {
            "anchors": self.req("GET", "/devices/D1/anchors")[1],
            "cycles": self.req("GET", "/devices/D1/cycles")[1],
            "versions": self.req("GET", "/versions")[1],
        }
        self.assertEqual(before, after)

    # ------------------------------------------------------------ 并发

    def test_concurrent_audit_with_correction_and_publish(self):
        """与锚点更正/发布并发：每次分析响应都必须对应某个一致快照——
        锚点数一致、各段首尾相接、跨度与锚点差值吻合、整体=首末锚点。"""
        self.make_device("D1")
        self.add_anchor("D1", 0, 0)
        self.add_anchor("D1", 100000, 100000)
        errors, results = [], []

        def correct(k):
            try:
                self.req("PUT", "/devices/D1/anchors",
                         {"anchors": [{"local_ms": 0, "utc_ms": 0},
                                      {"local_ms": 100000, "utc_ms": 100000 + k * 50}]})
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def publish():
            try:
                self.req("POST", "/versions")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def audit():
            try:
                status, body = self.drift("D1", "default", 100000)
                if status != 200:
                    errors.append(("audit_status", status, body))
                else:
                    results.append(body)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = []
        for k in range(1, 7):
            threads.append(threading.Thread(target=correct, args=(k,)))
            threads.append(threading.Thread(target=publish))
            threads.append(threading.Thread(target=audit))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertTrue(results)
        for body in results:
            self.assertEqual(body["anchor_count"], 2)
            iv = body["intervals"][0]
            self.assertEqual(iv["local_span_ms"],
                             iv["to_anchor"]["local_ms"] - iv["from_anchor"]["local_ms"])
            self.assertEqual(iv["utc_span_ms"],
                             iv["to_anchor"]["utc_ms"] - iv["from_anchor"]["utc_ms"])
            self.assertEqual(body["overall"]["first_anchor"], iv["from_anchor"])
            self.assertEqual(body["overall"]["last_anchor"], iv["to_anchor"])


LEGACY_SCHEMA = """
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY, name TEXT, created_at TEXT NOT NULL
);
CREATE TABLE anchors (
    device_id TEXT NOT NULL, local_ms INTEGER NOT NULL, utc_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL, PRIMARY KEY (device_id, local_ms)
);
CREATE TABLE segments (
    capture_id TEXT PRIMARY KEY, device_id TEXT NOT NULL,
    local_start_ms INTEGER NOT NULL, local_end_ms INTEGER NOT NULL,
    note TEXT, content_hash TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE versions (
    version_no INTEGER PRIMARY KEY, note TEXT, created_at TEXT NOT NULL
);
CREATE TABLE version_anchors (
    version_no INTEGER NOT NULL, device_id TEXT NOT NULL,
    local_ms INTEGER NOT NULL, utc_ms INTEGER NOT NULL,
    PRIMARY KEY (version_no, device_id, local_ms)
);
CREATE TABLE version_segments (
    version_no INTEGER NOT NULL, capture_id TEXT NOT NULL, device_id TEXT NOT NULL,
    local_start_ms INTEGER NOT NULL, local_end_ms INTEGER NOT NULL, note TEXT,
    status TEXT NOT NULL, utc_start_ms INTEGER, utc_end_ms INTEGER, created_at TEXT,
    PRIMARY KEY (version_no, capture_id)
);
"""


class LegacyVersionDriftTestCase(unittest.TestCase):
    """旧版（format_rev=1）快照：缺失的周期归属按 default 处理。"""

    server = None
    base = None
    tmpdir = None

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.server = server.make_server("127.0.0.1", 0, ":memory:")
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmpdir.cleanup()

    def setUp(self):
        self.db_path = os.path.join(self.tmpdir.name, f"{self.id().split('.')[-1]}.db")
        self._seed_legacy()
        db.init_db(self.db_path)  # 在线迁移
        self.server.ctx.db_path = self.db_path

    def req(self, method, path, body=None):
        return request(self.base, method, path, body)

    def _seed_legacy(self):
        """构造 v1 旧库并发布 v1：冻结锚点 (0,0)-(100000,100100)，+1000 ppm。"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(LEGACY_SCHEMA)
            conn.execute("INSERT INTO devices VALUES ('D1','录音笔','t')")
            conn.execute("INSERT INTO anchors VALUES ('D1',0,0,'t')")
            conn.execute("INSERT INTO anchors VALUES ('D1',100000,100100,'t')")
            canon = json.dumps(
                {"capture_id": "OLD1", "device_id": "D1",
                 "local_start_ms": 1000, "local_end_ms": 9000, "note": None},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
            conn.execute(
                "INSERT INTO segments VALUES ('OLD1','D1',1000,9000,NULL,?,'t')",
                (digest,))
            conn.execute("INSERT INTO versions VALUES (1,'旧版','t')")
            conn.execute("INSERT INTO version_anchors VALUES (1,'D1',0,0)")
            conn.execute("INSERT INTO version_anchors VALUES (1,'D1',100000,100100)")
            conn.execute(
                "INSERT INTO version_segments VALUES (1,'OLD1','D1',1000,9000,NULL,"
                "'located',1001,9009,'t')")
            conn.commit()
        finally:
            conn.close()

    def test_legacy_version_drift_cycle_defaults_to_default(self):
        """旧版快照漂移分析：冻结锚点周期归属为 NULL，按 default 取到并正常计算。"""
        status, body = self.req(
            "POST", "/devices/D1/cycles/default/drift-audit",
            {"max_drift_ppm": 500, "version_no": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["source"], "version")
        self.assertEqual(body["cycle_id"], "default")
        self.assertEqual(body["anchor_count"], 2)
        self.assertEqual(body["intervals"][0]["drift_ppm"], 1000.0)
        self.assertFalse(body["ok"])
        self.assertEqual(body["intervals"][0]["reason"]["direction"],
                         "local_clock_runs_slow")

    def test_legacy_version_drift_frozen_after_correction(self):
        """迁移后更正当前锚点不影响旧版漂移结果。"""
        # 在线迁移把旧锚点归入 default；更正为不同漂移
        status, _ = self.req("PUT", "/devices/D1/anchors",
                             {"anchors": [{"local_ms": 0, "utc_ms": 0},
                                          {"local_ms": 100000, "utc_ms": 100000}]})
        self.assertEqual(status, 200)
        status, body = self.req(
            "POST", "/devices/D1/cycles/default/drift-audit",
            {"max_drift_ppm": 500, "version_no": 1})
        self.assertEqual(status, 200)
        self.assertEqual(body["intervals"][0]["drift_ppm"], 1000.0)
        # 当前时间轴已是恒等映射
        status, cur = self.req(
            "POST", "/devices/D1/cycles/default/drift-audit",
            {"max_drift_ppm": 500})
        self.assertEqual(status, 200)
        self.assertEqual(cur["intervals"][0]["drift_ppm"], 0.0)
        self.assertTrue(cur["ok"])

    def test_legacy_nondefault_cycle_is_insufficient(self):
        """旧版快照中显式查非 default 周期：冻结锚点为 NULL 归属，取不到 -> 409。"""
        # 先在迁移后的库里登记另一个周期（当前数据中存在，不影响历史快照）
        status, _ = self.req("POST", "/devices/D1/cycles", {"cycle_id": "boot-2"})
        self.assertEqual(status, 201)
        status, body = self.req(
            "POST", "/devices/D1/cycles/boot-2/drift-audit",
            {"max_drift_ppm": 500, "version_no": 1})
        self.assertEqual(status, 409)
        self.assertEqual(body["error"]["code"], "insufficient_anchors")
        self.assertEqual(body["error"]["details"]["anchor_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
