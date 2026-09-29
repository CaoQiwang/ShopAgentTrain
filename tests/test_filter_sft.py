"""过滤器离线测试：真实 HTTP 协议替身，无需启动模型或购物环境。"""

import copy
import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from scripts.filter_sft import (
    JudgeClient,
    build_parser,
    filter_dataset,
    load_cache,
    main,
    parse_verdict,
    precheck,
    rejection,
)


GOOD = {"is_excellent": True, "reason": "关键需求及所选规格价格均有购买前证据，操作合法", "issues": []}


def trajectory():
    messages = [
        {"role": "system", "content": "每轮只调用一个工具，至少选择一个规格后购买。"},
        {"role": "user", "content": json.dumps({
            "task": "买一个红色防水背包，不超过100元",
            "observation": {"page_type": "search", "actions": [], "search_available": True},
        }, ensure_ascii=False)},
    ]
    steps = [
        ("search_products", {"query": "红色 防水 背包"}, {
            "page_type": "search_results", "actions": ["12345678"], "search_available": True,
            "products": [{"asin": "12345678", "title": "防水背包"}],
        }),
        ("open_product", {"asin": "12345678"}, {
            "page_type": "product_detail", "actions": ["红色", "Buy Now", "Features"],
            "search_available": False, "product": {"asin": "12345678", "title": "防水背包", "price": "80"},
            "available_options": {"颜色": ["红色"]}, "selected_options": {},
        }),
        ("select_option", {"value": "红色"}, {
            "page_type": "product_detail", "actions": ["红色", "buy now", "Features"],
            "search_available": False, "product": {"asin": "12345678", "title": "防水背包", "price": "80"},
            "available_options": {"颜色": ["红色"]}, "selected_options": {"颜色": "红色"},
        }),
        ("buy_now", {}, {"page_type": "terminal", "actions": [], "search_available": False}),
    ]
    for index, (name, arguments, observation) in enumerate(steps):
        call_id = f"call_{index}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }]})
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps(observation, ensure_ascii=False)})
    return {"messages": messages, "metadata": {"preserve": "原样保留"}}


class PrecheckTests(unittest.TestCase):
    def test_good_short_trajectory_and_product_error_word_are_allowed(self):
        example = trajectory()
        self.assertIsNone(precheck(example["messages"]))
        content = json.loads(example["messages"][5]["content"])
        content["product"]["description"] = "error correction technology"
        example["messages"][5]["content"] = json.dumps(content)
        self.assertIsNone(precheck(example["messages"]))

    def test_latest_page_rejects_historical_buy_now(self):
        messages = trajectory()["messages"]
        messages[7]["content"] = json.dumps({"page_type": "product_subpage", "actions": ["< Prev"]})
        result = precheck(messages)
        self.assertEqual(result["issues"][0]["code"], "invalid_tool_call")
        self.assertEqual(result["issues"][0]["message_index"], 8)

    def test_corrected_tool_error_is_rejected_even_if_later_purchase_succeeds(self):
        messages = trajectory()["messages"]
        failed_call = copy.deepcopy(messages[6])
        failed_call["tool_calls"][0]["id"] = "failed_call"
        messages[6:6] = [failed_call, {
            "role": "tool", "tool_call_id": "failed_call",
            "content": json.dumps({"error": {"type": "action_unavailable", "message": "不可用"}}),
        }]
        self.assertEqual(precheck(messages)["issues"][0]["code"], "environment_correction")

    def test_illegal_argument_shapes_are_rejected(self):
        for arguments in ('{"extra":1}', '[]', '{bad', '{"query":1}'):
            with self.subTest(arguments=arguments):
                messages = trajectory()["messages"]
                messages[2]["tool_calls"][0]["function"]["arguments"] = arguments
                self.assertEqual(precheck(messages)["issues"][0]["code"], "invalid_tool_call")

    def test_missing_tool_result_is_not_a_complete_trajectory(self):
        messages = trajectory()["messages"][:-1]
        self.assertEqual(precheck(messages)["issues"][0]["code"], "incomplete_trajectory")

    def test_mismatched_tool_id_is_rejected(self):
        messages = trajectory()["messages"]
        messages[3]["tool_call_id"] = "unknown"
        self.assertEqual(precheck(messages)["issues"][0]["code"], "incomplete_trajectory")


class VerdictTests(unittest.TestCase):
    def test_thinking_and_fences_are_supported(self):
        content = '<think>内部分析</think>\n```json\n' + json.dumps(GOOD) + '\n```'
        self.assertEqual(parse_verdict(content, 10), GOOD)

    def test_invalid_verdicts_never_pass(self):
        cases = [
            {**GOOD, "is_excellent": "false"},
            {**GOOD, "is_excellent": False},
            {**GOOD, "reason": ""},
            {**GOOD, "unexpected": 1},
            rejection("insufficient_evidence", 999, "缺少防水证据"),
            rejection("invented_code", 8, "缺少防水证据"),
            {**rejection("insufficient_evidence", 8, "缺少防水证据"), "is_excellent": True},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_verdict(json.dumps(value), 10)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.args = build_parser().parse_args(["--model", "test-judge", "--workers", "2", "--retries", "0"])
        self.args.input_path = self.directory / "input.jsonl"
        self.args.output_path = self.directory / "output.jsonl"
        self.args.report_path = self.directory / "reviews.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def write_input(self, examples):
        rows = [json.dumps(example, ensure_ascii=False) for example in examples]
        self.args.input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return rows

    def test_filter_preserves_rows_and_writes_rejection_reasons(self):
        good = trajectory()
        bad = trajectory()
        bad["messages"][8]["tool_calls"][0]["function"]["arguments"] = '{"extra":1}'
        rows = self.write_input([good, bad])
        with patch.object(JudgeClient, "judge", return_value=GOOD) as judge:
            result = filter_dataset(self.args)
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(result["kept"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(self.args.output_path.read_text(encoding="utf-8"), rows[0] + "\n")
        reviews = [json.loads(line) for line in self.args.report_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["line_number"] for item in reviews], [1, 2])
        self.assertEqual(reviews[1]["source"], "rule")
        self.assertTrue(reviews[1]["reason"])

    def test_semantic_rejection_from_llm_is_excluded(self):
        self.write_input([trajectory()])
        verdict = rejection("insufficient_evidence", 8, "购买前缺少所需材料的商品证据")
        with patch.object(JudgeClient, "judge", return_value=verdict):
            result = filter_dataset(self.args)
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(self.args.output_path.read_text(encoding="utf-8"), "")
        self.assertEqual(result["issue_counts"], {"insufficient_evidence": 1})

    def test_errors_do_not_publish_and_resume_retries_only_errors(self):
        examples = [trajectory(), trajectory()]
        examples[1]["metadata"]["different"] = True
        self.write_input(examples)
        self.args.output_path.write_text("old-output\n", encoding="utf-8")
        with patch.object(JudgeClient, "judge", side_effect=[GOOD, RuntimeError("服务失败")]), redirect_stderr(io.StringIO()):
            first = filter_dataset(self.args)
        self.assertFalse(first["output_written"])
        self.assertEqual(self.args.output_path.read_text(encoding="utf-8"), "old-output\n")
        self.args.resume = True
        with patch.object(JudgeClient, "judge", return_value=GOOD) as judge:
            second = filter_dataset(self.args)
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(second["cached"], 1)
        self.assertEqual(second["kept"], 2)
        self.assertTrue(second["output_written"])

    def test_resume_rejudges_changed_model_and_changed_input(self):
        self.write_input([trajectory()])
        with patch.object(JudgeClient, "judge", return_value=GOOD) as judge:
            filter_dataset(self.args)
            self.args.resume = True
            self.assertEqual(filter_dataset(self.args)["cached"], 1)
            self.args.model = "another-judge"
            self.assertEqual(filter_dataset(self.args)["cached"], 0)
            example = trajectory()
            example["metadata"]["changed"] = True
            self.write_input([example])
            self.assertEqual(filter_dataset(self.args)["cached"], 0)
        self.assertEqual(judge.call_count, 3)

    def test_malformed_input_is_error_not_rejection(self):
        self.args.input_path.write_text('{bad json}\n', encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            result = filter_dataset(self.args)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["rejected"], 0)
        self.assertFalse(self.args.output_path.exists())

    def test_resume_recovers_partial_tail_but_rejects_middle_corruption(self):
        self.args.report_path.write_bytes(b'{"status":"error"}\n{"partial"')
        self.assertEqual(load_cache(self.args.report_path, "signature"), {})
        self.assertEqual(self.args.report_path.read_bytes(), b'{"status":"error"}\n')
        self.args.report_path.write_bytes(b'{bad}\n{"status":"error"}\n')
        with self.assertRaises(ValueError):
            load_cache(self.args.report_path, "signature")

    def test_main_refuses_input_output_collision(self):
        self.write_input([trajectory()])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--model", "test", "--input-path", str(self.args.input_path),
                  "--output-path", str(self.args.input_path)])

    def test_broken_http_body_is_retried_and_reported_as_error(self):
        self.write_input([trajectory()])
        self.args.retries = 1
        with (patch("scripts.filter_sft.urlopen", side_effect=IncompleteRead(b"partial")) as request,
              patch("scripts.filter_sft.time.sleep"), redirect_stderr(io.StringIO())):
            result = filter_dataset(self.args)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["errors"], 1)
        self.assertFalse(result["output_written"])

    def test_http_configuration_error_is_not_retried(self):
        self.write_input([trajectory()])
        self.args.retries = 2
        failure = HTTPError("http://local/v1", 400, "Bad Request", {}, None)
        with patch("scripts.filter_sft.urlopen", side_effect=failure) as request, redirect_stderr(io.StringIO()):
            result = filter_dataset(self.args)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["errors"], 1)
        self.assertFalse(result["output_written"])

    def test_http_contract_and_truncated_reply_retry(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps({"choices": [{
                    "finish_reason": "length" if len(requests) == 1 else "stop",
                    "message": {"content": json.dumps(GOOD)},
                }]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.args.model_base_url = f"http://127.0.0.1:{server.server_port}/v1"
            self.args.retries = 1
            self.args.model_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            self.write_input([trajectory()])
            with patch("scripts.filter_sft.time.sleep"), redirect_stdout(io.StringIO()):
                summary = filter_dataset(self.args)
            self.assertEqual(summary["kept"], 1)
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0][0], "/v1/chat/completions")
            payload = requests[0][1]
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertNotIn("tools", payload)
            nested = json.loads(payload["messages"][1]["content"])
            self.assertEqual([m["message_index"] for m in nested["trajectory"]], list(range(10)))
            self.assertEqual(nested["trajectory"][-1]["message"], trajectory()["messages"][-1])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
