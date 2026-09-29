"""ShopSimulator observation 解析的回归测试。"""

import json
import unittest

from src.env.observation import ObservationConverter


def _raw(page_tokens, *, search_available=False, actions=()):  # type: ignore[no-untyped-def]
    page = " [SEP] ".join(page_tokens)
    return {
        "instruction": (
            f"{page}\n\n搜索功能是否可用: {search_available}"
            f"\n\n可点击的按钮: {json.dumps(list(actions), ensure_ascii=False)}"
        )
    }


class ObservationConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.asin = "907589003170"
        self.converter = ObservationConverter(product_catalog={})

    def test_bare_instruction_echo_is_removed_from_subpage(self) -> None:
        task = "想找一款儿童写字椅，带脚踏和扶手"
        self.converter.set_instruction(f"Instruction: {task}")
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                [task, "真实商品功能", "Back to Search", "< Prev"],
                actions=["back to search", "< prev"],
            ),
            action="click[Features]",
        )

        self.assertEqual(observation["content"], ["真实商品功能"])

    def test_instruction_filter_uses_exact_match(self) -> None:
        task = "想找一款儿童写字椅"
        self.converter.set_instruction(task)
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                [task, f"商品说明：{task}", "Back to Search", "< Prev"],
                actions=["back to search", "< prev"],
            )
        )

        self.assertEqual(observation["content"], [f"商品说明：{task}"])

    def test_instruction_token_is_removed_from_subpage_content(self) -> None:
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                [
                    "Instruction: 想找一款儿童写字椅",
                    "支持五脚底座和重力自锁轮",
                    "Back to Search",
                    "< Prev",
                ],
                actions=["back to search", "< prev"],
            ),
            action="click[Features]",
        )

        self.assertEqual(observation["page_type"], "product_subpage")
        self.assertEqual(observation["content"], ["支持五脚底座和重力自锁轮"])
        self.assertNotIn("想找一款", json.dumps(observation, ensure_ascii=False))

    def test_empty_subpage_does_not_echo_task_as_product_evidence(self) -> None:
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                [
                    "Instruction: 想找一款儿童写字椅",
                    "Back to Search",
                    "< Prev",
                ],
                actions=["back to search", "< prev"],
            ),
            action="click[Features]",
        )

        self.assertEqual(observation["content"], [])

    def test_search_uses_latest_page_anchor_and_ignores_instruction(self) -> None:
        observation = self.converter.convert(
            _raw(
                [
                    "Instruction: 测试任务",
                    "Page 1 (Total results: 1)",
                    "OLD000001",
                    "历史商品",
                    "1.0",
                    "click[OLD000001]",
                    "Page 2 (Total results: 1)",
                    self.asin,
                    "当前商品",
                    "99.0",
                ],
                actions=["back to search", self.asin],
            ),
            action="search[儿童写字椅]",
        )

        self.assertEqual(observation["page_type"], "search_results")
        self.assertEqual(len(observation["products"]), 1)
        self.assertEqual(observation["products"][0]["title"], "当前商品")

    def test_detail_wins_over_historical_search_marker_and_uses_latest_price(self) -> None:
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                [
                    "Instruction: 测试任务",
                    "Page 1 (Total results: 1)",
                    self.asin,
                    "历史标题",
                    "100.0",
                    "历史详情标题",
                    "价格: 818",
                    "当前详情标题",
                    "价格: 958",
                    "Buy Now",
                ],
                actions=["back to search", "< prev", "buy now"],
            ),
            action=f"click[{self.asin}]",
        )

        self.assertEqual(observation["page_type"], "product_detail")
        self.assertEqual(observation["product"]["title"], "当前详情标题")
        self.assertEqual(observation["product"]["price"], "958")

    def test_output_never_exposes_separator(self) -> None:
        self.converter.current_asin = self.asin
        observation = self.converter.convert(
            _raw(
                ["Instruction: 测试任务", "真实说明", "Back to Search", "< Prev"],
                actions=["back to search", "< prev"],
            )
        )
        self.assertNotIn("[SEP]", json.dumps(observation, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
