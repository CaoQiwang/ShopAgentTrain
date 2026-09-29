"""Build the public structured observation returned by ShopSimulator.

This module reads the simulator's current session directly.  It deliberately
does not accept a task goal, reward, or the legacy ``[SEP]`` observation, so
hidden answers and historical pages cannot leak into the agent observation.
"""

from __future__ import annotations


OBSERVATION_VERSION = "my-shopping-observation-v1"


def _compact_list(value, limit=10):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in list(value)[:limit] if str(item).strip()]


def _available_options(product):
    options = product.get("options")
    if isinstance(options, dict):
        return {
            str(name): _compact_list(values, limit=100)
            for name, values in options.items()
        }

    # Keep compatibility with products before ``load_products`` normalizes
    # ``customization_options`` into the simpler ``options`` mapping.
    customization = product.get("customization_options")
    if not isinstance(customization, dict):
        return {}
    result = {}
    for name, raw_values in customization.items():
        values = []
        if isinstance(raw_values, list):
            for item in raw_values[:100]:
                if isinstance(item, dict):
                    if item.get("is_available") is False:
                        continue
                    item = item.get("value")
                if item is not None and str(item).strip():
                    values.append(str(item))
        result[str(name)] = values
    return result


def product_summary(product, rank=None):
    """Return only public fields needed by the shopping agent."""
    price = product.get("Price")
    if price is None:
        price = product.get("pricing")
    summary = {
        "asin": str(product.get("asin", "")),
        "title": str(product.get("title") or product.get("Title") or ""),
        "price": price,
        "brand": str(product.get("brand") or product.get("shop_name") or ""),
        "category": str(
            product.get("category") or product.get("product_category") or ""
        ),
        "key_attributes": _compact_list(
            product.get("attribute") or product.get("Attributes") or []
        ),
    }
    if rank is not None:
        summary["rank"] = int(rank)
    return summary


def build_observation_state(
    *,
    page_type,
    session,
    product_item_dict,
    available_actions,
):
    """Build one current-page-only, answer-free observation object."""
    clickables = [
        str(value)
        for value in available_actions.get("clickables", [])
        if str(value).casefold() != "search"
    ]
    state = {
        "observation_version": OBSERVATION_VERSION,
        "page_type": str(page_type),
        "search_available": bool(available_actions.get("has_search_bar")),
        "actions": clickables,
    }

    if page_type == "search_results":
        all_asins = list(session.get("search_result_asins") or [])
        page_asins = list(session.get("current_page_asins") or [])
        rank_by_asin = {
            str(asin): index for index, asin in enumerate(all_asins, start=1)
        }
        products = []
        for asin in page_asins:
            asin = str(asin)
            product = product_item_dict.get(asin)
            if product is not None:
                products.append(
                    product_summary(product, rank=rank_by_asin.get(asin, 0))
                )
        state.update(
            {
                "query": " ".join(session.get("keywords") or []),
                "page": int(session.get("page") or 1),
                "total_pages": int(session.get("total_pages") or 1),
                "total_results": int(session.get("total_results") or 0),
                "products": products,
            }
        )
    elif page_type in {"product_detail", "information_subpage"}:
        asin = session.get("asin")
        product = product_item_dict.get(asin, {})
        state.update(
            {
                "product": product_summary(product),
                "selected_options": dict(session.get("options") or {}),
                "available_options": _available_options(product),
            }
        )
        if session.get("selected_price") is not None:
            state["selected_price"] = session["selected_price"]
            state["product"]["price"] = session["selected_price"]

        if page_type == "information_subpage":
            subpage = str(session.get("subpage") or "information")
            field = {
                "description": "Description",
                "features": "BulletPoints",
                "reviews": "Reviews",
                "attributes": "Attributes",
            }.get(subpage.casefold())
            state["subpage"] = subpage
            state["content"] = product.get(field, "") if field else ""

    return state


def page_type_from_name(page_name):
    return {
        "": "search_home",
        None: "search_home",
        "search_results": "search_results",
        "item_page": "product_detail",
        "item_sub_page": "information_subpage",
        "done": "terminal",
    }.get(page_name, "unknown")

