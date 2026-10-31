def recommend_action(category, condition, local_rule):
    """
    Explainable rules. These rules do not claim to determine local recyclability;
    local collection instructions take precedence.
    """
    category = (category or "").lower()
    condition = condition or "Unsure"
    local_rule = local_rule or "Check local rules"

    if condition == "Dirty or contaminated":
        return {
            "action": "Clean if safe and practical, or check local disposal guidance",
            "reason": "Food residue or contamination can affect whether an item is accepted. Do not recycle it based on material alone."
        }

    if condition == "Clean and intact" and category in {"cardboard", "glass", "metal", "paper", "plastic"}:
        if category == "cardboard":
            return {
                "action": "Reuse first; otherwise check cardboard recycling",
                "reason": "A clean, dry, sturdy box may be reused for storage or packing. If it is no longer useful, follow local cardboard collection rules."
            }
        if category == "glass":
            return {
                "action": "Reuse if safe; otherwise follow local glass collection rules",
                "reason": "A clean, undamaged jar or container may be reusable. Broken glass needs careful handling and may have a separate collection route."
            }
        if category == "paper":
            return {
                "action": "Reuse the paper where possible; otherwise check paper recycling",
                "reason": "Clean paper may be reused for notes or packing. Wet or heavily soiled paper may not be accepted."
            }
        if category == "metal":
            return {
                "action": "Reuse if suitable; otherwise check metal recycling",
                "reason": "Many metal containers are recyclable, but acceptance and preparation requirements vary by location."
            }
        if category == "plastic":
            if local_rule == "Accepted for recycling locally":
                return {
                    "action": "Reuse if suitable; otherwise use the local plastic recycling stream",
                    "reason": "Plastic type and local collection rules matter. Follow the accepted-item guidance for your area."
                }
            return {
                "action": "Reuse if suitable; check local plastic recycling rules",
                "reason": "Not all plastic types are accepted everywhere. The image model cannot identify every resin type reliably."
            }

    if category == "trash":
        return {
            "action": "Check whether a specialist collection or reuse option exists",
            "reason": "The model's 'trash' category is broad and does not prove the item is non-recyclable."
        }

    return {
        "action": "Check local rules before disposal",
        "reason": "The item's condition or local acceptance is uncertain. Confirm the material and use your local waste authority's instructions."
    }
