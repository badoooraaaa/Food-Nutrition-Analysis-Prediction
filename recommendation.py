"""
Composite Recommendation Scoring Logic
Baby Food Quality Assessment System
=====================================
Final Score = Model Confidence (50%) + Nutrition Balance (30%) + Child Safety (20%)
"""


def compute_recommendation_score(predicted_quality_proba: list, nutrition_dict: dict) -> dict:
    """
    Compute a composite recommendation score for baby food.

    Args:
        predicted_quality_proba: list of probabilities from the ensemble model
        nutrition_dict: dict with keys matching the actual dataset columns

    Returns:
        dict with scores, recommendation label, and actionable tips
    """

    # ── A) Model Quality Score (0–100) ─────────────────────────────────
    model_quality_score = float(max(predicted_quality_proba)) * 100

    # ── B) Nutrition Balance Score (0–100) ─────────────────────────────
    scores = []
    tips = []

    # Protein (ideal 1.5–4 g per 100g for baby food)
    protein = nutrition_dict.get("protein_g", 0) or 0
    if 1.5 <= protein <= 4.0:
        scores.append(100)
    elif protein < 1.5:
        scores.append(max(0, 100 - (1.5 - protein) * 30))
        tips.append("⚠️ Protein is lower than ideal for infant growth. Consider adding protein-rich foods.")
    else:
        scores.append(max(0, 100 - (protein - 4.0) * 20))
        tips.append("⚠️ High protein may stress infant kidneys. Choose lower-protein options.")

    # Fiber (ideal 0.5–3 g)
    fiber = nutrition_dict.get("fiber_g", 0) or 0
    if 0.5 <= fiber <= 3.0:
        scores.append(100)
    elif fiber < 0.5:
        scores.append(max(0, 100 - (0.5 - fiber) * 40))
    else:
        scores.append(max(0, 100 - (fiber - 3.0) * 15))
        tips.append("⚠️ High fiber content may cause digestive discomfort in infants.")

    # Sugar (penalize > 5g)
    sugar = nutrition_dict.get("sugar_g", 0) or 0
    if sugar <= 5:
        scores.append(100)
    else:
        scores.append(max(0, 100 - (sugar - 5) * 10))
        tips.append(f"⚠️ Sugar ({sugar:.1f}g) exceeds infant-safe limit of 5g. High sugar is harmful for babies.")

    # Sodium (penalize > 100mg)
    sodium = nutrition_dict.get("sod_mg", 0) or 0
    if sodium <= 100:
        scores.append(100)
    else:
        scores.append(max(0, 100 - (sodium - 100) / 10 * 5))
        tips.append(f"⚠️ Sodium ({sodium:.1f}mg) is high. Infants' kidneys cannot process excess salt.")

    # Fats (penalize > 10g)
    fats = nutrition_dict.get("fats_g", 0) or 0
    if fats <= 10:
        scores.append(100)
    else:
        scores.append(max(0, 100 - (fats - 10) * 8))
        tips.append("⚠️ Total fat is above average. Prefer lower-fat options for daily meals.")

    # Calories (ideal 50–150 kcal per 100g for baby food)
    calories = nutrition_dict.get("calories_kcal", 0) or 0
    if 50 <= calories <= 150:
        scores.append(100)
    elif calories < 50:
        scores.append(max(0, 100 - (50 - calories) * 1.5))
        tips.append("⚠️ Very low calorie content — may not meet infant energy needs.")
    else:
        scores.append(max(0, 100 - (calories - 150) * 0.8))
        tips.append(f"⚠️ High calorie density ({calories:.0f} kcal). Use in moderation.")

    nutrition_balance_score = sum(scores) / len(scores)

    # ── C) Child Safety Score (0–100) ──────────────────────────────────
    child_safety_score = 100.0

    cholesterol = nutrition_dict.get("cholesterol_mg", 0) or 0
    if cholesterol > 50:
        child_safety_score -= 15
        tips.append(f"🚫 Cholesterol ({cholesterol:.1f}mg) is above baby-safe limit of 50mg.")

    if sodium > 200:
        child_safety_score -= 30
        tips.append(f"🚫 Very high sodium ({sodium:.1f}mg). This food is not safe for infants.")
    elif sodium > 100:
        child_safety_score -= 10

    if sugar > 10:
        child_safety_score -= 25
        tips.append(f"🚫 Excessive sugar ({sugar:.1f}g) detected. Avoid this food for babies.")
    elif sugar > 5:
        child_safety_score -= 10

    if calories > 200:
        child_safety_score -= 10

    if fats > 15:
        child_safety_score -= 15
        tips.append(f"🚫 Very high fat content ({fats:.1f}g). Not suitable for daily infant feeding.")

    child_safety_score = max(0.0, min(100.0, child_safety_score))

    # ── Final Composite Score ───────────────────────────────────────────
    final_score = (
        model_quality_score * 0.50
        + nutrition_balance_score * 0.30
        + child_safety_score * 0.20
    )

    # ── Recommendation Label ────────────────────────────────────────────
    if final_score >= 70:
        recommendation = "Highly Recommended 🟢"
        rec_message = "✅ This food is safe and nutritious for your baby. Great choice!"
        rec_color = "#D4EDDA"
        rec_border = "#06D6A0"
    elif final_score >= 45:
        recommendation = "Recommended with Care 🟡"
        rec_message = "⚠️ This food is acceptable occasionally. Consult your pediatrician for regular use."
        rec_color = "#FFF3CD"
        rec_border = "#FFD166"
    else:
        recommendation = "Not Recommended 🔴"
        rec_message = "🚫 This food has safety concerns for infants. Avoid or use only on medical advice."
        rec_color = "#F8D7DA"
        rec_border = "#FF6B6B"

    # Add positive tips if score is good
    if final_score >= 70 and not tips:
        tips.append("✅ Protein levels are appropriate for infant growth.")
        tips.append("✅ Sodium is within safe limits for babies.")
        tips.append("✅ Sugar content is well-controlled.")

    return {
        "final_score": round(final_score, 1),
        "model_quality_score": round(model_quality_score, 1),
        "nutrition_balance_score": round(nutrition_balance_score, 1),
        "child_safety_score": round(child_safety_score, 1),
        "recommendation": recommendation,
        "rec_message": rec_message,
        "rec_color": rec_color,
        "rec_border": rec_border,
        "tips": tips[:5] if tips else ["✅ All nutritional values are within safe ranges for infants."],
    }
