"""
generate_unstructured_data.py
Generates an unstructured JSON dataset for the 6 baby meals,
containing free-text fields: descriptions, parent reviews,
doctor notes, preparation steps, and safety remarks.
Run: python generate_unstructured_data.py
"""

import json
import random
from datetime import datetime, timedelta

UNSTRUCTURED_DATA = [
    {
        "meal_id": "BF001",
        "meal_name": "Banana & Oat Puree",
        "emoji": "🍌",
        "timestamp": "2024-01-10T08:30:00",
        "free_text_description": (
            "This is a simple, smooth puree made by mashing one ripe banana "
            "with two tablespoons of rolled oats that have been softened in warm "
            "breast milk or formula. The banana provides natural sweetness and "
            "energy, while the oats contribute fiber and slow-release carbohydrates. "
            "Ideal as a first solid food for babies aged 6 months and above."
        ),
        "preparation_notes": (
            "Peel and slice a ripe banana. Cook oats in water or formula until soft. "
            "Blend both together until completely smooth. Add extra milk to reach "
            "desired consistency. No salt, sugar, or honey should be added. "
            "Serve immediately or refrigerate for up to 24 hours."
        ),
        "doctor_notes": [
            "Suitable for first-stage weaning at 6 months.",
            "Oats may rarely cause sensitivity — monitor after first serving.",
            "Banana is a known binder — useful if baby has loose stools.",
            "Potassium content supports early heart muscle development.",
            "Avoid overripe blackened bananas as sugar content spikes significantly."
        ],
        "parent_reviews": [
            {
                "parent_id": "P1001",
                "baby_age_months": 6,
                "rating": 5,
                "comment": "My baby absolutely loved this! Finished the whole bowl on the first try. No gas or discomfort after eating.",
                "date": "2024-01-15"
            },
            {
                "parent_id": "P1002",
                "baby_age_months": 7,
                "rating": 4,
                "comment": "Easy to prepare and my daughter enjoys it. I added a tiny pinch of cinnamon which made it even better.",
                "date": "2024-02-03"
            },
            {
                "parent_id": "P1003",
                "baby_age_months": 6,
                "rating": 5,
                "comment": "Perfect first food. No allergic reaction, no fussiness. Will keep using this regularly.",
                "date": "2024-02-20"
            }
        ],
        "allergen_warnings": "Contains oats (possible gluten trace). No nuts, dairy, or eggs.",
        "storage_info": "Refrigerate in airtight container for up to 24 hours. Do not freeze after blending.",
        "tags": ["first-food", "6-months", "fruit", "grain", "no-added-sugar", "iron-free"],
        "raw_nutritional_notes": (
            "Per 100g approx: 88kcal, 1.8g protein, 18g carbs, 8g natural sugar "
            "from banana, 1.5g fiber, minimal sodium (<5mg), 280mg potassium. "
            "No cholesterol. Rich in Vitamin B6 and Vitamin C."
        ),
        "ai_score_log": {
            "model_prediction": "Good Quality",
            "final_score": 91.0,
            "child_safety_score": 100.0,
            "nutrition_balance_score": 83.0,
            "recommendation": "Highly Recommended"
        }
    },
    {
        "meal_id": "BF002",
        "meal_name": "Sweet Potato Mash",
        "emoji": "🍠",
        "timestamp": "2024-01-12T10:00:00",
        "free_text_description": (
            "Sweet potato mash is one of the most nutritionally dense first foods "
            "for infants. Steamed and blended sweet potato delivers an outstanding "
            "amount of beta-carotene and Vitamin A, which are critical for eye "
            "development and immune function. The natural orange color also makes "
            "it visually appealing to babies."
        ),
        "preparation_notes": (
            "Peel one medium sweet potato and cut into chunks. Steam for 15-20 "
            "minutes until very soft. Mash or blend with a small amount of olive "
            "oil and cooled boiled water. Do not add salt or butter. "
            "Texture can be adjusted with more water for younger babies."
        ),
        "doctor_notes": [
            "Excellent source of Vitamin A — one serving can provide 100% of baby's daily needs.",
            "Beta-carotene is a powerful antioxidant safe for infants.",
            "Orange coloring of skin (carotenemia) may occur with frequent consumption — harmless.",
            "Suitable from 6 months with no allergy concerns for most infants.",
            "Olive oil addition supports fat-soluble vitamin absorption."
        ],
        "parent_reviews": [
            {
                "parent_id": "P2001",
                "baby_age_months": 6,
                "rating": 5,
                "comment": "The natural sweetness was a hit! My son opened his mouth wide every spoonful. Super easy to make.",
                "date": "2024-01-20"
            },
            {
                "parent_id": "P2002",
                "baby_age_months": 8,
                "rating": 4,
                "comment": "Great taste, but I noticed the orange tinge on my baby's nose after a week. Doctor said it's harmless.",
                "date": "2024-03-01"
            }
        ],
        "allergen_warnings": "No common allergens. Safe for most infants.",
        "storage_info": "Can be frozen in ice cube trays for up to 1 month. Thaw overnight in fridge.",
        "tags": ["6-months", "vegetable", "vitamin-A", "beta-carotene", "freezer-friendly"],
        "raw_nutritional_notes": (
            "Per 100g approx: 76kcal, 1.4g protein, 13g carbs, 4.5g sugar, "
            "2g fiber, 2.5g fat (from olive oil), 30mg sodium, 320mg potassium, "
            "high beta-carotene (580mcg), 12mg Vitamin C, 0.96g Vitamin A."
        ),
        "ai_score_log": {
            "model_prediction": "Good Quality",
            "final_score": 83.0,
            "child_safety_score": 100.0,
            "nutrition_balance_score": 78.0,
            "recommendation": "Highly Recommended"
        }
    },
    {
        "meal_id": "BF003",
        "meal_name": "Chicken & Carrot Puree",
        "emoji": "🍗",
        "timestamp": "2024-01-18T11:15:00",
        "free_text_description": (
            "A savory protein-forward puree combining tender boiled chicken breast "
            "with naturally sweet steamed carrots. This meal is a critical step "
            "in introducing animal protein to babies around 8 months. "
            "Chicken provides heme iron which is highly bioavailable and essential "
            "for preventing iron-deficiency anemia in growing infants."
        ),
        "preparation_notes": (
            "Boil 40g of skinless chicken breast in plain water until fully cooked. "
            "Steam 50g of peeled carrot until very soft. "
            "Blend chicken and carrot together with some cooking water "
            "to achieve a smooth, lump-free consistency. "
            "Do not add salt, stock cubes, or seasonings. "
            "Ensure chicken is fully cooked with no pink parts before blending."
        ),
        "doctor_notes": [
            "Introduce after 8 months — protein load can stress earlier kidney function.",
            "Heme iron from chicken is absorbed 2-3x more efficiently than plant iron.",
            "Combine with Vitamin C-rich food to further boost iron absorption.",
            "Suitable for babies showing interest in thicker textures.",
            "Cholesterol at this level (28mg) is normal and supports infant brain myelination."
        ],
        "parent_reviews": [
            {
                "parent_id": "P3001",
                "baby_age_months": 8,
                "rating": 4,
                "comment": "My pediatrician recommended starting chicken at 8 months. This was well accepted. Took 2-3 tries before baby got used to the savory taste.",
                "date": "2024-02-10"
            },
            {
                "parent_id": "P3002",
                "baby_age_months": 9,
                "rating": 5,
                "comment": "Best meal for iron! My baby's iron levels improved after we added this twice a week. Highly recommend.",
                "date": "2024-03-15"
            },
            {
                "parent_id": "P3003",
                "baby_age_months": 8,
                "rating": 3,
                "comment": "Baby was a bit hesitant at first with the smell but got used to it after a week.",
                "date": "2024-04-01"
            }
        ],
        "allergen_warnings": "Contains chicken (poultry). No other allergens.",
        "storage_info": "Refrigerate up to 24 hours. Freeze in portions for up to 2 months.",
        "tags": ["8-months", "protein", "iron", "savory", "chicken", "first-meat"],
        "raw_nutritional_notes": (
            "Per 100g approx: 72kcal, 9.5g protein (high), 5g carbs, 2.5g sugar, "
            "1.5g fat, 45mg sodium (safe), 28mg cholesterol, 0.6mg zinc, "
            "0.35mg Vitamin B6, 220mg potassium."
        ),
        "ai_score_log": {
            "model_prediction": "Good Quality",
            "final_score": 87.0,
            "child_safety_score": 100.0,
            "nutrition_balance_score": 80.0,
            "recommendation": "Highly Recommended"
        }
    },
    {
        "meal_id": "BF004",
        "meal_name": "Apple & Pear Blend",
        "emoji": "🍎",
        "timestamp": "2024-01-22T09:00:00",
        "free_text_description": (
            "A light, naturally sweet fruit blend combining steamed apple and pear "
            "with a gentle hint of cinnamon. This meal is extremely gentle on the "
            "digestive system and is often recommended when babies experience "
            "constipation or digestive discomfort. The natural fiber from both fruits "
            "supports regular bowel movements without causing gas."
        ),
        "preparation_notes": (
            "Peel and core one small apple and one small pear. "
            "Cut into chunks and steam for 8-10 minutes until completely soft. "
            "Blend until smooth, adding a tiny pinch of cinnamon (optional). "
            "No sugar or honey should ever be added. "
            "For younger babies, strain through a fine sieve for ultra-smooth texture."
        ),
        "doctor_notes": [
            "Pear puree is a classic remedy for infant constipation — safe and effective.",
            "Apple provides pectin fiber which is prebiotic and feeds good gut bacteria.",
            "Cinnamon in tiny amounts is anti-inflammatory and safe after 6 months.",
            "Natural sugar (10g) is acceptable given the fiber content that slows absorption.",
            "Low sodium (<2mg) makes this one of the safest fruits for infant kidneys."
        ],
        "parent_reviews": [
            {
                "parent_id": "P4001",
                "baby_age_months": 6,
                "rating": 5,
                "comment": "Used this when my baby was constipated and it worked within a day! Will keep this as a staple.",
                "date": "2024-02-05"
            },
            {
                "parent_id": "P4002",
                "baby_age_months": 7,
                "rating": 5,
                "comment": "One of my baby's favorites! The natural sweetness without any added sugar is perfect.",
                "date": "2024-02-28"
            }
        ],
        "allergen_warnings": "No common allergens. Safe for virtually all infants.",
        "storage_info": "Refrigerate up to 48 hours. Freeze in ice cube trays for 1 month.",
        "tags": ["6-months", "fruit", "constipation-relief", "digestive", "no-allergens"],
        "raw_nutritional_notes": (
            "Per 100g approx: 52kcal (low), 0.3g protein (low), 13.5g carbs, "
            "10g natural sugars, 1.8g fiber, 0.2g fat, 2mg sodium (very low), "
            "110mg potassium, 5mg Vitamin C."
        ),
        "ai_score_log": {
            "model_prediction": "Good Quality",
            "final_score": 89.0,
            "child_safety_score": 100.0,
            "nutrition_balance_score": 82.0,
            "recommendation": "Highly Recommended"
        }
    },
    {
        "meal_id": "BF005",
        "meal_name": "Lentil & Spinach Soup",
        "emoji": "🫘",
        "timestamp": "2024-01-28T12:00:00",
        "free_text_description": (
            "A nutritional powerhouse combining red lentils and baby spinach — "
            "two of the richest plant-based sources of iron, folate and Vitamin K. "
            "This soup is particularly important for breastfed babies whose iron "
            "stores begin to deplete around 6 months. Lentils also provide "
            "complete plant protein when combined with the amino acids in spinach, "
            "and cumin aids gas reduction during digestion."
        ),
        "preparation_notes": (
            "Rinse 30g red lentils thoroughly. Boil in water with a small piece of "
            "onion and one garlic clove until lentils are very soft (20-25 minutes). "
            "Add 20g fresh baby spinach in the last 3 minutes of cooking. "
            "Add a tiny pinch of cumin. Blend everything until smooth. "
            "Thin with cooking water as needed. Never add salt for babies under 12 months."
        ),
        "doctor_notes": [
            "Red lentils are among the best first iron foods for plant-based diets.",
            "Folate is critical for DNA synthesis and neural tube integrity.",
            "Vitamin K from spinach (80mcg) supports bone mineralization.",
            "Introduce slowly — fiber content (4g) may cause gas initially.",
            "Pair with breast milk or Vitamin C fruit to maximize iron absorption.",
            "Safe from 8 months when digestive system is more mature."
        ],
        "parent_reviews": [
            {
                "parent_id": "P5001",
                "baby_age_months": 9,
                "rating": 5,
                "comment": "My baby is on a plant-based diet and this has been amazing for his iron levels. Pediatrician confirmed iron is now in normal range.",
                "date": "2024-03-05"
            },
            {
                "parent_id": "P5002",
                "baby_age_months": 8,
                "rating": 3,
                "comment": "Caused a bit of gas the first few days. After introducing slowly, baby adjusted and now loves it.",
                "date": "2024-03-22"
            },
            {
                "parent_id": "P5003",
                "baby_age_months": 10,
                "rating": 5,
                "comment": "Super easy to batch cook and freeze. The green color is a bit off-putting but baby doesn't care!",
                "date": "2024-04-10"
            }
        ],
        "allergen_warnings": "No common allergens. Introduce lentils gradually to monitor for gas sensitivity.",
        "storage_info": "Refrigerate up to 3 days. Freeze in portions for up to 3 months.",
        "tags": ["8-months", "plant-protein", "iron", "folate", "vitamin-K", "freezer-friendly"],
        "raw_nutritional_notes": (
            "Per 100g approx: 95kcal, 6.5g protein (excellent for plant source), "
            "16g carbs, 1.5g sugar (very low), 4g fiber (high), 0.8g fat, "
            "20mg sodium (very low), 310mg potassium, 80mcg Vitamin K, "
            "8mg Vitamin C, 28mg magnesium, 0.9mg zinc."
        ),
        "ai_score_log": {
            "model_prediction": "Excellent Quality",
            "final_score": 85.0,
            "child_safety_score": 95.0,
            "nutrition_balance_score": 79.0,
            "recommendation": "Highly Recommended"
        }
    },
    {
        "meal_id": "BF006",
        "meal_name": "Greek Yogurt & Berry Mix",
        "emoji": "🫐",
        "timestamp": "2024-02-05T09:30:00",
        "free_text_description": (
            "A probiotic-rich, calcium-packed snack or meal combining full-fat "
            "Greek yogurt with mashed blueberries and a small piece of banana. "
            "Greek yogurt provides more protein per gram than regular yogurt "
            "and delivers live cultures that colonize the infant gut microbiome. "
            "Blueberries provide one of the highest antioxidant concentrations "
            "of any common baby food, supporting brain cell protection and "
            "long-term cognitive development."
        ),
        "preparation_notes": (
            "Use only full-fat, plain Greek yogurt with no added sugar or flavoring. "
            "Wash blueberries and mash or blend with a fork until smooth. "
            "Mash half a small banana. Mix all three together. "
            "Serve fresh at room temperature — do not heat yogurt. "
            "Do not add honey. For older babies (12+ months), "
            "blueberries can be served quartered rather than mashed."
        ),
        "doctor_notes": [
            "Introduce dairy after 10 months unless there is known family history of dairy allergy.",
            "Full-fat dairy is essential under 2 years — do NOT use low-fat versions for babies.",
            "Live cultures (Lactobacillus, Bifidobacterium) support gut microbiome development.",
            "Calcium content (90mg) supports tooth and bone mineralization.",
            "Blueberry anthocyanins cross the blood-brain barrier and support neural development.",
            "Monitor for dairy allergy: rash, vomiting, or excessive gas after first serving."
        ],
        "parent_reviews": [
            {
                "parent_id": "P6001",
                "baby_age_months": 11,
                "rating": 5,
                "comment": "This is our morning go-to! My daughter claps her hands when she sees the yogurt bowl. The purple color from the blueberries is adorable.",
                "date": "2024-03-10"
            },
            {
                "parent_id": "P6002",
                "baby_age_months": 10,
                "rating": 4,
                "comment": "Great for gut health. My baby had recurring constipation and this helped regulate digestion within a week.",
                "date": "2024-03-25"
            },
            {
                "parent_id": "P6003",
                "baby_age_months": 12,
                "rating": 5,
                "comment": "We've been eating this as a family now — it's that good. Healthy, simple, and baby loves it.",
                "date": "2024-04-05"
            }
        ],
        "allergen_warnings": "Contains DAIRY (cow's milk). Check for dairy allergy before introducing. No nuts or eggs.",
        "storage_info": "Serve fresh. Do not freeze once blueberries are mixed. Store assembled mix max 4 hours in fridge.",
        "tags": ["10-months", "dairy", "probiotic", "calcium", "antioxidant", "brain-health"],
        "raw_nutritional_notes": (
            "Per 100g approx: 88kcal, 5g protein, 11g carbs, 8.5g sugar (natural), "
            "3.5g fat (healthy dairy fat), 0.8g fiber, 28mg sodium (low), "
            "90mg calcium (highest of all meals), 12mg cholesterol, "
            "6mg Vitamin C, 160mg potassium, live probiotic cultures."
        ),
        "ai_score_log": {
            "model_prediction": "Good Quality",
            "final_score": 78.0,
            "child_safety_score": 85.0,
            "nutrition_balance_score": 74.0,
            "recommendation": "Highly Recommended"
        }
    }
]

# ── Metadata wrapper ─────────────────────────────────────────────────────
OUTPUT = {
    "dataset_name": "BabyFoodAI Unstructured Meal Descriptions",
    "version": "1.0.0",
    "created_at": datetime.now().isoformat(),
    "description": (
        "Unstructured free-text dataset for 6 curated baby meals. "
        "Includes narrative descriptions, doctor notes, parent reviews, "
        "preparation instructions, allergen warnings, storage info, "
        "nutritional summaries, and AI model score logs. "
        "Intended for NLP tasks, recommendation systems, or LLM fine-tuning."
    ),
    "fields": [
        "meal_id", "meal_name", "emoji", "timestamp",
        "free_text_description", "preparation_notes", "doctor_notes",
        "parent_reviews", "allergen_warnings", "storage_info",
        "tags", "raw_nutritional_notes", "ai_score_log"
    ],
    "total_meals": len(UNSTRUCTURED_DATA),
    "total_reviews": sum(len(m["parent_reviews"]) for m in UNSTRUCTURED_DATA),
    "meals": UNSTRUCTURED_DATA
}

# ── Save as JSON ─────────────────────────────────────────────────────────
output_path = "baby_meals_unstructured.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(OUTPUT, f, indent=2, ensure_ascii=False)

print(f"✅ Unstructured dataset saved → {output_path}")
print(f"   Total meals:   {OUTPUT['total_meals']}")
print(f"   Total reviews: {OUTPUT['total_reviews']}")
print(f"   Fields per meal: {len(OUTPUT['fields'])}")
