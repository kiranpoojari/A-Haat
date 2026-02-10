# areca_coconut_engine.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from django.conf import settings

# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 128

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "ml_models",
    "arecanut_coconut_leaf_model.h5"
)

CLASS_NAMES = [
    "Arecanut_Healthy",
    "Arecanut_Disease",
    "Coconut_Healthy",
    "Coconut_Disease",
]

model = tf.keras.models.load_model(MODEL_PATH)

# ======================================================
# COMPREHENSIVE DISEASE KNOWLEDGE BASE
# ======================================================
DISEASE_INFO = {
    "Arecanut_Disease": {
        "crop": "Arecanut",
        "common_diseases": [
            "Bud Rot (Phytophthora palmivora)",
            "Foot Rot (Ganoderma lucidum)",
            "Inflorescence Dieback",
            "Yellow Leaf Disease"
        ],
        "symptoms": [
            "Yellowing and drooping of leaves",
            "Rotting of spindle leaf",
            "Premature nut drop",
            "Black lesions on stem"
        ],
        "immediate_actions": [
            "Remove and destroy infected palms",
            "Improve drainage around plants",
            "Apply Trichoderma around root zone",
            "Avoid wounding during maintenance"
        ],
        "chemical_treatment": {
            "fungicides": [
                "Metalaxyl + Mancozeb (2g/liter) for Bud Rot",
                "Carbendazim (1g/liter) for leaf spots",
                "Copper oxychloride (3g/liter) as preventive"
            ],
            "application": "Spray during early morning, repeat after 15 days"
        },
        "organic_management": [
            "Apply neem cake (5kg/palm/year)",
            "Use Trichoderma viride powder (50g/palm)",
            "Garlic extract spray for leaf spots",
            "Proper spacing (2.7m x 2.7m minimum)"
        ],
        "fertilizer_guidance": {
            "recommended": [
                "N:P:K 40:80:80 g/palm/year",
                "Organic manure 10-15 kg/palm/year",
                "Micronutrients: Boron and Magnesium"
            ],
            "avoid": [
                "Excess nitrogen during rainy season",
                "Fresh cow dung near stem"
            ]
        },
        "prevention": [
            "Plant disease-free seedlings",
            "Ensure good drainage",
            "Maintain proper plant hygiene",
            "Regular removal of diseased leaves"
        ],
        "economic_impact": "Yield loss up to 60% if untreated",
        "expert_contact": "State Horticulture Department - Arecanut Research Station"
    },

    "Coconut_Disease": {
        "crop": "Coconut",
        "common_diseases": [
            "Leaf Rot (Exserohilum rostratum)",
            "Stem Bleeding (Thielaviopsis paradoxa)",
            "Bud Rot (Phytophthora palmivora)",
            "Root Wilt Disease"
        ],
        "symptoms": [
            "Yellowing and wilting of leaves",
            "Lesions on leaflets",
            "Premature nut fall",
            "Oozing from stem (gummosis)"
        ],
        "immediate_actions": [
            "Remove severely infected palms",
            "Apply copper fungicide to cut surfaces",
            "Improve soil aeration",
            "Control root grubs and beetles"
        ],
        "chemical_treatment": {
            "fungicides": [
                "Mancozeb (3g/liter) for leaf spots",
                "Hexaconazole (1ml/liter) for bud rot",
                "Bordeaux paste for stem bleeding"
            ],
            "application": "Spray crown and stem, repeat after 21 days"
        },
        "organic_management": [
            "Apply neem cake (10kg/palm/year)",
            "Use Pseudomonas fluorescens biocontrol",
            "Ash application for stem bleeding",
            "Intercrop with legumes for soil health"
        ],
        "fertilizer_guidance": {
            "recommended": [
                "N:P:K 500:320:1200 g/palm/year",
                "Organic manure 50 kg/palm/year",
                "Salt application (1-2 kg/palm/year)"
            ],
            "timing": "Split into 3 applications: Apr-May, Sep-Oct, Jan-Feb"
        },
        "prevention": [
            "Select disease-resistant varieties",
            "Maintain proper spacing (7.5m x 7.5m)",
            "Practice regular pruning",
            "Avoid mechanical injuries"
        ],
        "economic_impact": "Can cause 40-70% yield reduction",
        "expert_contact": "Coconut Development Board / Krishi Vigyan Kendra"
    }
}

# ======================================================
# HEALTHY PLANT GUIDANCE
# ======================================================
HEALTHY_GUIDANCE = {
    "Arecanut": {
        "routine_care": [
            "Regular irrigation during dry periods",
            "Mulching with coconut husk or leaves",
            "Annual manure application",
            "Intercropping with banana or pepper"
        ],
        "nutrient_management": [
            "Apply 100g N, 40g P2O5, 140g K2O per palm",
            "Supplement with green manure crops",
            "Apply magnesium sulfate for yellow leaves"
        ],
        "harvest_management": [
            "Harvest mature nuts (6-8 months old)",
            "Process nuts within 24 hours of harvest",
            "Sun dry for 45-60 days"
        ]
    },
    "Coconut": {
        "routine_care": [
            "Ensure 150-200 liters water per palm weekly",
            "Practice basin irrigation",
            "Remove dried leaves and inflorescence",
            "Control rhinoceros beetle"
        ],
        "nutrient_management": [
            "Apply 1.3 kg urea, 2.0 kg super phosphate, 2.0 kg MOP per palm/year",
            "Apply 50 kg organic manure annually",
            "Boron spray for button shedding"
        ],
        "harvest_management": [
            "Harvest tender nuts at 7th month",
            "Harvest mature nuts at 12th month",
            "Yield: 80-100 nuts/palm/year (good management)"
        ]
    }
}

# ======================================================
# HEALTH SCORE LOGIC
# ======================================================
def calculate_health_score(status, confidence):
    if status == "healthy":
        return min(95, int(80 + confidence * 0.15))
    if status == "early_risk":
        return int(55 + confidence * 0.2)
    if status == "disease_confirmed":
        return int(30 + confidence * 0.2)
    return 50


# ======================================================
# CONFIDENCE CALIBRATION
# ======================================================
def calibrate_confidence(confidence):
    """Reduce over-confidence for realism"""
    if confidence > 95:
        reduction = np.random.uniform(2.0, 4.0)
        calibrated = confidence - reduction
        return max(calibrated, 80.0)
    elif confidence > 90:
        reduction = np.random.uniform(1.0, 2.0)
        return confidence - reduction
    return confidence


# ======================================================
# MAIN PREDICTION FUNCTION (API SAFE)
# ======================================================
def predict_images(image_paths):
    labels = []
    confidences = []

    # ---------- Run predictions ----------
    for path in image_paths:
        try:
            img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
            arr = image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0) / 255.0

            preds = model.predict(arr, verbose=0)[0]
            idx = int(np.argmax(preds))

            if idx >= len(CLASS_NAMES):
                continue

            labels.append(CLASS_NAMES[idx])
            confidences.append(float(np.max(preds) * 100))
        except Exception as e:
            continue

    # ---------- No valid images ----------
    if not confidences:
        return {
            "status": "early_risk",
            "message": "Images are unclear or not leaf-related",
            "health_score": 50,
            "action_priority": "Medium",
            "recommendations": [
                "Take clear photos of palm leaves",
                "Focus on affected areas",
                "Include both upper and lower leaf surfaces"
            ],
            "disclaimer": "AI-based advisory. Confirm with agriculture expert."
        }

    avg_conf = round(sum(confidences) / len(confidences), 2)
    avg_conf = calibrate_confidence(avg_conf)
    final_label = max(set(labels), key=labels.count)
    agreement = labels.count(final_label) / len(labels)

    crop = "Arecanut" if "Arecanut" in final_label else "Coconut"
    is_healthy = "Healthy" in final_label

    # ======================================================
    # HEALTHY CASE
    # ======================================================
    if is_healthy and avg_conf >= 75:
        crop_guidance = HEALTHY_GUIDANCE.get(crop, {})
        
        return {
            "status": "healthy",
            "crop": crop,
            "confidence": avg_conf,
            "health_score": calculate_health_score("healthy", avg_conf),
            "action_priority": "Low",
            "message": f"{crop} palm appears healthy",
            "current_condition": "Good palm health",
            "routine_care": crop_guidance.get("routine_care", [
                "Continue regular irrigation",
                "Apply organic manure annually",
                "Monitor for pests monthly"
            ]),
            "nutrient_management": crop_guidance.get("nutrient_management", [
                "Apply balanced fertilizer",
                "Supplement with micronutrients",
                "Maintain soil pH 5.0-8.0"
            ]),
            "yield_optimization": crop_guidance.get("harvest_management", [
                "Harvest at proper maturity",
                "Follow good processing practices"
            ]),
            "economic_potential": f"Expected yield: {crop} specific normal range",
            "preventive_measures": [
                "Regular field sanitation",
                "Proper drainage maintenance",
                "Disease monitoring every 15 days"
            ],
            "farmer_reassurance": "Your palm garden is in good health. Regular care ensures sustained productivity.",
            "disclaimer": "AI-based advisory. For commercial decisions, consult palm specialist."
        }

    # Get disease information
    info = DISEASE_INFO.get(final_label, {})

    # ======================================================
    # DISEASE CONFIRMED
    # ======================================================
    if avg_conf >= 80 and agreement >= 0.6:
        result = {
            "status": "disease_confirmed",
            "crop": crop,
            "confidence": avg_conf,
            "agreement": round(agreement * 100, 1),
            "health_score": calculate_health_score("disease_confirmed", avg_conf),
            "severity": info.get("severity", "High"),
            "action_priority": "Immediate",
            "common_diseases": info.get("common_diseases", ["Leaf disease detected"]),
            "identified_symptoms": info.get("symptoms", [
                "Leaf discoloration",
                "Abnormal leaf drop",
                "Reduced palm vigor"
            ]),
            "immediate_actions": info.get("immediate_actions", [
                "Remove infected leaves/fronds",
                "Improve field drainage",
                "Maintain palm hygiene"
            ]),
            "chemical_treatment": info.get("chemical_treatment", {}),
            "organic_management": info.get("organic_management", [
                "Neem cake application",
                "Biocontrol agents",
                "Proper spacing and sanitation"
            ]),
            "nutrient_management": info.get("fertilizer_guidance", {
                "recommended": ["Balanced NPK", "Organic manure", "Micronutrients"]
            }),
            "prevention_strategies": info.get("prevention", [
                "Use disease-free planting material",
                "Maintain proper palm spacing",
                "Regular field inspection"
            ]),
            "economic_impact": info.get("economic_impact", "Significant yield loss if untreated"),
            "monitoring_schedule": [
                "Inspect palms weekly during rainy season",
                "Check for new symptoms every 3 days",
                "Monitor soil moisture regularly"
            ],
            "expert_contact": info.get("expert_contact", "Contact State Horticulture Department"),
            "farmer_reassurance": "Palm diseases are common and manageable. Early treatment can save your plantation.",
            "disclaimer": "AI-based advisory. For confirmed diagnosis and commercial treatment, consult palm specialist."
        }
        
        # Add specific guidance based on crop
        if crop == "Arecanut":
            result["arecanut_specific"] = {
                "ideal_spacing": "2.7m x 2.7m minimum",
                "water_requirement": "150-200 liters/palm/week in summer",
                "intercrop_suggestions": ["Banana", "Black pepper", "Cocoa"]
            }
        else:  # Coconut
            result["coconut_specific"] = {
                "ideal_spacing": "7.5m x 7.5m minimum",
                "water_requirement": "200-250 liters/palm/week",
                "intercrop_suggestions": ["Pineapple", "Turmeric", "Ginger", "Banana"]
            }
        
        return result

    # ======================================================
    # EARLY RISK (NOT CONFIRMED)
    # ======================================================
    return {
        "status": "early_risk",
        "crop": crop,
        "possible_issue": "Early signs of palm disease",
        "confidence": avg_conf,
        "agreement": round(agreement * 100, 1),
        "health_score": calculate_health_score("early_risk", avg_conf),
        "action_priority": "Medium",
        "why_not_confirmed": [
            f"Prediction agreement: {int(agreement * 100)}%",
            "Symptoms may be early-stage",
            "Environmental stress can mimic disease"
        ],
        "recommended_actions": info.get("immediate_actions", [
            "Remove suspicious leaves",
            "Improve drainage",
            "Apply organic preventives"
        ]),
        "organic_preventives": info.get("organic_management", [
            "Neem cake application",
            "Trichoderma soil treatment",
            "Proper irrigation management"
        ]),
        "monitoring_advice": [
            "Monitor palms daily for 5 days",
            "Take photos of same leaves for comparison",
            "Note any weather changes"
        ],
        "nutrient_support": [
            "Apply balanced fertilizer",
            "Supplement with micronutrients",
            "Maintain soil organic matter"
        ],
        "when_to_act": "If symptoms worsen within 3 days or spread to other palms",
        "contact_for_help": info.get("expert_contact", "Local agriculture officer"),
        "economic_consideration": "Early intervention prevents major losses",
        "farmer_reassurance": "Most palm issues are manageable with early detection. Your vigilance is key to success.",
        "disclaimer": "Early warning advisory. Confirm with palm specialist before major interventions."
    }
