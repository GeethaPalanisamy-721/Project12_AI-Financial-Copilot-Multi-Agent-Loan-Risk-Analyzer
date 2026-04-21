from src.agents.decision_agent import final_decision

# 🔥 Sample customer (must match your dataset columns)
sample = {
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "EXT_SOURCE_2": 0.7,
    "EXT_SOURCE_3": 0.65,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -4000,
    "AMT_GOODS_PRICE": 450000,
    "CODE_GENDER": 1,
    "FLAG_OWN_CAR": 1,
    "NAME_EDUCATION_TYPE": 2
}

# 🚀 Run decision
result = final_decision(sample)

print("\n🧠 FINAL DECISION OUTPUT:\n")
for key, value in result.items():
    print(f"{key}: {value}")
