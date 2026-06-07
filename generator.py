import json
import random
from datetime import datetime, timedelta


def generate_synthetic_data(days=500, filename="500_days_synthetic.json"):
    base_date = datetime(2025, 1, 1)
    history = {}

    current_fatigue = 1.0
    current_soreness = 1.0

    for i in range(days):
        current_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")

        # Визначення типу дня (Симплексний розподіл вірогідностей)
        day_type = random.choices(
            ["REST", "LIGHT", "HEAVY", "EXTREME"],
            weights=[0.30, 0.40, 0.25, 0.05],
            k=1
        )[0]

        workout = None

        if day_type == "EXTREME":
            sleep = random.uniform(4.0, 5.5)
            cals = random.uniform(1500, 2000)
            prot = random.uniform(70, 100)
            rpe = random.uniform(9.5, 10.0)
            weight = random.uniform(130, 160)
            reps = random.randint(3, 5)
            sets = 5

            # Екстремальне руйнування (Рівні 4-5)
            current_soreness = min(5.0, current_soreness + random.uniform(2.5, 4.0))
            current_fatigue = min(5.0, current_fatigue + random.uniform(2.5, 4.0))

            workout = {
                "exercises": [{
                    "exerciseId": "ex_heavy", "category": "SQUAT_OR_DEADLIFT",
                    "sets": [{"setIndex": j + 1, "weight": weight, "reps": reps, "rpe": rpe} for j in range(sets)]
                }]
            }

        elif day_type == "HEAVY":
            sleep = random.uniform(6.5, 7.5)
            cals = random.uniform(2800, 3200)
            prot = random.uniform(150, 180)
            rpe = random.uniform(8.0, 9.0)
            weight = random.uniform(90, 120)
            reps = random.randint(6, 8)
            sets = 4

            current_soreness = min(5.0, current_soreness + random.uniform(1.0, 2.0))
            current_fatigue = min(5.0, current_fatigue + random.uniform(1.0, 1.5))

            workout = {
                "exercises": [{
                    "exerciseId": "ex_hypertrophy", "category": "COMPOUND",
                    "sets": [{"setIndex": j + 1, "weight": weight, "reps": reps, "rpe": rpe} for j in range(sets)]
                }]
            }

        elif day_type == "LIGHT":
            sleep = random.uniform(7.5, 8.5)
            cals = random.uniform(3000, 3500)
            prot = random.uniform(160, 190)
            rpe = random.uniform(6.0, 7.0)
            weight = random.uniform(60, 80)
            reps = random.randint(10, 15)
            sets = 3

            current_soreness = max(1.0, current_soreness - random.uniform(0.5, 1.0))
            current_fatigue = max(1.0, current_fatigue - random.uniform(0.5, 1.0))

            workout = {
                "exercises": [{
                    "exerciseId": "ex_pump", "category": "ISOLATION",
                    "sets": [{"setIndex": j + 1, "weight": weight, "reps": reps, "rpe": rpe} for j in range(sets)]
                }]
            }

        else:  # REST
            sleep = random.uniform(8.0, 9.5)
            cals = random.uniform(2500, 3000)
            prot = random.uniform(140, 170)

            # Відновлення гомеостазу
            current_soreness = max(1.0, current_soreness - random.uniform(1.0, 1.5))
            current_fatigue = max(1.0, current_fatigue - random.uniform(1.0, 1.5))

        history[current_date] = {
            "biometrics": {
                "sleepHours": round(sleep, 1),
                "steps": random.randint(4000, 12000),
                "waterMl": random.randint(2000, 4000),
                "caloriesConsumed": round(cals),
                "proteinGrams": round(prot, 1),
                "carbsGrams": round(random.uniform(200, 400), 1),
                "fatGrams": round(random.uniform(60, 100), 1),
                "muscleSoreness": max(1, min(5, round(current_soreness))),
                "cnsFatigue": max(1, min(5, round(current_fatigue)))
            },
            "workout": workout
        }

    dataset = {
        "userId": "athlete_synthetic_500",
        "baseline": {"weightKg": 82.0, "heightCm": 180.0, "gender": "M", "bodyFatPercentage": 14.0},
        "history": history
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"[Успіх] Згенеровано синтетичний датасет на {days} днів: {filename}")


if __name__ == "__main__":
    generate_synthetic_data()