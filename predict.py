import json, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w_ih = nn.Linear(input_size, hidden_size)
        self.w_yh = nn.Linear(output_size, hidden_size, bias=False)
        self.w_ho = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        y_prev = torch.zeros(
            batch_size,
            self.output_size,
            device=x.device
        )

        for t in range(seq_len):
            h_t = self.tanh(
                self.w_ih(x[:, t, :])
                + self.w_yh(y_prev)
            )

            y_prev = self.w_ho(h_t)

        return y_prev


class AdvancedSportsParser:
    def __init__(self, json_path):
        self.json_path = json_path

    def execute_pipeline(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        history = data.get("history", {})
        X_raw = []

        for date in sorted(history.keys()):
            day = history[date]

            biometrics = day.get("biometrics", {})
            workout = day.get("workout", {}) or {}
            exercises = workout.get("exercises", [])

            total_tonnage = sum(
                s.get("weight", 0.0) * s.get("reps", 0)
                for ex in exercises
                for s in ex.get("sets", [])
            )

            rpe_list = [
                s.get("rpe", 0.0)
                for ex in exercises
                for s in ex.get("sets", [])
            ]

            X_raw.append([
                total_tonnage,
                max(rpe_list) if rpe_list else 0.0,

                biometrics.get("sleepHours", 7.0),
                biometrics.get("steps", 5000),
                biometrics.get("waterMl", 2500),
                biometrics.get("caloriesConsumed", 2500.0),
                biometrics.get("proteinGrams", 150.0),
                biometrics.get("carbsGrams", 300.0),
                biometrics.get("fatGrams", 80.0)
            ])

        return np.array(X_raw, dtype=np.float32)


class IntelligentSystemApp:
    def __init__(self):
        self.meta_path = "model_metadata.json"
        self.weights_path = "jordan_model_weights.pth"
        self.new_data_path = "../../../PycharmProjects/PythonProject/new_user_data.json"

    def load_model(self):
        if not (
            os.path.exists(self.weights_path)
            and os.path.exists(self.meta_path)
        ):
            print("[Помилка] Файли моделі відсутні.")
            return None

        with open(self.meta_path, 'r') as f:
            meta = json.load(f)

        model = JordanRNN(
            input_size=9,
            hidden_size=meta["hidden_size"],
            output_size=2
        )

        model.load_state_dict(
            torch.load(self.weights_path)
        )

        model.eval()

        return (
            model,
            np.array(meta["max_X"], dtype=np.float32),
            np.array(meta["max_Y"], dtype=np.float32)
        )

    def parse_targets(self):
        with open(self.new_data_path, 'r') as f:
            data = json.load(f)

        history = data.get("history", {})
        Y_raw = []

        for date in sorted(history.keys()):
            biometrics = history[date].get("biometrics", {})

            Y_raw.append([
                biometrics.get("muscleSoreness", 1.0),
                biometrics.get("cnsFatigue", 1.0)
            ])

        return np.array(Y_raw, dtype=np.float32)

    def generate_autoregressive_forecast(self):
        loaded = self.load_model()

        if not loaded:
            return

        model, max_X, max_Y = loaded

        parser = AdvancedSportsParser(
            self.new_data_path
        )

        X_norm = (
            parser.execute_pipeline()
            / max_X
        )

        try:
            forecast_days = int(
                input("\nВведіть кількість днів прогнозу: ")
            )

        except ValueError:
            print("[Помилка] Некоректне значення.")
            return

        avg_lifestyle = X_norm[:, 2:].mean(axis=0)

        print("\n" + "=" * 60)
        print("БАЗОВИЙ ПРОФІЛЬ ВІДНОВЛЕННЯ")
        print("=" * 60)

        print(
            f"Сон: {avg_lifestyle[0] * max_X[2]:.1f} год"
        )

        print(
            f"Калорії: {avg_lifestyle[3] * max_X[5]:.0f} ккал"
        )

        print(
            f"Білок: {avg_lifestyle[4] * max_X[6]:.0f} г"
        )

        accumulated_seq = X_norm.tolist()

        soreness_predictions = []
        fatigue_predictions = []

        print("\n" + "=" * 60)
        print(f"ПРОГНОЗ НА {forecast_days} ДНІВ")
        print("=" * 60)

        with torch.no_grad():
            for day in range(1, forecast_days + 1):

                pred_norm = model(
                    torch.tensor(
                        [accumulated_seq],
                        dtype=torch.float32
                    )
                ).numpy()[0]

                soreness = np.clip(
                    pred_norm[0] * max_Y[0],
                    1.0,
                    5.0
                )

                fatigue = np.clip(
                    pred_norm[1] * max_Y[1],
                    1.0,
                    5.0
                )

                soreness_predictions.append(soreness)
                fatigue_predictions.append(fatigue)

                print(
                    f"День {day}: "
                    f"Soreness = {soreness:.2f} | "
                    f"Fatigue = {fatigue:.2f}"
                )

                next_day = np.zeros(9, dtype=np.float32)
                next_day[2:] = avg_lifestyle

                accumulated_seq.append(
                    next_day.tolist()
                )

        future_days = np.arange(
            1,
            forecast_days + 1
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            future_days,
            soreness_predictions,
            marker='o',
            linewidth=2,
            label="Muscle Soreness"
        )

        plt.plot(
            future_days,
            fatigue_predictions,
            marker='o',
            linewidth=2,
            label="CNS Fatigue"
        )

        plt.xlabel("Future Day")
        plt.ylabel("Predicted Value")

        plt.title(
            "Future Recovery Forecast"
        )

        plt.grid(True)
        plt.legend()
        plt.show()

    def evaluate_model(self):
        loaded = self.load_model()

        if not loaded:
            return

        model, max_X, max_Y = loaded

        parser = AdvancedSportsParser(
            self.new_data_path
        )

        X_norm = (
            parser.execute_pipeline()
            / max_X
        )

        Y_norm = (
            self.parse_targets()
            / max_Y
        )

        predictions = []

        with torch.no_grad():
            for i in range(1, len(X_norm)):

                pred = model(
                    torch.tensor(
                        [X_norm[:i]],
                        dtype=torch.float32
                    )
                ).numpy()[0]

                predictions.append(pred)

        predictions = np.array(predictions)

        real = Y_norm[1:]

        pred_denorm = predictions * max_Y
        real_denorm = real * max_Y

        mae_soreness = mean_absolute_error(
            real_denorm[:, 0],
            pred_denorm[:, 0]
        )

        rmse_soreness = np.sqrt(
            mean_squared_error(
                real_denorm[:, 0],
                pred_denorm[:, 0]
            )
        )

        r2_soreness = r2_score(
            real_denorm[:, 0],
            pred_denorm[:, 0]
        )

        mae_fatigue = mean_absolute_error(
            real_denorm[:, 1],
            pred_denorm[:, 1]
        )

        rmse_fatigue = np.sqrt(
            mean_squared_error(
                real_denorm[:, 1],
                pred_denorm[:, 1]
            )
        )

        r2_fatigue = r2_score(
            real_denorm[:, 1],
            pred_denorm[:, 1]
        )

        print("\n" + "=" * 60)
        print("METRICS: MUSCLE SORENESS")
        print("=" * 60)

        print(f"MAE  : {mae_soreness:.4f}")
        print(f"RMSE : {rmse_soreness:.4f}")
        print(f"R²   : {r2_soreness:.4f}")

        print("\n" + "=" * 60)
        print("METRICS: CNS FATIGUE")
        print("=" * 60)

        print(f"MAE  : {mae_fatigue:.4f}")
        print(f"RMSE : {rmse_fatigue:.4f}")
        print(f"R²   : {r2_fatigue:.4f}")

        days = np.arange(len(pred_denorm))

        graphs = [
            (
                real_denorm[:, 0],
                pred_denorm[:, 0],
                "Muscle Soreness"
            ),

            (
                real_denorm[:, 1],
                pred_denorm[:, 1],
                "CNS Fatigue"
            )
        ]

        for real_data, pred_data, title in graphs:
            plt.figure(figsize=(12, 6))

            plt.plot(
                days,
                real_data,
                marker='o',
                linewidth=2,
                label=f"Real {title}"
            )

            plt.plot(
                days,
                pred_data,
                marker='x',
                linewidth=2,
                linestyle='--',
                label=f"Predicted {title}"
            )

            plt.xlabel("Day")
            plt.ylabel(title)

            plt.title(
                f"Real vs Predicted {title}"
            )

            plt.grid(True)
            plt.legend()
            plt.show()

        soreness_error = np.abs(
            real_denorm[:, 0]
            - pred_denorm[:, 0]
        )

        fatigue_error = np.abs(
            real_denorm[:, 1]
            - pred_denorm[:, 1]
        )

        plt.figure(figsize=(12, 6))

        plt.plot(
            days,
            soreness_error,
            marker='o',
            linewidth=2,
            label="Soreness Error"
        )

        plt.plot(
            days,
            fatigue_error,
            marker='o',
            linewidth=2,
            label="Fatigue Error"
        )

        plt.xlabel("Day")
        plt.ylabel("Absolute Error")

        plt.title("Prediction Error")

        plt.grid(True)
        plt.legend()
        plt.show()

    def trigger_auto_retrain(self):
        print("\n" + "=" * 60)
        print("АВТОМАТИЧНЕ ПЕРЕНАВЧАННЯ")
        print("=" * 60)

        try:
            args = [
                input("seq_len (default 3): ") or "3",
                input("hidden_size (default 32): ") or "32",
                input("epochs (default 120): ") or "120",
                input("learning rate (default 0.005): ") or "0.005"
            ]

            result = subprocess.run(
                [sys.executable, "train.py", *args],
                capture_output=False,
                text=True
            )

            print(
                "\n[Успіх] Модель перенавчена."
                if result.returncode == 0
                else "\n[Помилка] train.py завершився з помилкою."
            )

        except Exception as e:
            print(f"[Критична помилка]: {e}")

    def start_main_loop(self):
        while True:

            print("\n" + "#" * 60)
            print("JORDAN RNN SPORT RECOVERY SYSTEM")
            print("#" * 60)

            menu = [
                "1. Інформація",
                "2. Статус моделі",
                "3. Прогноз",
                "4. Перенавчання",
                "5. Оцінка моделі",
                "0. Вихід"
            ]

            for item in menu:
                print(item)

            print("#" * 60)

            action = input("Оберіть дію: ")

            if action == '1':

                print("\nОзнаки:")

                for i, name in enumerate([
                    "Tonnage",
                    "RPE",
                    "Sleep",
                    "Steps",
                    "Water",
                    "Calories",
                    "Protein",
                    "Carbs",
                    "Fat"
                ], 1):
                    print(f"{i}. {name}")

            elif action == '2':

                print(
                    "\nМодель готова."
                    if (
                        os.path.exists(self.weights_path)
                        and os.path.exists(self.meta_path)
                    )
                    else "\nФайли моделі не знайдено."
                )

            elif action == '3':
                self.generate_autoregressive_forecast()

            elif action == '4':
                self.trigger_auto_retrain()

            elif action == '5':
                self.evaluate_model()

            elif action == '0':
                break


if __name__ == "__main__":
    IntelligentSystemApp().start_main_loop()