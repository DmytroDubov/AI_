import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class JordanRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(JordanRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Шари перетворення згідно з топологією Джордана
        self.w_ih = nn.Linear(input_size, hidden_size)  # Вхід -> Прихований
        self.w_yh = nn.Linear(output_size, hidden_size, bias=False)  # Вихід(t-1) -> Прихований
        self.w_ho = nn.Linear(hidden_size, output_size)  # Прихований -> Вихід

        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Size]
        batch_size, seq_len, _ = x.size()

        # Ініціалізація початкового виходу y_(t-1) нулями
        y_prev = torch.zeros(batch_size, self.output_size, device=x.device)

        # Прохід по часовій послідовності (Рекурентний крок Джордана)
        for t in range(seq_len):
            x_t = x[:, t, :]  # Поточний день

            # h_t = tanh(W_ih * x_t + W_yh * y_(t-1) + b)
            h_t = self.tanh(self.w_ih(x_t) + self.w_yh(y_prev))

            # y_t = W_ho * h_t + b
            y_curr = self.w_ho(h_t)

            y_prev = y_curr

        return y_curr


# --- ШАР ПАРСИНГУ ТА ПІДГТОВКИ ДАНИХ ---
def prepare_training_data(json_path, seq_len=3):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Отримуємо словник і сортуємо ключі-дати хронологічно
    history_map = data.get("history", {})
    sorted_dates = sorted(history_map.keys())

    print(f"[Успіх] Зчитано {len(sorted_dates)} днів історії з календарної сітки.")
    X_raw, Y_raw = [], []

    for date in sorted_dates:
        day_obj = history_map[date]  # Витягуємо сам об'єкт дня за його датою
        bio = day_obj.get("biometrics", {})
        workout = day_obj.get("workout", {}) if day_obj.get("workout") else {}
        exercises = workout.get("exercises", [])

        total_tonnage = sum(s.get("weight", 0.0) * s.get("reps", 0) for ex in exercises for s in ex.get("sets", []))
        rpe_list = [s.get("rpe", 0.0) for ex in exercises for s in ex.get("sets", [])]
        max_rpe = max(rpe_list) if rpe_list else 0.0

        features = [
            total_tonnage, max_rpe, bio.get("sleepHours", 7.0), bio.get("steps", 5000),
            bio.get("waterMl", 2500), bio.get("caloriesConsumed", 2500.0),
            bio.get("proteinGrams", 150.0), bio.get("carbsGrams", 300.0), bio.get("fatGrams", 80.0)
        ]
        targets = [bio.get("muscleSoreness", 1.0), bio.get("cnsFatigue", 1.0)]
        X_raw.append(features)
        Y_raw.append(targets)

    X_np, Y_np = np.array(X_raw, dtype=np.float32), np.array(Y_raw, dtype=np.float32)
    max_X = np.max(X_np, axis=0);
    max_X[max_X == 0] = 1.0
    max_Y = np.max(Y_np, axis=0);
    max_Y[max_Y == 0] = 1.0

    X_norm, Y_norm = X_np / max_X, Y_np / max_Y
    X_seq, Y_seq = [], []
    for i in range(len(X_norm) - seq_len):
        X_seq.append(X_norm[i: i + seq_len])
        Y_seq.append(Y_norm[i + seq_len])

    return torch.tensor(X_seq), torch.tensor(Y_seq), max_X, max_Y


def main():
    print("\n" + "=" * 50)
    print("  ЕТАП НАВЧАННЯ МОДЕЛІ (JORDAN RNN PIPELINE) ©")
    print("  Розробник: Дубов Д.П. |  Група: ПП-33")
    print("=" * 50)

    json_path = "30_days_test.json"

    seq_len = 3
    hidden_size = 32
    epochs = 120
    lr = 0.005

    if len(sys.argv) > 1:
        try:
            if len(sys.argv) > 1: seq_len = int(sys.argv[1])
            if len(sys.argv) > 2: hidden_size = int(sys.argv[2])
            if len(sys.argv) > 3: epochs = int(sys.argv[3])
            if len(sys.argv) > 4: lr = float(sys.argv[4])
            print("[Конфігурація] Параметри успішно зчитано з аргументів CLI.")
        except ValueError:
            print("[Попередження] Некоректні аргументи CLI. Дефолтні значення.")
    else:
        print("[Конфігурація] Використовуються дефолтні параметри алгоритму.")

    print(f" -> Вікно пам'яті (seq_len): {seq_len}")
    print(f" -> Розмір прихованого шару (hidden_size): {hidden_size}")
    print(f" -> Кількість епох (epochs): {epochs}")
    print(f" -> Швидкість навчання (lr): {lr}\n")

    X_train, Y_train, max_X, max_Y = prepare_training_data(json_path, seq_len)

    # Ініціалізація кастомної моделі Джордана
    model = JordanRNN(input_size=9, hidden_size=hidden_size, output_size=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("[Оптимізація] Старт обчислень...")
    print("-" * 60)
    print(f"{'Епоха':<15}{'MSE Loss (Похибка)':<25}{'MAE'}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - Y_train)).item()
            print(f"[{epoch + 1:>3}/{epochs:<3}]       {loss.item():<25.6f}{mae:.6f}")

    print("-" * 60)

    weights_path = "jordan_model_weights.pth"
    meta_path = "model_metadata.json"

    torch.save(model.state_dict(), weights_path)
    metadata = {
        "seq_len": seq_len, "hidden_size": hidden_size,
        "max_X": max_X.tolist(), "max_Y": max_Y.tolist()
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)

    print(f"[Успіх] Ваги мережі Джордана успішно експортовано у файл: {weights_path}")
    print(f"[Успіх] Метадані нормалізації збережено у файл: {meta_path}\n")


if __name__ == "__main__":
    main()