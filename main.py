import numpy as np
import itertools

import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# 入力
num_employees = 20

days = pd.date_range("2021-08-01", "2021-08-31")  # 日付一覧
employees = pd.Series([f"employees{i}" for i in range(num_employees)])  # 従業員一覧
shifts = pd.Series(["早番", "遅番", "夜勤", "明け", "休暇"])  # シフト一覧

# 形状が (日付, 3) の DataFrame。(i, k) 成分は日付 i のシフト k の必要人数を表す。
need = pd.DataFrame([[2, 2, 2]], index=days, columns=["早番", "遅番", "夜勤"])
print(need)

# ["",]

# 許可するパターン
# 早番 or 遅番 → 夜勤 → 明け → 休暇のパターンのみ OK
ok_patterns = [
    ("早番", "夜勤"),
    ("遅番", "夜勤"),
    ("夜勤", "明け"),
    ("明け", "休暇"),
    ("休暇", "早番"),
    ("休暇", "遅番"),
]

# モデルを作成する。
model = LpProblem(sense=LpMinimize)

# 変数を作成する。
X = [
    [[LpVariable(f"{d}_{e}_{s}", cat="Binary") for s in shifts] for e in employees]
    for d in days
]

# 目的変数及び制約を作成する。
###########################################
objective = 0
for i in range(days.size):
    for k in range(need.columns.size):
        # 日付 days[i] のシフト k に実際に働いた人数
        actual = lpSum(X[i][j][k] for j in range(employees.size))
        # 日付 days[i] のシフト k に必要だった人数
        expected = need.iloc[i, k]
        # 余剰人数を目的関数に足す。
        objective += actual - expected
        # (制約1) シフト k の必要人数は満たす制約
        model += actual >= expected
model += objective  # 余剰人数を最小化する。

# (制約2) 一人が同じ日に複数のシフトはできない。
for i in range(days.size):
    for j in range(employees.size):
        model += lpSum(X[i][j]) == 1

# OK のパターンをインデックスに変換する。
name_to_idx = {v: i for i, v in shifts.items()}
ok = [tuple([name_to_idx[name] for name in p]) for p in ok_patterns]

# NG のパターンを作成する。
ng = [p for p in itertools.product(name_to_idx.values(), repeat=2) if p not in ok]

# (制約3) NG のパターンが出ないように制約に追加する。
for j in range(employees.size):
    for pattern in ng:
        for i in range(days.size - len(pattern) + 1):
            model += (
                lpSum(X[i + k][j][pattern[k]] for k in range(len(pattern)))
                <= len(pattern) - 1
            )


# モデルを解く。
ret = model.solve()
assert ret == 1, f"No Solution found, status: {ret}"

schedule = pd.DataFrame(index=days, columns=employees)
for i in range(days.size):
    for j in range(employees.size):
        # one-hot 表現からインデックスを取得する。 例: [0, 0, 1, 0] -> 2
        binary_repr = [X[i][j][k].value() for k in range(len(shifts))]
        assert np.sum(binary_repr) == 1
        k = np.argmax(binary_repr)
        schedule.iloc[i, j] = shifts.iloc[k]

schedule.to_csv("schedule.csv", encoding="utf_8_sig")
