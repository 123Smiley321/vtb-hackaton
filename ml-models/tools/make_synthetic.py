"""
Синтетические данные со СТРУКТУРОЙ КОЛОНОК PaySim и Sparkov — только для проверки,
что ноутбуки запускаются от начала до конца. Цифры на синтетике ничего не значат.

python tools/make_synthetic.py out_dir
"""
import sys, os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
os.makedirs(f"{out}/paysim", exist_ok=True)
os.makedirs(f"{out}/sparkov", exist_ok=True)

# ------------------------------------------------------------ PaySim
n = 300_000
types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], n,
                   p=[0.34, 0.08, 0.35, 0.22, 0.01])
step = np.sort(rng.integers(1, 744, n))
amount = np.round(rng.lognormal(10.5, 1.2, n), 2)
orig = np.array([f"C{x}" for x in rng.integers(1e8, 9e8, n)])
dest_pool = np.array([f"C{x}" for x in rng.integers(1e8, 9e8, 40_000)])
dest = rng.choice(dest_pool, n)
dest[types == "PAYMENT"] = [f"M{x}" for x in rng.integers(1e8, 9e8, (types == "PAYMENT").sum())]
old_o = np.round(rng.lognormal(10, 2, n), 2)
new_o = np.maximum(old_o - amount, 0)
old_d = np.round(rng.lognormal(11, 2, n), 2)
new_d = old_d + amount
df = pd.DataFrame(dict(step=step, type=types, amount=amount, nameOrig=orig,
                       oldbalanceOrg=old_o, newbalanceOrig=new_o, nameDest=dest,
                       oldbalanceDest=old_d, newbalanceDest=new_d, isFraud=0, isFlaggedFraud=0))
# мошенничество: TRANSFER на новый счёт (опустошение) + CASH_OUT той же суммы в том же шаге
k = 400
fr_rows = []
for i in range(k):
    s = int(rng.integers(1, 744)); a = float(np.round(rng.lognormal(12.5, 1.0), 2))
    o = f"C{rng.integers(1e8, 9e8)}"; mule = f"C{rng.integers(1e8, 9e8)}"
    h = rng.random() < 0.6  # ночь
    fr_rows.append(dict(step=s, type="TRANSFER", amount=a, nameOrig=o, oldbalanceOrg=a, newbalanceOrig=0,
                        nameDest=mule, oldbalanceDest=0, newbalanceDest=0, isFraud=1, isFlaggedFraud=0))
    fr_rows.append(dict(step=s, type="CASH_OUT", amount=a, nameOrig=o, oldbalanceOrg=a, newbalanceOrig=0,
                        nameDest=rng.choice(dest_pool), oldbalanceDest=float(rng.lognormal(11, 2)),
                        newbalanceDest=0, isFraud=1, isFlaggedFraud=0))
df = pd.concat([df, pd.DataFrame(fr_rows)]).sort_values("step", kind="mergesort").reset_index(drop=True)
df.to_csv(f"{out}/paysim/PS_20174392719_1491204439457_log.csv", index=False)
print("paysim", df.shape, df.isFraud.mean())

# ------------------------------------------------------------ Sparkov
cats = ["gas_transport", "grocery_pos", "home", "shopping_pos", "kids_pets", "shopping_net",
        "entertainment", "food_dining", "personal_care", "health_fitness", "misc_pos",
        "misc_net", "grocery_net", "travel"]
n_cli = 1200
cc = rng.integers(10**15, 10**16 - 1, n_cli)
dob = pd.to_datetime("2019-01-01") - pd.to_timedelta(rng.integers(18 * 365, 85 * 365, n_cli), unit="D")
gender = rng.choice(["F", "M"], n_cli)
lat = rng.uniform(30, 45, n_cli); lon = rng.uniform(-120, -75, n_cli)
pop = rng.integers(500, 2_000_000, n_cli)
rows = []
start = pd.Timestamp("2019-01-01").value // 10**9
end = pd.Timestamp("2020-12-31").value // 10**9
for i in range(n_cli):
    m = int(rng.integers(80, 260))
    ts = np.sort(rng.integers(start, end, m))
    fav = rng.choice(cats, 4, replace=False)
    cat = np.where(rng.random(m) < 0.8, rng.choice(fav, m), rng.choice(cats, m))
    amt = np.round(rng.lognormal(3.6, 0.9, m), 2)
    fraud = np.zeros(m, int)
    age = (2020 - dob[i].year)
    if rng.random() < (0.10 if age >= 60 else 0.05):
        j = int(rng.integers(0, m - 6))
        burst = np.arange(j, j + int(rng.integers(3, 7)))
        fraud[burst] = 1
        base = ts[j]
        ts[burst] = base + np.sort(rng.integers(0, 3 * 3600, len(burst)))
        ts[burst] = ts[burst] - ts[burst] % 86400 + rng.integers(22 * 3600, 26 * 3600, len(burst)) % 86400
        cat[burst] = rng.choice(["shopping_net", "grocery_pos", "misc_net"], len(burst))
        amt[burst] = np.round(rng.lognormal(5.7, 0.6, len(burst)), 2)
    for t, c, a, f in zip(ts, cat, amt, fraud):
        rows.append((pd.Timestamp(int(t), unit="s").strftime("%Y-%m-%d %H:%M:%S"), cc[i],
                     f"fraud_Merchant{rng.integers(1, 700)}", c, a, "Name", "Surname", gender[i],
                     "1 Main St", "City", "NY", 10001, lat[i], lon[i], pop[i], "Engineer",
                     dob[i].strftime("%Y-%m-%d"), f"{rng.integers(1e15, 9e15):x}", int(t),
                     lat[i] + rng.normal(0, 0.8), lon[i] + rng.normal(0, 0.8), f))
cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender",
        "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num",
        "unix_time", "merch_lat", "merch_long", "is_fraud"]
sp = pd.DataFrame(rows, columns=cols).sort_values("unix_time").reset_index(drop=True)
cut = pd.Timestamp("2020-06-21")
d = pd.to_datetime(sp.trans_date_trans_time)
sp[d < cut].reset_index(drop=True).to_csv(f"{out}/sparkov/fraudTrain.csv")
sp[d >= cut].reset_index(drop=True).to_csv(f"{out}/sparkov/fraudTest.csv")
print("sparkov", sp.shape, sp.is_fraud.mean())
