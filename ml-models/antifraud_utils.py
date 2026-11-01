"""
Общие функции для ноутбуков команды (Хакатон ВТБ, задача №6).

Метрики считаются строго по формулам кейса (с. 10):
  точность выявления          = TP / (TP + FP)
  доля ложных сигналов        = FP / (TP + FP)
  FPR (ложные среди законных) = FP / (FP + TN)
  полнота обнаружения         = TP / (TP + FN)
Дополнительно: полнота по сумме (какая доля похищаемых денег остановлена).

Этот же код вставлен первой ячейкой в каждый ноутбук, чтобы ноутбуки
запускались на Kaggle без дополнительных файлов.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------- сплит
def time_split(df, time_col, fracs=(0.6, 0.2, 0.2)):
    """Сплит по времени: train — самые ранние операции, test — самые поздние.
    Случайный сплит запрещён: он подмешивает «будущее» в обучение и завышает метрики."""
    t = df[time_col]
    q1 = t.quantile(fracs[0])
    q2 = t.quantile(fracs[0] + fracs[1])
    train = df[t <= q1]
    valid = df[(t > q1) & (t <= q2)]
    test = df[t > q2]
    return train, valid, test


# ---------------------------------------------------------------- история клиента без утечки
def past_stats(df, group_cols, time_col, value_col, lag=0, window=None):
    """Статистики по ПРОШЛЫМ операциям группы (клиент, клиент+категория, получатель...).

    Для строки с временем t учитываются операции той же группы со временем < t − lag
    (и, если задано window, ≥ t − lag − window). Текущая операция и будущее не учитываются.
    lag=0      — онлайн-признак (поток операций доступен сразу)
    lag=86400  — «ночной профиль»: данные приходят с задержкой до суток (условие кейса)
    Возвращает DataFrame (cnt, sum, mean, std) в порядке индекса df. Векторизовано.
    """
    g = df.groupby(group_cols, sort=False).ngroup().to_numpy().astype(np.int64)
    t = df[time_col].to_numpy().astype(np.int64)
    v = df[value_col].to_numpy().astype(float)
    t = t - t.min()
    big = np.int64(t.max() + lag + (window or 0) + 2)
    key = g * big + t
    order = np.argsort(key, kind="mergesort")
    ks, vs = key[order], v[order]
    cs = np.r_[0.0, np.cumsum(vs)]
    cs2 = np.r_[0.0, np.cumsum(vs ** 2)]
    gs = g[order] * big                                   # начало «окна» группы на оси ключей
    hi = np.searchsorted(ks, ks - lag, side="left")       # строки со временем < t − lag
    lo = np.searchsorted(ks, gs, side="left")             # первая строка группы
    if window is not None:
        lo = np.maximum(lo, np.searchsorted(ks, ks - lag - window, side="left"))
    hi = np.maximum(hi, lo)
    cnt = (hi - lo).astype(float)
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / cnt, np.nan)
        var = np.where(cnt > 1, (s2 - s ** 2 / cnt) / (cnt - 1), np.nan)
    out = np.empty((len(df), 4))
    out[order] = np.c_[cnt, s, mean, np.sqrt(np.clip(var, 0, None))]
    return pd.DataFrame(out, index=df.index, columns=["cnt", "sum", "mean", "std"])


# ---------------------------------------------------------------- метрики
def metrics(y, score, threshold, amount=None):
    """Все метрики кейса при заданном пороге. y — 1 мошенничество, 0 законная."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    res = {
        "n": len(y),
        "n_fraud": int(y.sum()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "false_signal_share": fp / (tp + fp) if tp + fp else np.nan,
        "alert_rate": (tp + fp) / len(y),
        "roc_auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
        "pr_auc": average_precision_score(y, score) if y.sum() > 0 else np.nan,
        "threshold": threshold,
    }
    if amount is not None:
        amount = np.asarray(amount, dtype=float)
        fraud_amt = amount[y == 1].sum()
        caught = amount[(y == 1) & pred].sum()
        res["fraud_amount"] = fraud_amt
        res["caught_amount"] = caught
        res["missed_amount"] = fraud_amt - caught
        res["amount_recall"] = caught / fraud_amt if fraud_amt else np.nan
    return res


def metrics_by_group(df, group_col, y_col, score_col, threshold, amount_col=None):
    """Метрики по сегментам. threshold — число или dict {сегмент: порог}."""
    rows = []
    for g, part in df.groupby(group_col, observed=True):
        thr = threshold[g] if isinstance(threshold, dict) else threshold
        m = metrics(part[y_col], part[score_col], thr,
                    part[amount_col] if amount_col else None)
        m[group_col] = g
        rows.append(m)
    return pd.DataFrame(rows).set_index(group_col)


def threshold_for_fpr(y, score, target_fpr):
    """Порог, при котором FPR на этих данных не превышает target_fpr."""
    y = np.asarray(y).astype(int)
    legit = np.sort(np.asarray(score, dtype=float)[y == 0])[::-1]
    k = int(np.floor(target_fpr * len(legit)))
    # флагуем только то, что строго выше k-го законного скора (корректно и при одинаковых скорах)
    return float(np.nextafter(legit[min(k, len(legit) - 1)], np.inf))


# ---------------------------------------------------------------- цена ошибки
def cost_curve(y, score, amount, cost_fp, loss_share=1.0):
    """Полная цена ошибок для КАЖДОГО возможного порога (точный перебор).

    Цена пропуска (FN)  = сумма хищения × loss_share
    Цена ложного (FP)   = cost_fp (обработка + ожидаемый отказ от операции)
    Возвращает DataFrame: threshold, TP, FP, missed_amount, cost.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=float)
    a = np.asarray(amount, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    s, y, a = s[order], y[order], a[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    caught = np.cumsum(a * y)
    total_fraud_amt = (a * y).sum()
    # позиции, где score меняется: порог = s[i], флагуем всё с индексом <= i
    last = np.r_[s[1:] != s[:-1], True]
    idx = np.where(last)[0]
    out = pd.DataFrame({
        "threshold": s[idx],
        "TP": tp[idx], "FP": fp[idx],
        "missed_amount": total_fraud_amt - caught[idx],
    })
    # вариант «не флагуем ничего»
    out = pd.concat([pd.DataFrame({"threshold": [np.inf], "TP": [0], "FP": [0],
                                   "missed_amount": [total_fraud_amt]}), out],
                    ignore_index=True)
    out["cost"] = out["missed_amount"] * loss_share + out["FP"] * cost_fp
    return out


def best_threshold(y, score, amount, cost_fp, loss_share=1.0):
    cc = cost_curve(y, score, amount, cost_fp, loss_share)
    return float(cc.loc[cc["cost"].idxmin(), "threshold"])


def best_threshold_by_group(df, group_col, y_col, score_col, amount_col,
                            cost_fp, loss_share=1.0):
    """Свой порог для каждого сегмента. cost_fp может быть dict по сегментам."""
    thr = {}
    for g, part in df.groupby(group_col, observed=True):
        c = cost_fp[g] if isinstance(cost_fp, dict) else cost_fp
        thr[g] = best_threshold(part[y_col], part[score_col], part[amount_col], c, loss_share)
    return thr


def total_cost(y, score, amount, threshold, cost_fp, loss_share=1.0):
    m = metrics(y, score, threshold, amount)
    return m["missed_amount"] * loss_share + m["FP"] * cost_fp


# ---------------------------------------------------------------- многоуровневая реакция
# Уровень реакции (tier) — dict:
#   name           название
#   stop_fraud     доля мошеннических операций, которые этот уровень останавливает
#   legit_blocked  доля законных операций, которые на этом уровне не состоятся
#                  (приостановка или отказ клиента) — «ошибочно ограниченные» по кейсу
#   legit_churn    доля законных клиентов, которые после этого снижают активность
#   legit_kc       доля законных клиентов, которые обращаются в КЦ
#   op_cost        прямые затраты банка на одну операцию этого уровня, руб. (SMS, звонок)
# Экономика (econ) — dict:
#   loss_share     доля суммы хищения, которая считается потерей (цена пропуска)
#   kc_cost        стоимость одного обращения в КЦ, руб.
#   churn_value    потеря дохода от одного клиента, снизившего активность, руб.

_TIER_KEYS = ["stop_fraud", "legit_blocked", "legit_churn", "legit_kc", "op_cost"]


def _tier_arrays(tiers):
    return {k: np.array([t[k] for t in tiers], dtype=float) for k in _TIER_KEYS}


def legit_cost(tier, econ):
    """Цена одного ложного сигнала на этом уровне для законного клиента и банка, руб."""
    return tier["op_cost"] + tier["legit_kc"] * econ["kc_cost"] + tier["legit_churn"] * econ["churn_value"]


def choose_tier(p, amount, tiers, econ, allowed=None):
    """Для каждой операции — уровень реакции с максимальной ожидаемой выгодой.

    Выгода = p·сумма·loss_share·stop_fraud                              (сохранённые деньги)
           − (1−p)·(legit_kc·kc_cost + legit_churn·churn_value)         (цена ложного сигнала)
           − op_cost                                                    (работа банка)
    p — откалиброванная вероятность мошенничества.
    allowed — необязательная маска (n_ops × n_tiers): какие уровни разрешены для операции.
    """
    T = _tier_arrays(tiers)
    p = np.asarray(p, dtype=float)[:, None]
    a = np.asarray(amount, dtype=float)[:, None]
    saved = p * a * econ["loss_share"] * T["stop_fraud"][None, :]
    friction = (1 - p) * (T["legit_kc"] * econ["kc_cost"] + T["legit_churn"] * econ["churn_value"])[None, :]
    benefit = saved - friction - T["op_cost"][None, :]
    if allowed is not None:
        benefit = np.where(np.asarray(allowed, dtype=bool), benefit, -np.inf)
    return benefit.argmax(axis=1)


def simulate_reaction(y, amount, tier_idx, tiers, econ, client_id=None):
    """Ожидаемые последствия выбранных уровней на РАЗМЕЧЕННЫХ данных.

    Ложные сигналы и ошибочно ограниченные операции считаются отдельно (кейс, с. 10):
      false_signals       — законные операции, получившие любую реакцию выше «пропустить»
      wrongly_restricted  — законные операции, которые в итоге не состоялись
    """
    T = _tier_arrays(tiers)
    y = np.asarray(y).astype(int)
    a = np.asarray(amount, dtype=float)
    ti = np.asarray(tier_idx)
    stop = T["stop_fraud"][ti]
    blocked = T["legit_blocked"][ti]
    fr, lg = y == 1, y == 0
    res = {
        "ops": len(y),
        "fraud_ops": int(fr.sum()),
        "fraud_amount": a[fr].sum(),
        "prevented_amount": (a * stop)[fr].sum(),
        "stolen_amount": (a * (1 - stop))[fr].sum(),
        "fraud_stopped_exp": stop[fr].sum(),          # ожидаемое число остановленных хищений (TP)
        "fraud_flagged": int((ti[fr] > 0).sum()),     # хищения, получившие любую реакцию
        "false_signals": int((ti[lg] > 0).sum()),
        "false_signal_rate": float((ti[lg] > 0).mean()),
        "wrongly_restricted_exp": blocked[lg].sum(),
        "wrongly_restricted_rate": float(blocked[lg].mean()),
        "kc_inbound_exp": T["legit_kc"][ti][lg].sum(),
        "ops_cost": T["op_cost"][ti].sum(),
        "churn_clients_exp": T["legit_churn"][ti][lg].sum(),
    }
    res["kc_cost"] = res["kc_inbound_exp"] * econ["kc_cost"]
    res["churn_cost"] = res["churn_clients_exp"] * econ["churn_value"]
    res["amount_recall"] = res["prevented_amount"] / res["fraud_amount"] if res["fraud_amount"] else np.nan
    if client_id is not None:
        cid = np.asarray(client_id)[lg]
        # вероятность, что у клиента за период было хотя бы одно ошибочное ограничение
        log_ok = pd.Series(np.log1p(-np.clip(blocked[lg], 0, 0.999999))).groupby(cid).sum()
        res["clients"] = int(len(np.unique(np.asarray(client_id))))
        res["clients_wrongly_restricted_exp"] = float((1 - np.exp(log_ok)).sum())
    return res


def tier_table(y, tier_idx, tiers):
    """Сколько законных и мошеннических операций попало на каждый уровень."""
    y = np.asarray(y).astype(int)
    df = pd.DataFrame({"tier": [tiers[i]["name"] for i in tier_idx], "y": y})
    t = df.groupby("tier")["y"].agg(ops="size", fraud="sum")
    t["legit"] = t["ops"] - t["fraud"]
    return t.reindex([x["name"] for x in tiers]).fillna(0).astype(int)


def fmt_rub(x):
    return f"{x/1e6:,.2f} млн руб.".replace(",", " ")
