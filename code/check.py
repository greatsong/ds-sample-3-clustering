# check.py · 앱(main.py)과 같은 계산을 터미널에서 돌려 README에 적은 수치를 모두 출력한다
# 실행: 저장소 맨 위 폴더에서 python3 code/check.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

데이터경로 = "data/kbo_hitters_2026_full.csv"
기준타석 = 100
비율열 = ["홈런율", "도루율", "볼넷율", "삼진율", "2루타율"]
원래열 = ["홈런", "도루", "볼넷", "삼진", "2루타"]


def 비율_만들기(df):
    out = df.copy()
    for 비율, 원래 in zip(비율열, 원래열):
        out[비율] = out[원래] / out["타석"]
    return out


def 묶기(X, k):
    모델 = KMeans(n_clusters=k, n_init=10, random_state=0)
    번호 = 모델.fit_predict(X)
    return 모델, 번호


def 유형_이름(평균표):
    # 군집 번호는 실행마다 바뀔 수 있으므로 평균값으로 유형을 정한다
    이름 = {}
    이름[평균표["홈런율"].idxmax()] = "거포형"
    남은 = 평균표.drop(index=list(이름))
    이름[남은["도루율"].idxmax()] = "준족형"
    남은 = 남은.drop(index=list(이름)[-1])
    for i in 남은.index:
        이름[i] = "교타형"
    return 이름


df = pd.read_csv(데이터경로, encoding="utf-8")
print("== 1. 데이터")
print("전체 행 수:", len(df), "· 열 수:", df.shape[1])
이름수 = df["선수명"].value_counts()
print("이름이 같은 선수가 있는 이름 수:", int((이름수 > 1).sum()), "· 팀 수:", df["팀"].nunique())
print("결측(빈칸) 수:", {c: int(v) for c, v in df.isna().sum().items() if v > 0})
print("타석 최솟값·중앙값·최댓값:", int(df["타석"].min()), int(df["타석"].median()), int(df["타석"].max()))
print("타석 50 미만 선수:", int((df["타석"] < 50).sum()))
print("타석 10 미만 선수:", int((df["타석"] < 10).sum()))
print("타석 0 선수(비율 계산 불가):", int((df["타석"] == 0).sum()))
비전체 = 비율_만들기(df)
print(f"전체 {len(df)}명 기준 홈런율 최댓값:", round(비전체["홈런율"].max(), 3),
      "(", 비전체.loc[비전체["홈런율"].idxmax(), "선수명"], ", 타석",
      int(비전체.loc[비전체["홈런율"].idxmax(), "타석"]), ")")
print("규정 타석(446) 이상 선수:", int((df["타석"] >= 446).sum()))
print("도루 20개 이상 선수:", int((df["도루"] >= 20).sum()),
      "명, 그 가운데 규정 타석 미만:", int(((df["도루"] >= 20) & (df["타석"] < 446)).sum()))
print("타석 100~445 선수(규정 타석 기준이면 빠지는 인원):", int(((df["타석"] >= 100) & (df["타석"] < 446)).sum()))
print("타석 100 이상 선수의 홈런율 최댓값:", round(비전체.loc[비전체["타석"] >= 100, "홈런율"].max(), 3))
print("타석 0 선수 가운데 타율이 빈칸인 선수:", int(((df["타석"] == 0) & df["타율"].isna()).sum()))
print("홈런율 0.1 이상인 선수:", int((비전체["홈런율"] >= 0.1).sum()),
      "명, 그 가운데 타석 100 이상:", int(((비전체["홈런율"] >= 0.1) & (비전체["타석"] >= 100)).sum()))

print("\n== 2. 타석 기준별 인원과 k=3 결과 (표준화 후)")
for 기준 in [50, 100, 200]:
    d = 비율_만들기(df[df["타석"] >= 기준])
    X = StandardScaler().fit_transform(d[비율열])
    _, 번호 = 묶기(X, 3)
    크기 = sorted(pd.Series(번호).value_counts().tolist(), reverse=True)
    평균 = d.assign(군집=번호).groupby("군집")[비율열].mean()
    이름 = 유형_이름(평균)
    유형크기 = {이름[i]: int((번호 == i).sum()) for i in 평균.index}
    print(f"타석 {기준} 이상: {len(d)}명 · 실루엣 {silhouette_score(X, 번호):.3f} · 크기(큰 순) {크기} · 유형별 {유형크기}")
    for i in 평균.index:
        print(f"    {이름[i]} 평균: " + " · ".join(f"{c} {평균.loc[i, c]:.3f}" for c in 비율열))

d = 비율_만들기(df[df["타석"] >= 기준타석]).reset_index(drop=True)
print("\n== 3. 분석 대상 (타석", 기준타석, "이상):", len(d), "명")
print("분석 대상의 결측:", {c: int(v) for c, v in d.isna().sum().items() if v > 0})
print("비율 변수 요약(평균·표준편차·최솟값·최댓값):")
print(d[비율열].agg(["mean", "std", "min", "max"]).round(3).to_string())

Xs = StandardScaler().fit_transform(d[비율열])
Xr = d[비율열].to_numpy()

print("\n== 4. k 후보 표 (표준화 후)")
앞값 = None
for k in range(2, 7):
    모델, 번호 = 묶기(Xs, k)
    줄어든값 = "" if 앞값 is None else f" · 앞 k에서 줄어든 값 {앞값 - 모델.inertia_:.1f}"
    print(f"k={k}: 실루엣 {silhouette_score(Xs, 번호):.3f} · 군집 내 제곱합 {모델.inertia_:.1f}{줄어든값}")
    앞값 = 모델.inertia_

print("\n== 4-1. k=4일 때 군집 평균 (표준화 후, 참고)")
_, 번호4 = 묶기(Xs, 4)
평균4 = d.assign(군집=번호4).groupby("군집")[비율열].mean()
print(평균4.assign(크기=pd.Series(번호4).value_counts()).round(3).to_string())

print("\n== 5. 표준화 전후 비교 (k=3)")
_, 번호r = 묶기(Xr, 3)
평균r = d.assign(군집=번호r).groupby("군집")[비율열].mean()
print(f"표준화 없이: 실루엣 {silhouette_score(Xr, 번호r):.3f}")
print("표준화 없이 군집 크기·평균:")
print(평균r.assign(크기=pd.Series(번호r).value_counts()).round(3).to_string())
print("비율 변수의 표준편차(척도):", d[비율열].std().round(3).to_dict())

print("\n== 6. 최종 결과 (표준화 후 k=3)")
_, 번호 = 묶기(Xs, 3)
print(f"실루엣 {silhouette_score(Xs, 번호):.3f}")
d["군집"] = 번호
평균 = d.groupby("군집")[비율열].mean()
이름 = 유형_이름(평균)
for i in [k for k, v in sorted(이름.items(), key=lambda kv: ["준족형", "거포형", "교타형"].index(kv[1]))]:
    g = d[d["군집"] == i].sort_values("타석", ascending=False)
    m = 평균.loc[i]
    print(f"{이름[i]} (군집 번호 {i}): {len(g)}명 · 평균 "
          + " · ".join(f"{c} {m[c]:.3f}" for c in 비율열))
    print("  타석 많은 선수 5명:", "·".join(g["선수명"].head(5)))
print("전체 평균:", " · ".join(f"{c} {d[c].mean():.3f}" for c in 비율열))
for i in 평균.index:
    print(f"{이름[i]} 평균 ÷ 전체 평균:", " · ".join(f"{c} {평균.loc[i, c] / d[c].mean():.2f}배" for c in 비율열))

print("\n== 6-1. k=3 유형(행)이 k=4 묶음(열)으로 어떻게 옮겨 가는가")
print(pd.crosstab(d["군집"].map(이름), 번호4, rownames=["k=3 유형"], colnames=["k=4 묶음 번호"]).to_string())

print("\n== 7. (거절한 제안 확인) 타율·OPS를 변수에 더했을 때 k=3")
d7 = d.copy()
d7["타율"] = pd.to_numeric(d7["타율"], errors="coerce")
d7["OPS"] = pd.to_numeric(d7["OPS"], errors="coerce")
열7 = 비율열 + ["타율", "OPS"]
X7 = StandardScaler().fit_transform(d7[열7])
_, 번호7 = 묶기(X7, 3)
print(f"실루엣 {silhouette_score(X7, 번호7):.3f}")
print(d7.assign(군집=번호7).groupby("군집")[열7].mean().assign(크기=pd.Series(번호7).value_counts()).round(3).to_string())
print("우리 결과(다섯 비율)의 유형별 평균 OPS:",
      {이름[i]: round(float(d7.loc[d7["군집"] == i, "OPS"].mean()), 3) for i in 평균.index})
