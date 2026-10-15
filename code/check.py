# check.py · 앱(main.py)과 같은 계산을 터미널에서 돌려 README에 적은 수치를 모두 출력한다
# 실행: 저장소 맨 위 폴더에서 python3 code/check.py
import pandas as pd
from itertools import permutations

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
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

print("\n== 8. 후보 모델 비교 (같은 155명 · 같은 표준화 다섯 비율 · 묶음 3개)")
# 비교 기준: 실루엣, 묶음 크기, K-평균 세 유형과 겹치는 비율
# 겹치는 비율: 후보의 묶음 번호를 K-평균 유형에 하나씩 짝지어 같은 유형에 든 선수가 가장 많아지는 짝을 찾고, 그 인원 ÷ 155
DBSCAN_최소이웃 = 10   # 변수 5개 × 2
DBSCAN_반경 = 1.5      # 각 선수의 10번째 이웃까지 거리의 중앙값(아래 출력)을 반올림
거리, _ = NearestNeighbors(n_neighbors=DBSCAN_최소이웃).fit(Xs).kneighbors(Xs)
# 이웃 거리는 DBSCAN의 최소 이웃 계산과 같게 자기 자신을 첫째 이웃으로 센다
print(f"DBSCAN 반경 근거: {DBSCAN_최소이웃}번째 이웃까지 거리 중앙값 {pd.Series(거리[:, -1]).median():.2f}")
print("(AI가 처음 정한 반경 0.5일 때 잡음 수:", int((DBSCAN(eps=0.5, min_samples=DBSCAN_최소이웃).fit_predict(Xs) == -1).sum()), "명)")
후보 = {
    "K-평균(k=3)": KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(Xs),
    "계층적 군집(응집, ward, 3묶음)": AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xs),
    f"DBSCAN(반경 {DBSCAN_반경}, 최소 이웃 {DBSCAN_최소이웃})": DBSCAN(eps=DBSCAN_반경, min_samples=DBSCAN_최소이웃).fit_predict(Xs),
    "가우시안 혼합(3개)": GaussianMixture(n_components=3, random_state=0).fit_predict(Xs),
}
유형 = d["군집"].map(이름)
유형목록 = ["준족형", "거포형", "교타형"]
for 모델이름, 번호8 in 후보.items():
    번호8 = pd.Series(번호8)
    잡음 = int((번호8 == -1).sum())
    묶음 = sorted(번호8[번호8 != -1].unique())
    크기 = sorted(번호8[번호8 != -1].value_counts().tolist(), reverse=True)
    if len(묶음) >= 2:
        남김 = (번호8 != -1).to_numpy()
        실루엣 = f"{silhouette_score(Xs[남김], 번호8[남김]):.3f}"
    else:
        실루엣 = "계산 불가(묶음 1개)"
    표8 = pd.crosstab(유형, 번호8).reindex(index=유형목록, fill_value=0)
    if len(묶음) == 3:
        최대 = max(sum(int(표8.iloc[r][c]) for r, c in enumerate(짝)) for 짝 in permutations(묶음))
        겹침 = f"{최대}/{len(d)} = {최대 / len(d):.2f}"
    else:
        겹침 = f"비교 불가(묶음 {len(묶음)}개)"
    print(f"{모델이름}: 실루엣 {실루엣} · 묶음 크기(큰 순) {크기} · 잡음 {잡음}명 · K-평균 유형과 겹치는 비율 {겹침}")
    print("   " + 표8.to_string().replace("\n", "\n   "))
    평균8 = d.assign(묶음=번호8).groupby("묶음")[비율열].mean()
    평균8["인원"] = 번호8.value_counts()
    print("   묶음별 평균(-1은 잡음):")
    print("   " + 평균8.round(3).to_string().replace("\n", "\n   "))

print("\n== 9. 인사이트 (우리 팀이 찾아낸 것)")
d9 = d.copy()
d9["유형"] = d9["군집"].map(이름)
d9["OPS"] = pd.to_numeric(d9["OPS"], errors="coerce")
리그삼진율 = d9["삼진율"].mean()

print("-- 9-1. 홈런과 삼진")
print(f"홈런율-삼진율 상관계수(155명): {d9['홈런율'].corr(d9['삼진율']):.3f}")
상위, 하위 = d9["홈런율"].quantile(0.75), d9["홈런율"].quantile(0.25)
print(f"홈런율 상위 25% 선수의 평균 삼진율: {d9.loc[d9['홈런율'] >= 상위, '삼진율'].mean():.3f}"
      f" · 하위 25%: {d9.loc[d9['홈런율'] <= 하위, '삼진율'].mean():.3f}")
print(f"155명 평균 삼진율: {리그삼진율:.3f}")
홈런순 = d9.sort_values("홈런", ascending=False).head(5)
for _, r in 홈런순.iterrows():
    print(f"  홈런 {int(r['홈런'])}개 {r['선수명']}({r['팀']}): 삼진율 {r['삼진율']:.3f} · OPS {r['OPS']:.3f} · {r['유형']}")
거포 = d9[d9["유형"] == "거포형"]
print(f"거포형 안에서 삼진율-OPS 상관계수: {거포['삼진율'].corr(거포['OPS']):.3f}")

print("-- 9-2. 준족형의 도루")
도루순 = d9.sort_values("도루", ascending=False)
상위5 = 도루순.head(5)
print(f"155명 도루 합계: {int(d9['도루'].sum())} · 도루 상위 5명 합계: {int(상위5['도루'].sum())}"
      f" · 비율 {상위5['도루'].sum() / d9['도루'].sum():.3f} ({상위5['도루'].sum() / d9['도루'].sum() * 100:.1f}%)")
print("  도루 상위 5명:", " · ".join(f"{r['선수명']} {int(r['도루'])}개({r['유형']})" for _, r in 상위5.iterrows()))
준족 = d9[d9["유형"] == "준족형"]
print(f"준족형 {len(준족)}명의 도루 개수 중앙값: {준족['도루'].median():.0f}개"
      f" · 도루 5개 이하: {int((준족['도루'] <= 5).sum())}명 · 도루 20개 이상: {int((준족['도루'] >= 20).sum())}명")
print(f"155명 가운데 도루 20개 이상: {int((d9['도루'] >= 20).sum())}명")

print("-- 9-3. 유형과 실력(OPS)")
요약 = d9.groupby("유형")["OPS"].agg(["mean", "min", "max"])
for t in ["거포형", "교타형", "준족형"]:
    g = d9[d9["유형"] == t].sort_values("OPS", ascending=False)
    print(f"  {t}: 평균 OPS {요약.loc[t, 'mean']:.3f} · 최고 {g.iloc[0]['선수명']} {g.iloc[0]['OPS']:.3f}"
          f" · 최저 {g.iloc[-1]['선수명']} {g.iloc[-1]['OPS']:.3f} · 범위 {요약.loc[t, 'max'] - 요약.loc[t, 'min']:.3f}")
print(f"유형별 평균 OPS의 가장 큰 차이: {요약['mean'].max() - 요약['mean'].min():.3f}")
print(f"준족형 가운데 거포형 평균 OPS({요약.loc['거포형', 'mean']:.3f})보다 높은 선수:"
      f" {int((준족['OPS'] > 요약.loc['거포형', 'mean']).sum())}명 / {len(준족)}명")
