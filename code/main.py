# main.py · KBO 타자는 정말 '거포형·준족형·교타형' 세 유형으로 나뉘는가
# 실행: 저장소 맨 위 폴더에서 streamlit run code/main.py
from itertools import permutations

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KBO 타자 유형 나누기", page_icon="⚾", layout="wide")

데이터경로 = "data/kbo_hitters_2026_full.csv"
비율열 = ["홈런율", "도루율", "볼넷율", "삼진율", "2루타율"]
원래열 = ["홈런", "도루", "볼넷", "삼진", "2루타"]


@st.cache_data
def 불러오기():
    return pd.read_csv(데이터경로, encoding="utf-8")


def 비율_만들기(df):
    out = df.copy()
    for 비율, 원래 in zip(비율열, 원래열):
        out[비율] = out[원래] / out["타석"]
    return out


def 묶기(X, k):
    모델 = KMeans(n_clusters=k, n_init=10, random_state=0)
    번호 = 모델.fit_predict(X)
    return 모델, 번호


def 이름_붙이기(평균표, k):
    # 군집 번호는 실행마다 바뀔 수 있어서 평균값을 보고 이름을 정한다
    if k != 3:
        순서 = 평균표["홈런율"].sort_values(ascending=False).index
        return {i: f"묶음 {n + 1}" for n, i in enumerate(순서)}
    이름 = {평균표["홈런율"].idxmax(): "거포형"}
    남은 = 평균표.drop(index=list(이름))
    이름[남은["도루율"].idxmax()] = "준족형"
    for i in 평균표.index:
        이름.setdefault(i, "교타형")
    return 이름


원본 = 불러오기()

st.title("⚾ KBO 타자는 정말 세 유형으로 나뉘는가")
st.caption("2026 KBO 타자 기록 · 10101 홍길동 · 10102 김영희 · 10103 이철수 (선생님이 만든 가상 팀의 샘플)")

with st.sidebar:
    st.header("설정")
    기준 = st.slider("분석에 넣을 최소 타석", 0, 300, 100, step=10)
    k = st.slider("묶음 수 k", 2, 6, 3)
    st.caption("우리 팀의 결론은 최소 타석 100, k=3 결과입니다. 값을 바꾸면 모든 탭의 결과가 함께 바뀝니다.")

대상 = 비율_만들기(원본[(원본["타석"] >= 기준) & (원본["타석"] > 0)]).reset_index(drop=True)
Xs = StandardScaler().fit_transform(대상[비율열])
Xr = 대상[비율열].to_numpy()

탭1, 탭2, 탭3, 탭4, 탭5 = st.tabs(["① 질문", "② 데이터 보기", "③ 기준선", "④ 군집 결과", "⑤ 해석·한계"])

with 탭1:
    st.subheader("우리 팀의 질문")
    st.markdown("> **KBO 타자는 정말 '거포형·준족형·교타형' 세 유형으로 나뉘는가**")
    st.write("야구 중계에서 '거포', '준족', '교타자'라는 말을 자주 듣습니다. "
             "선수 이름이나 인상은 빼고 기록만으로 묶어도 그 세 유형이 나오는지 확인하려고 했습니다.")
    st.write("방법은 9차시에 배운 K-평균 군집입니다. 정답(유형 이름)이 없는 데이터를 비슷한 선수끼리 묶어야 하므로 군집을 골랐습니다.")

with 탭2:
    st.subheader("타석 수가 선수마다 크게 다릅니다")
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 선수", f"{len(원본)}명")
    c2.metric(f"타석 {기준} 이상", f"{len(대상)}명")
    c3.metric("타석 중앙값", f"{int(원본['타석'].median())}")
    그림 = px.histogram(원본, x="타석", nbins=40, labels={"타석": "타석", "count": "선수 수"})
    그림.add_vline(x=기준, line_dash="dash", line_color="red", annotation_text=f"기준 {기준}")
    st.plotly_chart(그림, width="stretch")
    st.write("타석이 적은 선수는 비율이 크게 튑니다. 아래 표는 전체 선수 가운데 홈런율이 높은 순서입니다. "
             "타석이 적은 선수가 위에 올라옵니다.")
    전체비율 = 비율_만들기(원본[원본["타석"] > 0])
    st.dataframe(전체비율.sort_values("홈런율", ascending=False)[["선수명", "팀", "타석", "홈런", "홈런율"]].head(8).round(3),
                 width="stretch", hide_index=True)
    st.write("인원 변화: 슬라이더로 최소 타석을 바꾸면 분석에 들어가는 인원이 달라집니다.")
    인원 = pd.DataFrame({"최소 타석": [10, 50, 100, 200, 300]})
    인원["인원"] = [int((원본["타석"] >= t).sum()) for t in 인원["최소 타석"]]
    st.dataframe(인원, width="stretch", hide_index=True)
    st.caption("키·몸무게·생년월일·포지션 빈칸은 이번 분석에 사용하지 않는 열이라 결과에 영향이 없습니다.")
    st.subheader("분석에 사용하는 다섯 비율(타석당)")
    st.dataframe(대상[["선수명", "팀", "타석"] + 비율열].round(3), width="stretch", hide_index=True)

with 탭3:
    st.subheader("기준선 1 · k를 바꾸면 실루엣 점수가 어떻게 변하는가")
    행 = []
    for kk in range(2, 7):
        모델, 번호 = 묶기(Xs, kk)
        행.append({"묶음 수 k": kk, "실루엣": round(float(silhouette_score(Xs, 번호)), 3),
                  "군집 내 제곱합": round(float(모델.inertia_), 1)})
    후보 = pd.DataFrame(행)
    a, b = st.columns(2)
    a.plotly_chart(px.line(후보, x="묶음 수 k", y="실루엣", markers=True, title="묶음 수별 실루엣"), width="stretch")
    b.plotly_chart(px.line(후보, x="묶음 수 k", y="군집 내 제곱합", markers=True, title="엘보우(군집 내 제곱합)"), width="stretch")
    st.dataframe(후보, width="stretch", hide_index=True)
    st.caption("실루엣은 -1에서 1 사이이고, 1에 가까울수록 다른 묶음과 멀리 떨어져 있다는 뜻입니다. "
               "점수가 가장 높은 k가 꼭 가장 좋은 답은 아닙니다.")

    st.subheader(f"기준선 2 · 표준화 전후 비교 (k={k})")
    _, 번호r = 묶기(Xr, k)
    _, 번호s = 묶기(Xs, k)
    c1, c2 = st.columns(2)
    c1.metric("표준화 없이 실루엣", f"{silhouette_score(Xr, 번호r):.3f}")
    c2.metric("표준화 후 실루엣", f"{silhouette_score(Xs, 번호s):.3f}")
    평균r = 대상.assign(군집=번호r).groupby("군집")[비율열].mean()
    평균r["인원"] = pd.Series(번호r).value_counts()
    st.write("표준화 없이 묶은 결과의 묶음별 평균")
    st.dataframe(평균r.round(3), width="stretch")
    st.write("다섯 비율의 표준편차(값이 퍼진 정도)")
    st.dataframe(대상[비율열].std().round(3).to_frame("표준편차").T, width="stretch")
    st.caption("표준화하지 않으면 값이 가장 넓게 퍼진 삼진율이 거리 계산을 대부분 차지합니다. "
               "실루엣은 높게 나오지만, 묶음이 삼진율의 높낮이로 나뉠 뿐입니다.")

with 탭4:
    모델, 번호 = 묶기(Xs, k)
    대상["군집번호"] = 번호
    평균 = 대상.groupby("군집번호")[비율열].mean()
    이름 = 이름_붙이기(평균, k)
    대상["유형"] = 대상["군집번호"].map(이름)
    st.subheader(f"표준화 후 k={k} 결과 · 실루엣 {silhouette_score(Xs, 번호):.3f}")
    그림 = px.scatter(대상, x="홈런율", y="도루율", color="유형", hover_name="선수명",
                    hover_data={"팀": True, "타석": True, "볼넷율": ":.3f", "삼진율": ":.3f", "2루타율": ":.3f"},
                    title="홈런율과 도루율 (색은 묶음)")
    st.plotly_chart(그림, width="stretch")
    st.caption("그래프는 다섯 비율 가운데 두 개만 보여 줍니다. 묶음은 다섯 비율을 모두 사용해 만들었습니다.")
    표 = 평균.rename(index=이름).round(3)
    표["인원"] = 대상["유형"].value_counts()
    전체 = 대상[비율열].mean().round(3)
    전체["인원"] = len(대상)
    표.loc["전체 평균"] = 전체
    st.write("묶음별 평균")
    st.dataframe(표, width="stretch")
    if k == 3:
        st.caption("이름은 평균표를 보고 우리 팀이 붙였습니다. 홈런율이 가장 높은 묶음을 거포형, 나머지 가운데 도루율이 높은 묶음을 준족형, "
                   "남은 묶음을 교타형으로 불렀습니다. 다른 설정에서는 이 규칙이 맞지 않을 수 있으니 평균표를 직접 보고 판단합니다.")
    st.write("묶음별 선수 (타석 많은 순)")
    열 = st.columns(k)
    for c, (i, n) in zip(열, sorted(이름.items(), key=lambda kv: kv[1])):
        c.markdown(f"**{n}** ({int((번호 == i).sum())}명)")
        c.dataframe(대상[대상["군집번호"] == i].sort_values("타석", ascending=False)[["선수명", "팀", "타석"]],
                    width="stretch", hide_index=True, height=300)

    st.subheader(f"후보 모델 비교 · 같은 {len(대상)}명, 같은 표준화 다섯 비율, 묶음 {k}개")
    # DBSCAN은 묶음 수를 정하지 않는다. 반경 1.5는 각 선수의 10번째 이웃까지 거리 중앙값(1.47)을 보고 정했다
    후보모델 = {
        f"K-평균(k={k})": 번호,
        f"계층적 군집(응집, ward, {k}묶음)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs),
        "DBSCAN(반경 1.5, 최소 이웃 10)": DBSCAN(eps=1.5, min_samples=10).fit_predict(Xs),
        f"가우시안 혼합({k}개)": GaussianMixture(n_components=k, random_state=0).fit_predict(Xs),
    }
    비교행 = []
    for 모델이름, 번호c in 후보모델.items():
        번호c = pd.Series(번호c)
        남김 = (번호c != -1).to_numpy()
        묶음들 = sorted(번호c[남김].unique())
        실루엣c = f"{silhouette_score(Xs[남김], 번호c[남김]):.3f}" if len(묶음들) >= 2 else "계산 불가(묶음 1개)"
        if len(묶음들) == k:
            표c = pd.crosstab(pd.Series(번호), 번호c)
            최대 = max(sum(int(표c.loc[r, c]) for r, c in zip(표c.index, 짝)) for 짝 in permutations(묶음들))
            겹침 = f"{최대}/{len(대상)} = {최대 / len(대상):.2f}"
        else:
            겹침 = f"비교 불가(묶음 {len(묶음들)}개)"
        비교행.append({"모델": 모델이름, "실루엣": 실루엣c,
                    "묶음 크기(큰 순)": " · ".join(str(v) for v in sorted(번호c[남김].value_counts().tolist(), reverse=True)),
                    "잡음(어느 묶음에도 안 듦)": int((~남김).sum()),
                    "K-평균 묶음과 겹치는 비율": 겹침})
    st.dataframe(pd.DataFrame(비교행), width="stretch", hide_index=True)
    st.caption("겹치는 비율은 두 방법의 묶음을 하나씩 짝지었을 때 같은 묶음에 든 선수가 가장 많은 경우의 인원을 전체 인원으로 나눈 값입니다. "
               "우리 팀의 최종 선택은 K-평균입니다. 수업에서 배운 방법이고, 묶음 평균으로 유형 이름을 붙이기 쉽고, "
               "실루엣도 네 방법 가운데 가장 높습니다. 최소 타석 100, k=3일 때 README의 표와 같은 값이 나옵니다.")

with 탭5:
    st.subheader("우리 팀의 해석 (최소 타석 100, k=3 기준)")
    st.markdown(
        "- 표준화 후 k=3의 실루엣은 **0.182**입니다. 0에 가까운 값이라 묶음 사이의 경계가 흐릿합니다. "
        "많은 선수가 자기 묶음 한가운데보다 이웃 묶음과의 경계 근처에 있다는 뜻입니다.\n"
        "- k=3(0.182)은 k=2~6 다섯 후보 가운데 가장 낮은 값입니다. 가장 높은 k=5(0.204)와 0.022 차이입니다. "
        "'세 유형'은 데이터가 정해 준 답이 아니라 우리가 해석하기 좋아서 고른 k입니다.\n"
        "- 그래도 평균표에서는 홈런이 많은 묶음(홈런율 0.041), 2루타가 많은 묶음(2루타율 0.053), "
        "도루가 조금 많고 삼진이 적은 묶음(도루율 0.025, 삼진율 0.167)이 보입니다. 중계에서 쓰는 세 가지 표현과 비슷한 모양입니다.\n"
        "- 유형이 뚜렷하게 나뉘지 않는다는 것도 발견입니다. 대부분의 타자는 한 가지 특징만 가진 것이 아니라 여러 특징을 조금씩 섞어 가지고 있습니다."
    )
    st.subheader("한계")
    st.markdown(
        "- 2026년 한 시즌 기록만 사용했습니다.\n"
        "- 최소 타석 100은 우리 팀이 정한 기준입니다. 50으로 낮추면 준족형이 11명으로 줄고 나머지 115명 묶음은 특징이 없어집니다.\n"
        "- 유형 이름은 군집 평균을 보고 우리 팀이 붙인 해석입니다. 준족형은 도루율이 0.025로 작아서 이름이 가장 약합니다.\n"
        "- 이 묶음으로 선수의 가치를 매기면 안 됩니다. 묶음은 '어떤 방식으로 치는가'를 나눈 것이고 '얼마나 잘 치는가'는 담지 않았습니다."
    )
