import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="서울 대기오염 상관관계 분석", layout="wide")
st.title("서울 대기오염 측정정보 상관관계 분석")

DATA_DIR = "data"

@st.cache_data
def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ("cp949", "utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def pick_time_col(df: pd.DataFrame):
    candidates = ["MSRDT", "측정일시", "DATETIME", "dateTime", "측정시간", "측정일자"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c).lower()
        if "date" in s or "time" in s or "일시" in s:
            return c
    return None

def pick_station_col(df: pd.DataFrame):
    station_candidates = ["측정소 코드", "측정소코드", "측정소", "STATION", "SITE"]
    for c in station_candidates:
        if c in df.columns:
            return c
    return None

def pick_item_value_cols(df: pd.DataFrame):
    item_candidates = ["측정항목", "항목", "Item", "ITEM"]
    value_candidates = ["평균값", "값", "Value", "VALUE", "측정값"]

    item_col = next((c for c in item_candidates if c in df.columns), None)
    value_col = next((c for c in value_candidates if c in df.columns), None)
    return item_col, value_col

def main():
    st.sidebar.header("데이터 선택")

    if not os.path.isdir(DATA_DIR):
        st.error(f"'{DATA_DIR}/' 폴더가 없음. app.py와 같은 폴더에 data 폴더를 만들고 CSV를 넣어줘.")
        st.stop()

    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])
    if len(files) == 0:
        st.error(f"'{DATA_DIR}/' 안에 csv가 없음. 예: data/AIR_HOUR_2022.csv")
        st.stop()

    file_choice = st.sidebar.selectbox("파일", files, index=0)
    if not file_choice:
        st.error("파일 선택이 비어있음. data 폴더를 확인해줘.")
        st.stop()

    path = os.path.join(DATA_DIR, file_choice)
    st.sidebar.caption(f"경로: {path}")

    df = read_csv_auto(path)
    st.sidebar.caption(f"rows: {len(df):,}, cols: {len(df.columns)}")

    # 시간 컬럼 처리
    time_col = pick_time_col(df)
    if time_col:
        df["_dt"] = pd.to_datetime(df[time_col].astype(str), format="%Y%m%d%H", errors="coerce")
        df = df.dropna(subset=["_dt"])
    else:
        st.sidebar.info("시간 컬럼을 못 찾아서 기간 필터는 생략")

    # ---- 롱 포맷이면 와이드로 변환 (핵심) ----
    station_col = pick_station_col(df)
    item_col, value_col = pick_item_value_cols(df)

    if item_col and value_col:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        idx_cols = ["_dt"] if "_dt" in df.columns else []
        if station_col:
            idx_cols.append(station_col)

        if len(idx_cols) > 0:
            wide = (
                df.dropna(subset=idx_cols + [item_col, value_col])
                  .pivot_table(index=idx_cols, columns=item_col, values=value_col, aggfunc="mean")
                  .reset_index()
            )
            wide.columns = [str(c) for c in wide.columns]
            df = wide
    # ------------------------------------------

    # 기간 필터 (와이드 변환 이후에도 _dt가 유지됨)
    if "_dt" in df.columns and pd.api.types.is_datetime64_any_dtype(df["_dt"]):
        min_dt, max_dt = df["_dt"].min(), df["_dt"].max()
        if pd.notna(min_dt) and pd.notna(max_dt):
            start, end = st.sidebar.date_input(
                "기간",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(),
                max_value=max_dt.date(),
            )
            df = df[(df["_dt"].dt.date >= start) & (df["_dt"].dt.date <= end)]

    # 차트/상관행렬 렌더링용 샘플링 (너무 크면 비거나 느림)
    CHART_MAX = st.sidebar.slider("차트용 최대 행 수", 10_000, 200_000, 50_000, step=10_000)
    if len(df) > CHART_MAX:
        df_chart = df.sample(CHART_MAX, random_state=42)
    else:
        df_chart = df

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(30), use_container_width=True)

    # 숫자 컬럼 만들기
    num = df_chart.copy()
    for c in num.columns:
        if c == "_dt":
            continue
        if num[c].dtype == "object":
            num[c] = pd.to_numeric(num[c], errors="coerce")

    exclude = {"_dt"}
    if station_col and station_col in num.columns:
        exclude.add(station_col)

    numeric_cols = [c for c in num.columns if c not in exclude and pd.api.types.is_numeric_dtype(num[c])]
    numeric_cols = [c for c in numeric_cols if num[c].notna().sum() > 100]

    if len(numeric_cols) < 2:
        st.error("숫자 컬럼이 2개 미만이라 분석 불가. (오염물질 컬럼이 안 펼쳐졌을 가능성) CSV 컬럼명을 확인해줘.")
        st.stop()

    st.subheader("변수 선택")
    x = st.selectbox("X", numeric_cols, index=0)
    y = st.selectbox("Y", numeric_cols, index=1)

    clean = num[[x, y]].dropna()
    st.caption(f"유효 표본 수 n = {len(clean):,}")

    st.subheader("상관계수")
    r = clean[x].corr(clean[y])
    st.write({"pearson_r": float(r) if pd.notna(r) else None})

    st.subheader("산점도")
    if len(clean) == 0:
        st.info("필터/결측 제거 후 남은 데이터가 없음.")
    else:
        st.scatter_chart(clean.rename(columns={x: "x", y: "y"}))

    st.subheader("상관행렬")
    default_cols = numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
    cols_for_corr = st.multiselect("포함 변수", numeric_cols, default=default_cols)
    if len(cols_for_corr) >= 2:
        st.dataframe(num[cols_for_corr].corr().style.format("{:.3f}"), use_container_width=True)
    else:
        st.info("상관행렬은 2개 이상 변수를 선택해야 표시됨.")

    st.subheader("해석")
    st.write(
        "- r이 1에 가까우면 양의 상관, -1에 가까우면 음의 상관\n"
        "- 상관은 인과가 아님\n"
        "- 기간/측정소 필터에 따라 결과가 달라질 수 있음"
    )

if __name__ == "__main__":
    main()
