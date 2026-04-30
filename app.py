import os
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI


KNOWLEDGE_DIR = Path("./knowledge")
API_KEY_FILE = Path("./openai_api_key.txt")
MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = """
あなたは建築設計事務所向けの法規チェック・アシスタントです。
次の優先順位を厳守してください。
1) 数値判定（容積率・建ぺい率）は、ユーザー入力の計算結果と制限値を優先して判断する。
2) knowledge の条文テキストから「数値制限」「適用条件」「例外条件」を優先抽出し、出典付きで示す。
3) 判定不能時は「要追加情報」とし、追加で聞くべき項目と、条件付きの暫定判定を示す。
4) 不要に曖昧な「要確認」のみで終わらせない。
""".strip()

ZONING_DEFAULT_LIMITS: Dict[str, Dict[str, float]] = {
    "第一種低層住居専用地域": {"far": 80.0, "bcr": 50.0},
    "第二種低層住居専用地域": {"far": 100.0, "bcr": 60.0},
    "第一種中高層住居専用地域": {"far": 200.0, "bcr": 60.0},
    "第二種中高層住居専用地域": {"far": 200.0, "bcr": 60.0},
    "第一種住居地域": {"far": 200.0, "bcr": 60.0},
    "第二種住居地域": {"far": 200.0, "bcr": 60.0},
    "準住居地域": {"far": 200.0, "bcr": 60.0},
    "近隣商業地域": {"far": 300.0, "bcr": 80.0},
    "商業地域": {"far": 400.0, "bcr": 80.0},
    "準工業地域": {"far": 200.0, "bcr": 60.0},
    "工業地域": {"far": 200.0, "bcr": 60.0},
    "工業専用地域": {"far": 200.0, "bcr": 60.0},
}


def load_knowledge_chunks(knowledge_dir: Path) -> List[dict]:
    chunks: List[dict] = []
    if not knowledge_dir.exists():
        return chunks

    for file_path in knowledge_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for idx, paragraph in enumerate(paragraphs, start=1):
            chunks.append(
                {
                    "source": file_path.name,
                    "chunk_id": idx,
                    "text": paragraph,
                }
            )
    return chunks


def simple_retrieve(query: str, chunks: List[dict], top_k: int = 4) -> List[dict]:
    keywords = [w for w in query.replace("　", " ").split(" ") if w.strip()]
    if not keywords:
        return chunks[:top_k]

    scored: List[Tuple[int, dict]] = []
    for c in chunks:
        score = sum(1 for k in keywords if k in c["text"])
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [c for _, c in scored[:top_k]]
    return chunks[:top_k]


def calc_ratios(site_area: float, total_floor_area: float, building_area: float) -> Dict[str, float]:
    far = (total_floor_area / site_area) * 100.0
    bcr = (building_area / site_area) * 100.0
    return {"far": far, "bcr": bcr}


def evaluate_ratio(value: float, limit: float) -> Tuple[str, str]:
    if value <= limit:
        return ("適合", f"{value:.1f}% <= {limit:.1f}%")
    return ("不適合", f"{value:.1f}% > {limit:.1f}%")


def detect_missing_info(question: str, road_width: str, north_side_distance: str, height_district: str) -> List[str]:
    missing: List[str] = []
    q = question.strip()

    if any(k in q for k in ["北側斜線", "斜線", "高度地区"]):
        if not north_side_distance.strip():
            missing.append("北側隣地境界までの距離 (m)")
        if not height_district.strip():
            missing.append("高度地区の指定有無・種別")

    if any(k in q for k in ["道路斜線", "容積率", "前面道路"]):
        if not road_width.strip():
            missing.append("前面道路幅員 (m)")

    return missing


def build_prompt(
    zoning: str,
    site_area: float,
    total_floor_area: float,
    building_area: float,
    far_limit: float,
    bcr_limit: float,
    far_judgement: str,
    bcr_judgement: str,
    road_width: str,
    north_side_distance: str,
    height_district: str,
    missing_info: List[str],
    question: str,
    contexts: List[dict],
) -> str:
    context_text = "\n\n".join(
        [
            f"[出典: {c['source']} / チャンク{c['chunk_id']}]\n{c['text']}"
            for c in contexts
        ]
    )

    return f"""
あなたは建築設計事務所向けの法規チェック・アシスタントです。
以下の条件と参考条文を読み、回答してください。

## 入力条件
- 用途地域: {zoning}
- 敷地面積: {site_area} ㎡
- 延床面積: {total_floor_area} ㎡
- 建築面積: {building_area} ㎡
- 指定容積率: {far_limit} %
- 指定建ぺい率: {bcr_limit} %
- 前面道路幅員: {road_width or "未入力"}
- 北側隣地境界までの距離: {north_side_distance or "未入力"}
- 高度地区: {height_district or "未入力"}
- 質問: {question}

## 数値判定（コードで先行実施）
- 容積率判定: {far_judgement}
- 建ぺい率判定: {bcr_judgement}
- 不足情報: {", ".join(missing_info) if missing_info else "なし"}

## 参考条文（RAG取得）
{context_text}

## 回答要件
冒頭に必ず「思考プロセス」を出し、以下の順で記載:
① 入力された条件の整理
② 参照した法規の特定
③ 数値の照合・比較
④ 結論（判定）

その後、次の見出しをこの順で記載:
- 判定（適合 / 不適合 / 要追加情報）
- 根拠条文（数値制限を優先）
- 設計上のアドバイス
- 追加確認事項

特に重要:
- 容積率・建ぺい率は「コード計算結果」を覆さないこと。
- 断定に不足がある場合は、必ず逆質問を1つ以上含めること。
- 逆質問では「もし X が Y なら適合/不適合」の条件付き見解も示すこと。

注意:
- これはデモ用途。法的確定判断ではなく、最終判断は行政窓口や専門家確認が必要。
- 回答は日本語で簡潔に。
""".strip()


def load_api_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    if API_KEY_FILE.exists():
        lines = API_KEY_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                return raw
    return ""


def call_llm(prompt: str) -> str:
    api_key = load_api_key()
    if not api_key:
        return (
            "API キーが設定されていません。\n"
            "`openai_api_key.txt` の1行目に API キーを入力するか、"
            "`OPENAI_API_KEY` 環境変数を設定してください。"
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.output_text


def main() -> None:
    st.set_page_config(page_title="AI法規チェック・アシスタント", page_icon="🏗️", layout="wide")

    st.title("🏗️ AI法規チェック・アシスタント（PoC）")
    st.caption("建築基準法・条例テキストを参照して、設計条件に対する一次判定と根拠提示を行うデモです。")

    with st.sidebar:
        st.header("設定")
        st.markdown("- モデル: `gpt-4o-mini`\n- 参照元: `./knowledge/*.txt`")
        st.markdown("- APIキー読込: `OPENAI_API_KEY` または `./openai_api_key.txt`")
        st.info("※ 本ツールはデモ版です。最終的な法適合判断は必ず専門家確認を行ってください。")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("条件入力")
        zoning = st.selectbox(
            "用途地域",
            [
                "第一種低層住居専用地域",
                "第二種低層住居専用地域",
                "第一種中高層住居専用地域",
                "第二種中高層住居専用地域",
                "第一種住居地域",
                "第二種住居地域",
                "準住居地域",
                "近隣商業地域",
                "商業地域",
                "準工業地域",
                "工業地域",
                "工業専用地域",
            ],
        )
        site_area = st.number_input("敷地面積 (㎡)", min_value=1.0, value=120.0, step=1.0)
        total_floor_area = st.number_input("延床面積 (㎡)", min_value=1.0, value=180.0, step=1.0)
        building_area = st.number_input("建築面積 (㎡)", min_value=1.0, value=72.0, step=1.0)
        default_limits = ZONING_DEFAULT_LIMITS.get(zoning, {"far": 200.0, "bcr": 60.0})
        far_limit = st.number_input("指定容積率 (%)", min_value=1.0, value=default_limits["far"], step=1.0)
        bcr_limit = st.number_input("指定建ぺい率 (%)", min_value=1.0, value=default_limits["bcr"], step=1.0)
        st.caption("※ 上記制限値は用途地域の代表値。案件ごとの都市計画指定値で上書きしてください。")
        road_width = st.text_input("前面道路幅員 (m) ※任意", value="")
        north_side_distance = st.text_input("北側隣地境界までの距離 (m) ※任意", value="")
        height_district = st.text_input("高度地区の指定 (例: 第1種高度地区) ※任意", value="")
        question = st.text_area(
            "質問",
            value="この条件で北側斜線制限と容積率は問題ありませんか？",
            height=120,
        )
        run_btn = st.button("チェック実行", type="primary")

    with col2:
        st.subheader("回答")
        result_placeholder = st.empty()

    if run_btn:
        if not question.strip():
            st.warning("質問を入力してください。")
            return

        chunks = load_knowledge_chunks(KNOWLEDGE_DIR)
        if not chunks:
            result_placeholder.error("`./knowledge/` にテキストが見つかりません。サンプルデータを配置してください。")
            return

        rag_query = f"{zoning} 敷地面積{site_area} 延床面積{total_floor_area} {question}"
        contexts = simple_retrieve(rag_query, chunks, top_k=4)
        ratios = calc_ratios(site_area, total_floor_area, building_area)
        far_status, far_detail = evaluate_ratio(ratios["far"], far_limit)
        bcr_status, bcr_detail = evaluate_ratio(ratios["bcr"], bcr_limit)
        missing_info = detect_missing_info(question, road_width, north_side_distance, height_district)

        if "不適合" in [far_status, bcr_status]:
            final_status = "不適合"
        elif missing_info:
            final_status = "要追加情報"
        else:
            final_status = "適合"

        far_judgement = f"{far_status}（{far_detail}）"
        bcr_judgement = f"{bcr_status}（{bcr_detail}）"
        prompt = build_prompt(
            zoning,
            site_area,
            total_floor_area,
            building_area,
            far_limit,
            bcr_limit,
            far_judgement,
            bcr_judgement,
            road_width,
            north_side_distance,
            height_district,
            missing_info,
            question,
            contexts,
        )

        with st.spinner("法規チェック中..."):
            answer = call_llm(prompt)

        if final_status == "適合":
            result_placeholder.success("判定: 適合")
        elif final_status == "不適合":
            result_placeholder.error("判定: 不適合")
        else:
            result_placeholder.warning("判定: 要追加情報")

        st.markdown(
            f"- 計算容積率: **{ratios['far']:.1f}%**（上限 {far_limit:.1f}%）\n"
            f"- 計算建ぺい率: **{ratios['bcr']:.1f}%**（上限 {bcr_limit:.1f}%）"
        )
        if missing_info:
            st.warning("不足情報: " + " / ".join(missing_info))

        st.markdown(answer)

        with st.expander("参照したテキストチャンク"):
            for c in contexts:
                st.markdown(f"**{c['source']} / チャンク{c['chunk_id']}**")
                st.write(c["text"])


if __name__ == "__main__":
    main()
