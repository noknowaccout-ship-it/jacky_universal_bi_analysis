import streamlit as st
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===================== 配置 =====================
st.set_page_config(page_title="📊 通用智能数据分析引擎", layout="wide")
MAX_RETRY = 1

# 会话
if "history" not in st.session_state:
    st.session_state.history = []
if "df_dict" not in st.session_state:
    st.session_state.df_dict = {}
if "current_sheet" not in st.session_state:
    st.session_state.current_sheet = None

# ===================== 模型 =====================
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen2:7b-instruct", temperature=0.0, num_ctx=8192)

# ===================== 通用文件读取 =====================
def auto_read_file(f):
    sheets = {}
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(f, encoding="utf-8")
        except:
            df = pd.read_csv(f, encoding="gbk")
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        sheets["CSV数据"] = df
        return sheets
    try:
        xl = pd.ExcelFile(f)
        for s in xl.sheet_names:
            df = pd.read_excel(f, sheet_name=s, engine="openpyxl")
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]
            sheets[s] = df
        return sheets
    except:
        return {}

# ===================== 【核心】通用代码生成规则 =====================
def generate_code(df_info, question):
    prompt = f"""
你是通用数据分析助手，必须严格遵守以下规则，保证100%通用、不绑定业务、不崩溃：

规则：
1. 只对数值列进行计算，自动跳过字符串列（姓名、部门、ID等）
2. 不自动定义任何阈值、不自动定义异常、不使用z-score
3. 如果用户要找异常，必须使用明确条件 > < >= <= ，不能自动猜测
4. 代码必须通用，适用于任何表格
5. 结果存入变量 result
6. 只输出纯Python代码，不要解释

数据表结构：
{df_info}

用户问题：{question}
"""
    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "")
    return code

# ===================== 执行 =====================
def run_code(df, code):
    try:
        env = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, globals(), env)
        return True, env.get("result"), ""
    except Exception as e:
        return False, None, traceback.format_exc()

# ===================== 总结 =====================
def make_summary(q, res):
    if res is None or len(res) == 0:
        return "未查询到符合条件的数据"
    prompt = f"请用简洁通用语言总结结果，不使用业务术语：{q}，结果：{str(res.head(10))}"
    return llm.invoke(prompt).content

# ===================== 导出 =====================
def save_excel(data):
    p = "analysis_result.xlsx"
    pd.DataFrame(data).to_excel(p, index=False)
    return p

def save_pdf(txt):
    p = "analysis_report.pdf"
    c = canvas.Canvas(p, pagesize=A4)
    c.drawString(50, 800, "数据分析报告")
    c.drawString(50, 780, str(datetime.now()))
    y = 750
    for line in txt.split("\n")[:30]:
        c.drawString(50, y, line[:85])
        y -= 16
    c.save()
    return p

# ===================== 界面 =====================
st.title("📊 通用智能数据分析引擎")
st.markdown("✅ 100%通用 | ✅ Excel/CSV | ✅ 多轮对话 | ✅ 不绑定业务 | ✅ 稳定不崩溃")

file = st.file_uploader("上传文件", type=["xlsx", "csv"])
if file:
    sheets = auto_read_file(file)
    st.session_state.df_dict = sheets
    sel = st.selectbox("选择数据表", list(sheets.keys()))
    st.session_state.current_sheet = sel
    st.success(f"✅ 加载成功：{len(sheets)} 张表")
    st.dataframe(sheets[sel].head(10), width='stretch')

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        st.code(h["code"])
        st.dataframe(h["res"])
        st.info(h["sum"])

q = st.chat_input("输入分析需求（例如：排序、找最大值、筛选、统计、分组、对比...）")
if q and st.session_state.current_sheet:
    df = st.session_state.df_dict[st.session_state.current_sheet]
    info = f"列名：{list(df.columns)}\n前5行数据：{df.head().to_string()}"

    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        code = generate_code(info, q)
        st.subheader("📝 生成代码")
        st.code(code)

        ok, res, err = run_code(df, code)
        if not ok:
            st.error("❌ 执行失败，错误信息：")
            st.code(err)
        else:
            st.success("✅ 执行成功")
            st.dataframe(res, width='stretch')
            s = make_summary(q, res)
            st.subheader("📄 分析总结")
            st.info(s)
            st.session_state.history.append({"q":q,"code":code,"res":res,"sum":s})

            col1, col2 = st.columns(2)
            with col1:
                with open(save_excel(res), "rb") as f:
                    st.download_button("📥 Excel报告", f)
            with col2:
                with open(save_pdf(s), "rb") as f:
                    st.download_button("📥 PDF报告", f)