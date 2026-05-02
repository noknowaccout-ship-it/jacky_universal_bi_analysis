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

        # 确保数值列被正确识别和转换
        for col in df.columns:
            try:
                # 尝试将可能的数值列转换为数值类型
                if df[col].dtype == 'object':
                    # 检查是否包含数值数据
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # 尝试转换为数值
                        try:
                            pd.to_numeric(sample_values, errors='coerce')
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass  # 保持原类型
            except:
                pass  # 保持原类型

        sheets["CSV数据"] = df
        return sheets
    try:
        xl = pd.ExcelFile(f)
        for s in xl.sheet_names:
            df = pd.read_excel(f, sheet_name=s, engine="openpyxl")
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df.columns = [str(c).strip() for c in df.columns]

            # 确保数值列被正确识别和转换
            for col in df.columns:
                try:
                    if df[col].dtype == 'object':
                        sample_values = df[col].dropna().head(10)
                        if len(sample_values) > 0:
                            try:
                                pd.to_numeric(sample_values, errors='coerce')
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            except:
                                pass
                except:
                    pass

            sheets[s] = df
        return sheets
    except:
        return {}

# ===================== DataFrame 兼容性处理 =====================
def make_dataframe_arrow_compatible(df):
    """确保 DataFrame 与 Streamlit/Arrow 兼容"""
    if df.empty:
        return df

    df_copy = df.copy()

    for col in df_copy.columns:
        try:
            # 处理数值列
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                # 将 NaN 转换为合适的数值
                if df_copy[col].isna().any():
                    if pd.api.types.is_integer_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].fillna(0).astype(int)
                    else:
                        df_copy[col] = df_copy[col].fillna(0.0)
            # 处理字符串列
            elif df_copy[col].dtype == 'object':
                # 确保字符串列没有混合类型
                df_copy[col] = df_copy[col].astype(str)
        except Exception:
            # 如果转换失败，保持原类型
            pass

    return df_copy

# ===================== 【核心】通用代码生成规则 =====================
def generate_code(df_info, question, error_info=None):
    base_prompt = f"""
你是一个通用数据分析代码生成器。你的任务是根据表格结构和用户问题判断最适合的分析能力，并生成可直接执行的 Python 代码。

你要根据用户文字描述判断是否需要以下能力之一：
- 汇总统计（均值、总和、计数、最小/最大、中位数等）
- 分组对比（groupby、同类比较、类别比较）
- 异常分析（明确条件筛选、候选极端值、与同类/均值比较）
- 相关性分析（corr 矩阵、变量间关系）
- 分布分析（分位数、频数、区间分布）
- 趋势/变化分析（环比、同比、差异计算）
- 过滤与排序（筛选条件、top/bottom、排序）

根据用户描述，还要判断是否属于以下模式：
- 需要明确返回异常候选或极端数据 -> 异常分析
- 需要比较不同类别或组之间的 KPI -> 分组对比
- 需要查看指标之间的关联 -> 相关性分析
- 需要了解指标分布特征 -> 分布分析
- 需要计算时间序列变化趋势 -> 趋势分析
- 需要返回统计指标或汇总信息 -> 汇总统计
- 需要筛选、排序或获取 top/bottom -> 过滤与排序

请先判断用户请求是“记录查询”还是“分析计算”。
- 如果用户明确要求“列出记录”、“显示明细”、“导出行数据”、“查找记录”、“查看详情”、“明细”、“原始数据”等，且目标是获取具体数据行，则返回筛选或排序后的数据行。
- 如果用户提问中包含“分析”、“对比”、“趋势”、“相关”、“分布”、“异常”、“洞察”、“指标”、“结论”、“变化”、“偏离”、“贡献”、“表现”、“能力”、“能力分析”、“比较”、“排名”、“风险”、“问题”、“异常值”等词语，则优先返回计算后的分析结果，而不是简单返回原数据行。
- 如果用户同时要求“明细”和“分析结论”，应返回带有计算指标或对比结果的结构，如 DataFrame、dict 或 list，并附加说明性计算字段；不要只返回原始行数据。
- 如果用户没有明确要求“topN”、“前几行”、“样例数据”等，分析结果应基于全部有效数据进行计算，不要默认只使用部分代理人或部分行；不要在结果代码中使用 df.head()、df.tail()、sample 或只取部分原始数据作为结果。

对于分析型请求，输出应优先包含专业技术性计算结果，例如：
- 汇总统计：均值、总和、计数、最小/最大、中位数、方差、标准差等；
- 分组对比：按组计算平均值、占比、差异、排名等；
- 相关性：相关系数矩阵、协方差、变量间关系；
- 分布：分位数、频数分布、区间统计、偏态说明；
- 趋势变化：环比/同比、差异率、增长率、变化趋势；
- 异常候选：明确条件过滤、偏离均值/中位数的极端值、最差/最佳记录。

如果分析结果需要给出数据行明细，应同时返回必要的计算指标或对比说明，例如：
- 按部门排名时，返回部门排名和值；
- 异常分析时，返回异常候选行并附带"异常分数""偏离程度"等计算字段，且按异常程度排序；
- 相关性分析时，返回相关矩阵而不是仅列表。

特别地，对于异常分析：
- 不要只返回原数据行；必须添加计算字段说明为什么是异常；
- 计算每条记录的"异常分数"或"偏离指数"，定义为该记录中超出阈值的数值列占比、或偏离程度加权和；
- 按异常分数从高到低排序；
- 可以添加"异常字段列表"说明该行哪些列最异常；
- 返回的 DataFrame 应包含：原数据所有列 + 异常分数 + 异常字段列表（可选）。

分析结果应优先包含技术性计算值，如：
- 各类汇总指标、分组对比统计、相关性矩阵、分布区间、趋势变化、异常候选等；
- 如确实需要给出记录明细，也应附带说明性计算指标或差异结果；
- 仅在用户明确请求明细记录时，返回原数据行。

要求：
1. 只输出纯 Python 代码，不要解释、不要生成任何 markdown、代码块标记、注释或自然语言。
2. 代码执行后所有文本必须是代码行，不允许在代码后添加任何自然语言说明、总结、提示或中文解释。
2. 只使用 df、pd、np 作为变量，不能导入新的库，也不要调用未定义的函数。
3. 结果必须存入变量 result。
4. df 是完整的用户数据集，已在应用内从文件读取；不要假设 df 只有前几行，也不要生成注释说明 df 仅包含前 5 行数据。
5. 不要生成任何用于创建或替换 df 的代码，例如 `df = pd.DataFrame(...)`、`df = pd.read_csv(...)` 或其他硬编码数据；代码必须直接使用传入的 df 变量。
6. 生成代码应使用标准 ASCII 标点符号，例如英文逗号、英文引号、英文括号，避免使用中文全角符号。
7. 只对数值列进行计算，自动跳过非数值列；如果涉及字符串列，则仅用于分组、筛选或标签，不作为数值计算对象。
8. 当需要聚合或统计时，先用 `numeric_cols = df.select_dtypes(include=[np.number]).columns`，并仅对这些数值列执行 `mean()`、`std()`、`describe()`、`corr()` 等操作；避免对字符串列执行数值聚合。
9. 如果需要对多个数值列筛选异常或候选行，不要直接使用 `df[(df[numeric_cols] - mean_values).abs() > ...]` 这种方式，因为它会生成带 NaN 的掩码结果。必须先得到一个按行的布尔型 Series，例如 `mask = (df[numeric_cols] - mean_values).abs() > threshold; result = df[mask.any(axis=1)]`。
10. 不自动定义任何阈值、不自动定义异常、不使用 z-score。
10. 不得使用任何深度学习、机器学习或统计建模相关的库和概念，例如 tensorflow、torch、keras、sklearn、xgboost、lightgbm、catboost、statsmodels、prophet 等。
11. 如果用户问题涉及训练模型、预测、分类、回归、深度学习、特征工程等高级建模任务，不要生成模型训练或预测代码；返回一个空结果结构，例如 result = pd.DataFrame([]) 或 result = []。
12. 如果用户询问"异常"或"异常值"、"找出异常"等，必须在返回结果中添加计算字段说明"为什么异常"，不要只返回筛选后的原始行：
   - 计算每条记录的"异常分数"或"偏离指数"（例如该记录中超出均值 2*std 的数值列占比）；
   - 按异常分数从高到低排序；
   - 返回的 DataFrame 必须包含原数据所有列 + 新增的"异常分数""偏离程度"等计算字段；
   - 可选：添加"异常字段"列表说明该行哪些列最异常、异常方向（过高/过低）。
   - 如果用户明确给出条件（> < >= <=），优先使用用户条件；否则可用均值 +- 2*std 作为默认阈值。
13. 代码必须通用，适用于任何表格，不能绑定具体业务场景或字段名称。
14. 结果可以是 pandas.DataFrame、Series、list、dict、数字等常见可展示结构。
15. 如果用户希望“可视化”、“图表”、“图像”或“报告”，不要生成任何绘图代码；只返回数据计算结果。
16. 如果问题无法直接计算，返回一个空结构而不是报错，例如 result = pd.DataFrame([]) 或 result = []。
"""
    if error_info:
        base_prompt += f"\n\n之前的代码执行失败，错误信息：{error_info}\n请修复代码，确保其正确执行并符合所有要求。"
    
    prompt = base_prompt + f"\n\n数据表结构：\n{df_info}\n\n用户问题：{question}"
    code = llm.invoke(prompt).content.strip()
    code = code.replace("```python", "").replace("```", "")
    
    # 严格提取代码：移除代码后的中文说明文本
    lines = code.split('\n')
    result_lines = []
    for line in lines:
        stripped = line.strip()
        # 空行保留
        if not stripped:
            result_lines.append(line)
            continue
        # 如果行以#开头（注释），保留
        if stripped.startswith('#'):
            result_lines.append(line)
            continue
        # 检查是否全是中文（无代码特征）：包含中文字符且不包含任何代码标志符
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in stripped)
        has_code_chars = any(c in stripped for c in '()[]{}=+-*/%.,"\':;')
        if has_chinese and not has_code_chars:
            # 这是纯中文行，停止提取代码
            break
        result_lines.append(line)
    
    code = '\n'.join(result_lines).strip()
    return code

# ===================== 执行 =====================
def run_code(df, code):
    try:
        code = code.replace("，", ",").replace("；", ";").replace("：", ":")
        code = code.replace("（", "(").replace("）", ")")
        code = code.replace("“", '"').replace("”", '"')
        code = code.replace("‘", "'").replace("’", "'")
        if "df = pd.DataFrame(" in code or "df = pd.read_csv(" in code or "df = pd.read_excel(" in code:
            return False, None, "生成代码中包含硬编码或数据加载逻辑，请确保直接使用传入的 df 变量。"
        env = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, globals(), env)
        result = env.get("result")
        if isinstance(result, pd.DataFrame):
            object_cols = [c for c in result.columns if result[c].dtype == object]
            masked_cols = [c for c in object_cols if result[c].isna().all() and c in df.columns and df[c].notna().any()]
            if masked_cols:
                return False, None, (
                    "生成结果 DataFrame 中非数值列被全部掩码，通常是因为使用了 df[(df[numeric_cols] - mean_values).abs() > ...] 这种错误过滤方式。"
                    "请改用按行筛选：mask = ...; result = df[mask.any(axis=1)]。"
                )
        return True, result, ""
    except Exception as e:
        return False, None, traceback.format_exc()

# ===================== 总结 =====================
def make_summary(q, res):
    if res is None:
        return "未生成任何结果"

    # 准备传递给 LLM 的完整结果信息
    if isinstance(res, pd.DataFrame):
        if res.empty:
            result_info = "结果为空 DataFrame，未查询到符合条件的数据。"
        else:
            result_info = f"类型: DataFrame, 行数: {res.shape[0]}, 列数: {res.shape[1]}, 列名: {list(res.columns)}\n"
            result_info += f"完整结果数据:\n{res.to_csv(index=False)}"

    elif isinstance(res, pd.Series):
        if res.empty:
            result_info = "结果为空 Series，未查询到符合条件的数据。"
        else:
            result_info = f"类型: Series, 长度: {len(res)}"
            if pd.api.types.is_numeric_dtype(res):
                result_info += f", 统计: 平均值 {res.mean():.2f}, 标准差 {res.std(ddof=0):.2f}, 最小值 {res.min():.2f}, 最大值 {res.max():.2f}"
            result_info += f"\n完整结果数据:\n{res.to_string()}"

    elif isinstance(res, dict):
        if len(res) == 0:
            result_info = "结果为空字典。"
        else:
            result_info = f"类型: dict, 键数: {len(res)}, 完整内容: {res}"

    elif isinstance(res, list):
        if len(res) == 0:
            result_info = "结果为空列表。"
        else:
            result_info = f"类型: list, 长度: {len(res)}, 完整内容: {res}"

    else:
        result_info = f"类型: {type(res).__name__}, 值: {res}"

    prompt = (
        "你是通用数据分析助手。请基于提供的完整分析结果数据，用简洁通用语言总结分析结论。\n"
        "重要：只能使用结果数据中的实际数值和信息，不要编造或推测不存在的数据。\n"
        "请深入分析数据中的模式、趋势、异常和洞察。\n"
        "不要使用具体业务、公司、行业或岗位术语。\n"
        f"用户问题：{q}\n"
        f"完整结果数据：\n{result_info}\n"
        "请给出基于真实数据的详细分析结论。"
    )
    return llm.invoke(prompt).content

# ===================== 导出 =====================
def save_excel(data):
    p = "analysis_result.xlsx"
    if isinstance(data, pd.DataFrame):
        df_out = data
    elif isinstance(data, pd.Series):
        df_out = data.to_frame().reset_index()
    elif isinstance(data, dict):
        df_out = pd.DataFrame([data])
    elif isinstance(data, list):
        try:
            df_out = pd.DataFrame(data)
        except Exception:
            df_out = pd.DataFrame({"value": data})
    else:
        df_out = pd.DataFrame([{"value": data}])
    df_out.to_excel(p, index=False)
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

# 自动加载默认文件
import os
default_file = "kpi_agents.csv"
if os.path.exists(default_file) and not st.session_state.df_dict:
    try:
        with open(default_file, "rb") as f:
            from io import BytesIO
            file_like = BytesIO(f.read())
            file_like.name = default_file
            sheets = auto_read_file(file_like)
            if sheets:
                st.session_state.df_dict = sheets
                st.session_state.current_sheet = list(sheets.keys())[0]
                st.success(f"✅ 自动加载默认文件：{default_file} ({len(sheets)} 张表)")
                st.dataframe(make_dataframe_arrow_compatible(sheets[st.session_state.current_sheet].head(10)), width='stretch')
    except Exception as e:
        st.warning(f"⚠️ 自动加载默认文件失败：{e}")

file = st.file_uploader("上传文件", type=["xlsx", "csv"])
if file:
    sheets = auto_read_file(file)
    st.session_state.df_dict = sheets
    sel = st.selectbox("选择数据表", list(sheets.keys()))
    st.session_state.current_sheet = sel
    st.success(f"✅ 加载成功：{len(sheets)} 张表")
    st.dataframe(make_dataframe_arrow_compatible(sheets[sel].head(10)), width='stretch')

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        st.code(h["code"])
        st.dataframe(make_dataframe_arrow_compatible(h["res"]))
        st.info(h["sum"])

q = st.chat_input("输入分析需求（例如：排序、找最大值、筛选、统计、分组、对比...）")
if q and st.session_state.current_sheet:
    df = st.session_state.df_dict[st.session_state.current_sheet]
    info = (
        f"列名：{list(df.columns)}\n"
        f"数值列：{list(df.select_dtypes(include=[np.number]).columns)}\n"
        f"非数值列：{list(df.select_dtypes(exclude=[np.number]).columns)}\n"
        f"数据行数：{len(df)}"
    )

    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        retry_count = 0
        error_info = None
        while retry_count <= MAX_RETRY:
            code = generate_code(info, q, error_info)
            st.subheader("📝 生成代码" + (f" (重试 {retry_count})" if retry_count > 0 else ""))
            st.code(code)

            ok, res, err = run_code(df, code)
            if ok:
                st.success("✅ 执行成功")
                st.dataframe(make_dataframe_arrow_compatible(res), width='stretch')
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
                break
            else:
                st.error(f"❌ 执行失败 (尝试 {retry_count + 1}/{MAX_RETRY + 1})，错误信息：")
                st.code(err)
                if retry_count < MAX_RETRY:
                    error_info = err
                    retry_count += 1
                    st.info("🔄 正在重试生成修复代码...")
                else:
                    st.error("❌ 重试次数已达上限，无法修复。")
                    break