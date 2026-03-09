"""
文本规范化模块 - 处理各类绕过审核的手段
Text Normalization Module - handles various bypass attempts
"""

import re
import unicodedata

# ── 全角转半角映射表 ─────────────────────────────────────────────────────────
# 全角字符（FF01-FF5E）与半角字符（0021-007E）之间的偏移量为 0xFEE0
_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_SPACE = "\u3000"  # 全角空格单独处理

# ── 繁体字到简体字的基础映射（常用字） ───────────────────────────────────────
# 仅包含与风险词汇相关的常用繁体字，避免引入过重的依赖
_TRAD_TO_SIMP: dict[str, str] = {
    "幹": "干", "媽": "妈", "妳": "你", "妳": "你",
    "臺": "台", "發": "发", "傳": "传", "點": "点",
    "錢": "钱", "騙": "骗", "賺": "赚", "贏": "赢",
    "賭": "赌", "愛": "爱", "親": "亲", "覺": "觉",
    "說": "说", "話": "话", "讓": "让", "過": "过",
    "這": "这", "還": "还", "來": "来", "時": "时",
    "們": "们", "個": "个", "國": "国", "會": "会",
    "學": "学", "見": "见", "對": "对", "長": "长",
    "開": "开", "關": "关", "問": "问", "頭": "头",
    "號": "号", "當": "当", "電": "电", "機": "机",
    "網": "网", "線": "线", "轉": "转", "帳": "账",
    "實": "实", "際": "际", "業": "业", "務": "务",
    "費": "费", "獲": "获", "利": "利", "潤": "润",
    "歲": "岁", "裸": "裸", "體": "体", "視": "视",
    "頻": "频", "圖": "图", "片": "片", "色": "色",
    "情": "情", "黃": "黄", "毒": "毒", "品": "品",
}

# ── 谐音字替换表（常见以谐音绕过审核的字） ──────────────────────────────────
# key 是可能出现的替代字/词（用于绕过审核的变体），value 是真实/规范含义
# 注意：key 与 value 不能相同，否则会导致无限循环
_HOMOPHONES: dict[str, str] = {
    # 辱骂类谐音（变体 → 规范词）
    "艹": "操",
    "曹": "操",
    "肏": "操",
    "槽": "操",
    "草你": "操你",
    "草他": "操他",
    "草她": "操她",
    "马的": "妈的",    # "你马的" → "你妈的"
    "码的": "妈的",
    "麻的": "妈的",
    "比比": "屄屄",    # 辱骂性双叠用法
    "批批": "屄屄",
    "逼逼": "屄屄",
    "沙雕": "傻屌",
    # 诈骗类谐音
    "密玛": "密码",
    "验正码": "验证码",
}

# ── 拆字重组映射表（常见拆字写法） ──────────────────────────────────────────
# 用户将汉字拆开写，如"口人" → "囚"，"口十" → "叶"
_SPLIT_CHAR_MAP: dict[str, str] = {
    "口人": "囚",
    "人口": "囚",
    "日月": "明",
    "土也": "地",
    "氵去": "法",
    "扌莫": "摸",
    "扌莫": "摸",
    "女马": "妈",
    "女鸟": "妈",  # 另一种拆法
    "力口": "加",
    "讠人": "认",
    "讠司": "词",
    "讠周": "调",
    "纟合": "给",
    "钅钱": "钱",
    "月巴": "肥",
    "氵每": "海",
    "氵先": "洗",
    "木目": "相",
    "艹操": "操",  # 有时直接写艹
    "月+又": "胸",
}

# ── leet-speak / 数字字母替换 ────────────────────────────────────────────────
# 常见数字/符号替换字母
_LEET_MAP: dict[str, str] = {
    "@": "a",
    "4": "a",
    "3": "e",
    "1": "i",
    "!": "i",
    "0": "o",
    "5": "s",
    "$": "s",
    "7": "t",
    "+": "t",
    "8": "b",
    "6": "g",
    "9": "g",
    "|": "l",
    "¥": "y",
}

# 需要从 leet 转换的标点（避免干扰中文语义，仅在全 ASCII 上下文生效）
_LEET_PUNCT_RE = re.compile(r"[@4!|¥]")


def _fullwidth_to_halfwidth(text: str) -> str:
    """将全角字符转换为半角字符"""
    result = []
    for ch in text:
        if ch == _FULLWIDTH_SPACE:
            result.append(" ")
        else:
            code = ord(ch)
            # 全角字符范围 FF01-FF5E
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - _FULLWIDTH_OFFSET))
            else:
                result.append(ch)
    return "".join(result)


def _trad_to_simp(text: str) -> str:
    """将繁体字替换为简体字（基础映射）"""
    return "".join(_TRAD_TO_SIMP.get(ch, ch) for ch in text)


def _replace_homophones(text: str) -> str:
    """替换谐音字为标准形式，便于规则匹配"""
    for variant, standard in _HOMOPHONES.items():
        if variant != standard:  # 防止自引用条目导致无限循环
            text = text.replace(variant, standard)
    return text


def _recompose_split_chars(text: str) -> str:
    """将拆字写法还原为完整汉字"""
    for split, whole in _SPLIT_CHAR_MAP.items():
        text = text.replace(split, whole)
    return text


def _remove_separators(text: str) -> str:
    """
    去除字符之间的分隔符（空格、下划线、点、星号等），
    这类手段常被用于规避关键词检测，如 "操 你 妈"。
    循环处理直到文本不再变化，处理多空格间隔情况。
    """
    _space_re = re.compile(
        r"([\u4e00-\u9fff\u3400-\u4dbf])\s+([\u4e00-\u9fff\u3400-\u4dbf])"
    )
    _punct_re = re.compile(
        r"([\u4e00-\u9fff\u3400-\u4dbf])[_\-\.·•*＊×x×]([\u4e00-\u9fff\u3400-\u4dbf])"
    )
    # 循环替换，直至文本稳定（处理 "你 是 个 傻 逼" 这类多个空格的情况）
    for _ in range(20):  # 最多迭代 20 次，防止无限循环
        new_text = _space_re.sub(r"\1\2", text)
        new_text = _punct_re.sub(r"\1\2", new_text)
        if new_text == text:
            break
        text = new_text
    return text


def _apply_leet(text: str) -> str:
    """
    将 leet-speak 风格的替换还原（仅处理含字母的 ASCII 混合片段，
    避免将纯数字文本（如电话号码、价格）错误转换）
    """
    def replace_ascii_segment(m: re.Match) -> str:
        seg = m.group(0)
        for leet, real in _LEET_MAP.items():
            seg = seg.replace(leet, real)
        return seg

    # 仅匹配包含至少一个字母的 ASCII 片段（必须含字母才做 leet 转换）
    # 这样 "123" 不会被转换，但 "s3x"、"@ss" 会被处理
    return re.sub(r"(?=[A-Za-z@!$|¥])[A-Za-z0-9@!$|¥4150789+]{2,}", replace_ascii_segment, text)


def _normalize_repeated(text: str) -> str:
    """
    压缩重复字符：将 3 个以上相同字符压缩为 2 个。
    如 "哈哈哈哈哈" → "哈哈"，"aaaa" → "aa"
    保留 2 个重复是因为部分词汇本身含双字（如"哈哈"、"嘿嘿"）
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize(text: str) -> str:
    """
    规范化输入文本，处理各类绕过手段，返回规范化后的文本。

    处理流程：
    1. Unicode NFKC 标准化（处理合字、变体等）
    2. 全角转半角
    3. 繁体转简体（基础映射）
    4. 谐音字替换
    5. 拆字重组
    6. 去除字符间分隔符
    7. leet-speak 替换
    8. 压缩重复字符
    """
    if not text:
        return text

    # 步骤 1：Unicode NFKC 规范化
    text = unicodedata.normalize("NFKC", text)

    # 步骤 2：全角转半角
    text = _fullwidth_to_halfwidth(text)

    # 步骤 3：繁体字转简体字
    text = _trad_to_simp(text)

    # 步骤 4：谐音字替换
    text = _replace_homophones(text)

    # 步骤 5：拆字重组
    text = _recompose_split_chars(text)

    # 步骤 6：去除字符间分隔符
    text = _remove_separators(text)

    # 步骤 7：leet-speak 替换
    text = _apply_leet(text)

    # 步骤 8：压缩重复字符
    text = _normalize_repeated(text)

    return text


def normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    """
    规范化文本，同时返回规范化字符到原始字符的位置映射。

    返回：
    - normalized: 规范化后的字符串
    - mapping: 长度与 normalized 相同的列表，mapping[i] 表示
               normalized[i] 对应原始文本中的字符索引（取起始位置）

    注意：由于规范化可能改变字符数（如去除空格、拆字合并等），
    映射是尽力而为的近似值。多字符合并为一时，取首字符位置。
    """
    if not text:
        return text, []

    original = text

    # ── 阶段一：逐字符转换（全角→半角、繁→简、谐音），保留对应关系 ──
    chars: list[str] = []
    positions: list[int] = []

    for i, ch in enumerate(original):
        # 全角转半角
        if ch == _FULLWIDTH_SPACE:
            new_ch = " "
        else:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                new_ch = chr(code - _FULLWIDTH_OFFSET)
            else:
                new_ch = ch

        # Unicode NFKC
        new_ch = unicodedata.normalize("NFKC", new_ch)

        # 繁体转简体
        new_ch = _TRAD_TO_SIMP.get(new_ch, new_ch)

        # 每个原始字符产生一个（或多个，NFKC 有时展开）规范化字符
        for c in new_ch:
            chars.append(c)
            positions.append(i)

    intermediate = "".join(chars)

    # ── 阶段二：谐音替换（可能改变字符数，用近似映射） ──
    # 对 intermediate 做替换，同步调整 positions
    for variant, standard in _HOMOPHONES.items():
        if variant == standard:
            continue  # 安全保障：跳过自引用条目，防止无限循环
        while variant in intermediate:
            idx = intermediate.find(variant)
            orig_pos = positions[idx]  # 取命中片段的首字符原始位置
            # 替换字符串
            intermediate = intermediate[:idx] + standard + intermediate[idx + len(variant):]
            # 重建 positions：替换片段全部映射到 orig_pos
            new_positions = (
                positions[:idx]
                + [orig_pos] * len(standard)
                + positions[idx + len(variant):]
            )
            positions = new_positions

    # ── 阶段三：拆字重组 ──
    for split, whole in _SPLIT_CHAR_MAP.items():
        while split in intermediate:
            idx = intermediate.find(split)
            orig_pos = positions[idx]
            intermediate = intermediate[:idx] + whole + intermediate[idx + len(split):]
            new_positions = (
                positions[:idx]
                + [orig_pos] * len(whole)
                + positions[idx + len(split):]
            )
            positions = new_positions

    # ── 阶段四：去除分隔符（删除字符，同步删除 positions） ──
    def remove_seps(s: str, pos: list[int]) -> tuple[str, list[int]]:
        """
        多次迭代去除中文字符间的分隔符，直至稳定。
        每次迭代找出要删除的索引，循环直到没有新删除。
        """
        _sep_re = re.compile(
            r"([\u4e00-\u9fff\u3400-\u4dbf])([\s_\-\.·•*＊×x×]+)([\u4e00-\u9fff\u3400-\u4dbf])"
        )
        chars = list(s)
        for _ in range(20):  # 最多迭代 20 次
            joined = "".join(chars)
            to_delete: set[int] = set()
            for m in _sep_re.finditer(joined):
                for k in range(m.start(2), m.end(2)):
                    to_delete.add(k)
            if not to_delete:
                break
            chars = [c for k, c in enumerate(chars) if k not in to_delete]
            pos = [p for k, p in enumerate(pos) if k not in to_delete]
        return "".join(chars), pos

    intermediate, positions = remove_seps(intermediate, positions)

    # ── 阶段五：leet-speak（逐字符替换，不改变字符数） ──
    leet_chars: list[str] = []
    for ch, p in zip(intermediate, positions):
        replacement = _LEET_MAP.get(ch, ch)
        # leet 替换只改变单个字符
        leet_chars.append(replacement)
    intermediate = "".join(leet_chars)

    # ── 阶段六：压缩重复字符 ──
    compressed_chars: list[str] = []
    compressed_pos: list[int] = []
    i = 0
    while i < len(intermediate):
        ch = intermediate[i]
        compressed_chars.append(ch)
        compressed_pos.append(positions[i])
        # 跳过后续重复（超过 2 个）
        count = 1
        while i + count < len(intermediate) and intermediate[i + count] == ch:
            count += 1
        if count > 2:
            # 保留 2 个
            compressed_chars.append(ch)
            compressed_pos.append(positions[i + 1] if i + 1 < len(positions) else positions[i])
            i += count
        else:
            i += 1

    return "".join(compressed_chars), compressed_pos
