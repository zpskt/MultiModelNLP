from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline('rex-uninlu', model='iic/nlp_deberta_rex-uninlu_chinese-base', model_revision='v1.2.1')

# 命名实体识别 {实体类型: None}
semantic_cls(
    input='1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。',
    schema={
        '人物': None,
        '地理位置': None,
        '组织机构': None
    }
)
# 关系抽取 {主语实体类型: {关系(宾语实体类型): None}}
semantic_cls(
  input='1987年首播的央视版《红楼梦》是中央电视台和中国电视剧制作中心根据中国古典文学名著《红楼梦》摄制的一部古装连续剧',
    schema={
        '组织机构': {
            '注册资本(数字)': None,
            '创始人(人物)': None,
            '董事长(人物)': None,
            '总部地点(地理位置)': None,
            '代言人(人物)': None,
            '成立日期(时间)': None,
            '占地面积(数字)': None,
            '简称(组织机构)': None
        }
    }
)

# 事件抽取 {事件类型（事件触发词）: {参数类型: None}}
semantic_cls(
  input='7月28日，天津泰达在德比战中以0-1负于天津天海。',
    schema={
        '胜负(事件触发词)': {
            '时间': None,
            '败者': None,
            '胜者': None,
            '赛事名称': None
        }
    }
)

# 属性情感抽取 {属性词: {情感词: None}}
semantic_cls(
  input='很满意，音质很好，发货速度快，值得购买',
    schema={
        '属性词': {
            '情感词': None,
        }
    }
)

# 允许属性词缺省，#表示缺省
semantic_cls(
  input='#很满意，音质很好，发货速度快，值得购买',
    schema={
        '属性词': {
            '情感词': None,
        }
    }
)

# 支持情感分类
semantic_cls(
  input='很满意，音质很好，发货速度快，值得购买',
    schema={
        '属性词': {
            "正向情感(情感词)": None,
            "负向情感(情感词)": None,
            "中性情感(情感词)": None
        }
    }
)

# 指代消解，正文前添加[CLASSIFY]，schema按照“自行设计的prompt+候选标签”的形式构造
semantic_cls(
  input='[CLASSIFY]因为周围的山水，早已是一派浑莽无际的绿色了。任何事物（候选词）一旦达到某种限度，你就不能再给它(代词)增加什么了。',
    schema={
        '下面的句子中，代词“它”指代的是“事物”吗？是的': None, "下面的句子中，代词“它”指代的是“事物”吗？不是": None,
        }
)

# 情感分类，正文前添加[CLASSIFY]，schema列举期望抽取的候选“情感倾向标签”；同时也支持情绪分类任务，换成相应情绪标签即可，e.g. "无情绪,积极,愤怒,悲伤,恐惧,惊奇"
semantic_cls(
  input='[CLASSIFY]有点看不下去了，看作者介绍就觉得挺矫情了，文字也弱了点。后来才发现 大家对这本书评价都很低。亏了。',
    schema={
        '正向情感': None, "负向情感": None
        }
)


# 单标签文本分类，正文前添加[CLASSIFY]，schema列举期望抽取的候选“文本分类标签”
semantic_cls(
  input='[CLASSIFY]学校召开2018届升学及出国深造毕业生座谈会就业指导',
    schema={
        '民生': None, '文化': None, '娱乐': None, '体育': None, '财经': None, '教育': None
        }
)

# 多标签文本分类，正文前添加[MULTICLASSIFY]，schema列举期望抽取的候选“文本分类标签”
semantic_cls(
  input='[MULTICLASSIFY]《格林童话》是德国民间故事集。由德国的雅各格林和威廉格林兄弟根据民间口述材料改写而成。其中的《灰姑娘》、《白雪公主》、《小红帽》、《青蛙王子》等童话故事，已被译成多种文字，在世界各国广为流传，成为各地收集民间故事的范例。',
    schema={
        '民间故事': None, '推理': None, '心理学': None, '历史': None, '传记': None, '外国名著': None, '文化': None, '诗歌': None, '童话': None, '艺术': None, '科幻': None, '小说': None
        }
)

# 层次分类，正文前添加[CLASSIFY]或者[MULTICLASSIFY]（多标签层次分类），schema按照标签层级构造 {层级1标签: {层级2标签: None}}
semantic_cls(
  input='[CLASSIFY]雨刷刮不干净怎么办',
    schema={
        'APP系统功能': {'蓝牙钥匙故障': None, '界面显示异常': None, '数据显示不准确': None, '远程控制故障': None},
        '电器-附件': {'电子模块问题': None, '雨刮、洗涤器故障': None, '防盗报警系统': None, '定速巡航系统': None}
    }
)

# 文本匹配，正文前添加[CLASSIFY]，待匹配段落按照“段落1：段落1文本；段落2：段落2文本”，schema按照“文本匹配prompt+候选标签”的形式构造
semantic_cls(
  input='[CLASSIFY]段落1：高分子材料与工程排名；段落2：高分子材料与工程专业的完整定义',
    schema={
        '文本匹配：相似': None, '文本匹配：不相似': None
        }
)

# 自然语言推理，正文前添加[CLASSIFY]，待匹配段落按照“段落1：段落1文本；段落2：段落2文本”，schema按照“自然语言推理prompt+候选标签”的形式构造
semantic_cls(
  input='[CLASSIFY]段落1：呃,银行听说都要扣一点这个转手费；段落2：从未有银行扣过手续费',
    schema={
        '段落2和段落1的关系是：蕴含': None, '段落2和段落1的关系是：矛盾': None, '段落2和段落1的关系是：中立': None
        }
)

# 选择类阅读理解，正文前添加[CLASSIFY]，schema按照“问题+候选选项”的形式构造
semantic_cls(
  input='[CLASSIFY]A：最近飞机票打折挺多的，你还是坐飞机去吧。B：反正又不是时间来不及，飞机再便宜我也不坐，我一听坐飞机就头晕。',
    schema={
        'B为什么不坐飞机?飞机票太贵': None, 'B为什么不坐飞机?时间来不及': None, 'B为什么不坐飞机?坐飞机头晕': None, 'B为什么不坐飞机?飞机票太便宜': None,
        }
)

# 抽取类阅读理解
semantic_cls(
  input='大莱龙铁路位于山东省北部环渤海地区，西起位于益羊铁路的潍坊大家洼车站，向东经海化、寿光、寒亭、昌邑、平度、莱州、招远、终到龙口，连接山东半岛羊角沟、潍坊、莱州、龙口四个港口，全长175公里，工程建设概算总投资11.42亿元。铁路西与德大铁路、黄大铁路在大家洼站接轨，东与龙烟铁路相连。大莱龙铁路于1997年11月批复立项，2002年12月28日全线铺通，2005年6月建成试运营，是横贯山东省北部的铁路干线德龙烟铁路的重要组成部分，构成山东省北部沿海通道，并成为环渤海铁路网的南部干线。铁路沿线设有大家洼站、寒亭站、昌邑北站、海天站、平度北站、沙河站、莱州站、朱桥站、招远站、龙口西站、龙口北站、龙口港站。',
    schema={
        '大莱龙铁路位于哪里？': None
        }
)