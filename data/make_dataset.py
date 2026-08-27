# -*- coding: utf-8 -*-
"""
生成一套「干净、句级对齐」的中英平行语料 demo。
- 手写句对（PAIRS）：日常会话，中英严格一一对应
- 模板扩展（EXTRA_*）：用「已人工校对过的」中文/英文片段拼出大量严格对齐句对，
  重点系统覆盖“动词 × 宾语 / 主语 × 时态”等组合，帮助模型学会组合翻译
- train / validation / test 从同一分布随机切分 (8:1:1)
- 存储成 conversations 格式，train.py / predict.py 只需换路径
运行： python data/make_dataset.py
"""
import random

from datasets import Dataset, DatasetDict

# 手工校对、逐句对齐的中英句对（中文, 英文）
PAIRS = [
    # ---- 问候与礼貌 ----
    ("你好", "Hello."),
    ("早上好", "Good morning."),
    ("晚安", "Good night."),
    ("谢谢", "Thank you."),
    ("非常感谢", "Thank you very much."),
    ("不客气", "You are welcome."),
    ("对不起", "I am sorry."),
    ("没关系", "It is all right."),
    ("再见", "Goodbye."),
    ("我很好", "I am fine."),
    ("你呢", "How about you?"),
    ("请再说一遍", "Please say it again."),
    ("请问你叫什么名字", "What is your name?"),
    ("我叫小刘", "My name is Xiao Liu."),
    ("很高兴认识你", "Nice to meet you."),
    ("请坐", "Please sit down."),
    ("请进", "Please come in."),
    ("欢迎来我家", "Welcome to my home."),
    ("祝你生日快乐", "Happy birthday to you."),
    ("祝你成功", "I wish you success."),
    # ---- 饮食 ----
    ("我饿了", "I am hungry."),
    ("我渴了", "I am thirsty."),
    ("我想喝水", "I want to drink water."),
    ("我想喝茶", "I want to drink tea."),
    ("我想喝咖啡", "I want to drink coffee."),
    ("我想吃米饭", "I want to eat rice."),
    ("我想吃面条", "I want to eat noodles."),
    ("我喜欢吃苹果", "I like eating apples."),
    ("我喜欢吃香蕉", "I like eating bananas."),
    ("我不吃辣", "I do not eat spicy food."),
    ("这顿饭很好吃", "This meal is delicious."),
    ("太甜了", "It is too sweet."),
    ("太咸了", "It is too salty."),
    ("请给我一杯水", "Please give me a glass of water."),
    ("请给我一些盐", "Please give me some salt."),
    ("我们吃早饭吧", "Let us eat breakfast."),
    ("我们吃午饭吧", "Let us eat lunch."),
    ("我们吃晚饭吧", "Let us eat dinner."),
    ("我不喜欢喝牛奶", "I do not like drinking milk."),
    # ---- 天气 ----
    ("今天天气很好", "The weather is nice today."),
    ("今天很热", "It is hot today."),
    ("今天很冷", "It is cold today."),
    ("下雨了", "It is raining."),
    ("下雪了", "It is snowing."),
    ("起风了", "It is windy."),
    ("明天会下雨吗", "Will it rain tomorrow?"),
    ("今天天空很蓝", "The sky is blue today."),
    ("昨天下了大雨", "It rained heavily yesterday."),
    # ---- 购物与价格 ----
    ("这个多少钱", "How much is this?"),
    ("这件衣服多少钱", "How much is this dress?"),
    ("太贵了", "It is too expensive."),
    ("便宜一点吧", "Can it be a little cheaper?"),
    ("我买这个", "I will buy this one."),
    ("你们接受信用卡吗", "Do you accept credit cards?"),
    ("我想买一双鞋", "I want to buy a pair of shoes."),
    ("这个尺寸合适", "This size fits."),
    ("这个太小了", "This is too small."),
    ("这个太大了", "This is too big."),
    ("商店几点关门", "What time does the shop close?"),
    ("我去超市买东西", "I go to the supermarket to buy things."),
    # ---- 时间与作息 ----
    ("今天是几号", "What is the date today?"),
    ("今天是星期一", "Today is Monday."),
    ("今天是星期日", "Today is Sunday."),
    ("我七点起床", "I get up at seven."),
    ("我八点上班", "I go to work at eight."),
    ("我晚上十一点睡觉", "I go to bed at eleven at night."),
    ("我每天散步", "I take a walk every day."),
    ("我每天早上跑步", "I run every morning."),
    ("我周末休息", "I rest on weekends."),
    ("明天是星期五", "Tomorrow is Friday."),
    ("现在是中午", "It is noon now."),
    ("现在是晚上", "It is evening now."),
    ("我起床很早", "I get up very early."),
    ("我睡觉很晚", "I go to bed very late."),
    # ---- 家庭与人 ----
    ("我有一家人", "I have a family."),
    ("我爸爸是老师", "My father is a teacher."),
    ("我妈妈是医生", "My mother is a doctor."),
    ("我有一个弟弟", "I have a younger brother."),
    ("我有一个姐姐", "I have an elder sister."),
    ("我的哥哥很高", "My elder brother is tall."),
    ("我的妹妹很小", "My younger sister is young."),
    ("我爷爷八十岁了", "My grandfather is eighty years old."),
    ("我奶奶很健康", "My grandmother is healthy."),
    ("我的家人很爱我", "My family loves me very much."),
    ("我是中国人", "I am Chinese."),
    ("他是美国人", "He is American."),
    ("她是我的朋友", "She is my friend."),
    ("他是我的老师", "He is my teacher."),
    # ---- 旅行与地点 ----
    ("我要去北京", "I will go to Beijing."),
    ("我坐火车去上海", "I take the train to Shanghai."),
    ("机场在哪里", "Where is the airport?"),
    ("火车站怎么走", "How do I get to the railway station?"),
    ("我要一张去广州的票", "I want a ticket to Guangzhou."),
    ("我想租一辆车", "I want to rent a car."),
    ("这里离市中心远吗", "Is it far from the city center?"),
    ("我的房间在二楼", "My room is on the second floor."),
    ("公共汽车站很近", "The bus stop is near."),
    ("出租车在这里等", "The taxi is waiting here."),
    ("我在酒店门口等你", "I will wait for you at the hotel entrance."),
    ("这次旅行很愉快", "The trip was pleasant."),
    # ---- 喜好与能力 ----
    ("我会说中文", "I can speak Chinese."),
    ("我会说英文", "I can speak English."),
    ("我不会说日语", "I cannot speak Japanese."),
    ("我喜欢唱歌", "I like singing."),
    ("我喜欢跳舞", "I like dancing."),
    ("我喜欢读书", "I like reading."),
    ("我喜欢音乐", "I like music."),
    ("我喜欢看电影", "I like watching movies."),
    ("我喜欢听音乐", "I like listening to music."),
    ("我爱我的家人", "I love my family."),
    ("我爱我的祖国", "I love my country."),
    ("我想学英语", "I want to learn English."),
    ("我想学中文", "I want to learn Chinese."),
    ("我会游泳", "I can swim."),
    ("我会做饭", "I can cook."),
    ("他会开车", "He can drive."),
    ("她会弹钢琴", "She can play the piano."),
    # ---- 评价与描述 ----
    ("这本书很好看", "This book is good to read."),
    ("这部电影很有趣", "This movie is very interesting."),
    ("这首歌很好听", "This song sounds wonderful."),
    ("这里的风景很美", "The scenery here is beautiful."),
    ("这座城市很漂亮", "This city is beautiful."),
    ("这个地方很安静", "This place is quiet."),
    ("这个房间很干净", "This room is clean."),
    ("这个房间很乱", "This room is messy."),
    ("我觉得很累", "I feel tired."),
    ("我觉得很开心", "I feel happy."),
    ("我很忙", "I am busy."),
    ("我很兴奋", "I am excited."),
    ("汤是热的", "The soup is hot."),
    ("水是冷的", "The water is cold."),
    ("这个包很轻", "This bag is light."),
    ("这个箱子很重", "This box is heavy."),
    ("我不喜欢这个颜色", "I do not like this color."),
    # ---- 正在做的事 / 祈使 ----
    ("我在看书", "I am reading a book."),
    ("她在写信", "She is writing a letter."),
    ("他在画画", "He is drawing a picture."),
    ("我们在看电视", "We are watching TV."),
    ("他们在踢足球", "They are playing football."),
    ("我正在做饭", "I am cooking now."),
    ("她在洗衣服", "She is washing clothes."),
    ("他每天上班", "He goes to work every day."),
    ("我们一起去爬山吧", "Let us go hiking together."),
    ("我们去游泳吧", "Let us go swimming."),
    ("我们去公园散步吧", "Let us take a walk in the park."),
    ("我等你", "I will wait for you."),
    ("他在睡觉", "He is sleeping."),
    ("她笑了", "She smiled."),
    ("我找到了我的钥匙", "I found my keys."),
    ("我忘了带手机", "I forgot to bring my phone."),
    ("他回家了", "He went home."),
    ("我们见面吧", "Let us meet."),
    ("请开门", "Please open the door."),
    ("请关门", "Please close the door."),
    ("请快一点", "Please be quick."),
    ("小心一点", "Be careful."),
    ("慢慢来", "Take your time."),
    ("别担心", "Do not worry."),
    ("别害怕", "Do not be afraid."),
    ("加油", "Keep going."),
    # ---- 疑问 / 寒暄 ----
    ("你好吗", "How are you?"),
    ("你忙吗", "Are you busy?"),
    ("你饿吗", "Are you hungry?"),
    ("你累吗", "Are you tired?"),
    ("你喜欢什么", "What do you like?"),
    ("你想吃什么", "What do you want to eat?"),
    ("你想去哪里", "Where do you want to go?"),
    ("你在做什么", "What are you doing?"),
    ("你在哪里", "Where are you?"),
    ("他是谁", "Who is he?"),
    ("这是什么", "What is this?"),
    ("你在想什么", "What are you thinking about?"),
    ("你有弟弟吗", "Do you have a younger brother?"),
    ("你会游泳吗", "Can you swim?"),
    ("明天你休息吗", "Will you rest tomorrow?"),
    ("我们什么时候出发", "When do we set off?"),
    ("我们怎么去", "How do we get there?"),
    ("你吃饭了吗", "Have you eaten?"),
    ("你去过上海吗", "Have you been to Shanghai?"),
    ("你今年多大了", "How old are you?"),
    # ---- 看法与表达 ----
    ("你说得对", "You are right."),
    ("我也这么觉得", "I think so too."),
    ("我不同意", "I do not agree."),
    ("我明白了", "I understand."),
    ("我不明白", "I do not understand."),
    ("没问题", "No problem."),
    ("不好意思打扰了", "Sorry to bother you."),
    ("请多关照", "I look forward to working with you."),
    # ---- 工作 / 学习 / 生活 ----
    ("我住在北京", "I live in Beijing."),
    ("我在学英语", "I am learning English."),
    ("她在一家公司工作", "She works at a company."),
    ("他是一名学生", "He is a student."),
    ("我是一名工程师", "I am an engineer."),
    ("这份工作很累", "This job is tiring."),
    ("我很喜欢这份工作", "I like this job very much."),
    ("我每天坐地铁上班", "I take the subway to work every day."),
    ("我的办公室在楼上", "My office is upstairs."),
    ("下班后我去健身房", "After work I go to the gym."),
    ("我通常六点下班", "I usually finish work at six."),
    ("明天我要出差", "I will go on a business trip tomorrow."),
    ("我下个月休假", "I will take a vacation next month."),
    ("假期你去哪", "Where do you go during the holiday?"),
    ("我打算去云南旅行", "I plan to travel to Yunnan."),
    ("那里天气很好", "The weather there is nice."),
    ("我拍了很多照片", "I took many photos."),
    ("这张照片很美", "This photo is beautiful."),
    ("我打电话给你", "I will call you."),
    ("我的手机没电了", "My phone battery is out."),
    ("我经常上网", "I often go online."),
    ("我在看新闻", "I am reading the news."),
    ("这篇文章很有意思", "This article is interesting."),
    ("我的电脑坏了", "My computer is broken."),
    ("程序运行得很快", "The program runs fast."),
    # ---- 健康 / 时间 ----
    ("我经常运动", "I often exercise."),
    ("我每天早上喝牛奶", "I drink milk every morning."),
    ("吃太多不好", "Eating too much is not good."),
    ("你该休息一下", "You should rest for a while."),
    ("我去看医生", "I will see a doctor."),
    ("我感冒了", "I have a cold."),
    ("我头疼", "I have a headache."),
    ("我发烧了", "I have a fever."),
    ("请多喝水", "Please drink more water."),
    ("好好休息", "Get plenty of rest."),
    ("祝你早日康复", "Hope you get well soon."),
    ("不要太过疲劳", "Do not get too tired."),
    ("我昨天去了图书馆", "I went to the library yesterday."),
    ("他上星期去了上海", "He went to Shanghai last week."),
    ("我明天要去上班", "I will go to work tomorrow."),
    ("我中午吃了面条", "I ate noodles at noon."),
    ("他刚才走了", "He left just now."),
    ("我马上回来", "I will come back soon."),
    ("等我一下", "Wait for me a moment."),
    ("他从来不迟到", "He is never late."),
    ("我总是早起", "I always get up early."),
    ("她有时候很忙", "She is busy sometimes."),
]


# ============================================================
# 模板扩展：用「已人工校对过的」中/英片段组合出大量严格对齐句对
# 目标是让“动词×宾语 / 主语×时态”等组合被系统覆盖，教会模型组合翻译
# ============================================================

def _want_pairs():
    """我想 / 我要 + 喝·吃·买·看·听·学·用·带  →  I want to + do ..."""
    out = []
    vase = {
        # 动词: {宾语: 英文片段}
        "喝": {
            "水": "drink water", "茶": "drink tea", "咖啡": "drink coffee",
            "牛奶": "drink milk", "果汁": "drink juice", "可乐": "drink cola",
            "热水": "drink hot water", "汤": "drink soup",
        },
        "吃": {
            "米饭": "eat rice", "面条": "eat noodles", "面包": "eat bread",
            "苹果": "eat apples", "香蕉": "eat bananas", "鸡蛋": "eat eggs",
            "牛肉": "eat beef", "鱼": "eat fish", "蔬菜": "eat vegetables",
            "蛋糕": "eat cake", "饺子": "eat dumplings", "水果": "eat fruit",
        },
        "买": {
            "一本书": "buy a book", "一支笔": "buy a pen", "一台电脑": "buy a computer",
            "一部手机": "buy a phone", "一双鞋": "buy a pair of shoes",
            "一些水果": "buy some fruit", "一束花": "buy a bunch of flowers",
            "一个面包": "buy some bread", "一些牛奶": "buy some milk",
        },
        "看": {
            "一本书": "read a book", "一部电影": "watch a movie",
            "电视": "watch TV", "新闻": "watch the news",
        },
        "听": {
            "音乐": "listen to music", "新闻": "listen to the news",
            "广播": "listen to the radio",
        },
        "学": {
            "英语": "learn English", "中文": "learn Chinese", "日语": "learn Japanese",
            "法语": "learn French", "数学": "learn math", "画画": "learn painting",
        },
        "用": {
            "电脑": "use the computer", "手机": "use my phone", "词典": "use a dictionary",
        },
        "带": {
            "一个背包": "bring a backpack", "一把伞": "bring an umbrella",
            "一本书": "bring a book",
        },
    }
    for v, objs in vase.items():
        for obj, en in objs.items():
            out.append((f"我想{v}{obj}", f"I want to {en}."))
            out.append((f"我要{v}{obj}", f"I want to {en}."))
    return out


def _like_pairs():
    """我喜欢 / 她喜欢 / 他不喜欢 / 我们喜欢 + 动名词"""
    out = []
    acts = {
        "喝水": "drinking water", "喝茶": "drinking tea", "喝咖啡": "drinking coffee",
        "喝牛奶": "drinking milk", "吃苹果": "eating apples", "吃香蕉": "eating bananas",
        "吃米饭": "eating rice", "看书": "reading books", "写字": "writing",
        "唱歌": "singing", "跳舞": "dancing", "画画": "drawing", "跑步": "running",
        "游泳": "swimming", "爬山": "hiking", "摄影": "taking photos",
        "听音乐": "listening to music", "看电影": "watching movies", "读书": "reading",
        "学英语": "learning English", "做饭": "cooking",
    }
    for act, en in acts.items():
        out.append((f"我喜欢{act}", f"I like {en}."))
        out.append((f"他喜欢{act}", f"He likes {en}."))
        out.append((f"她喜欢{act}", f"She likes {en}."))
        out.append((f"我们喜欢{act}", f"We like {en}."))
        out.append((f"他不喜欢{act}", f"He does not like {en}."))
        out.append((f"她不喜欢{act}", f"She does not like {en}."))
        out.append((f"我不喜欢{act}", f"I do not like {en}."))
    return out


def _progressive_pairs():
    """我 / 他 / 她在 + 动词 →  I am / He is / She is + doing"""
    out = []
    acts = {
        "看书": "reading a book", "写字": "writing", "画画": "drawing a picture",
        "听音乐": "listening to music", "看电视": "watching TV",
        "看电影": "watching a movie", "做饭": "cooking", "洗衣服": "washing clothes",
        "跑步": "running", "游泳": "swimming", "睡觉": "sleeping", "上班": "working",
        "读报": "reading the newspaper", "上网": "surfing the Internet",
        "学英语": "learning English", "唱歌": "singing",
    }
    for pz, pe in [("我", "I am"), ("他", "He is"), ("她", "She is")]:
        for act, en in acts.items():
            out.append((f"{pz}在{act}", f"{pe} {en}."))
    return out


def _time_pairs():
    """我 + 点钟 + 起床/上班/上学/吃饭/睡觉 →  I do ... at time"""
    out = []
    times = {"六": "six", "七": "seven", "八": "eight", "九": "nine",
             "十": "ten", "十一": "eleven"}
    actions = {
        "起床": "get up", "上班": "go to work", "上学": "go to school",
        "吃早饭": "eat breakfast", "吃午饭": "eat lunch", "吃晚饭": "eat dinner",
        "睡觉": "go to bed", "回家": "go home",
    }
    for tz, te in times.items():
        for az, ae in actions.items():
            out.append((f"我{tz}点{az}", f"I {ae} at {te}."))
    return out


def _everyday_pairs():
    """我每天早上/晚上 + 动词 →  I do ... every morning/evening"""
    out = []
    acts = {
        "跑步": "run", "散步": "take a walk", "喝牛奶": "drink milk",
        "喝咖啡": "drink coffee", "读报": "read the newspaper",
        "听收音机": "listen to the radio", "看新闻": "watch the news", "锻炼": "exercise",
    }
    for act, en in acts.items():
        out.append((f"我每天早上{act}", f"I {en} every morning."))
        out.append((f"我每天晚上{act}", f"I {en} every evening."))
    return out


def _weather_pairs():
    """今天/昨天 + 天气 →  It is/It was ... today/yesterday"""
    out = []
    conds = {
        "热": "hot", "冷": "cold", "凉": "cool", "晴": "sunny",
        "阴": "cloudy", "有风": "windy", "下雨": "raining", "下雪": "snowing",
    }
    for cz, ce in conds.items():
        out.append((f"今天很{cz}", f"It is {ce} today."))
        out.append((f"昨天很{cz}", f"It was {ce} yesterday."))
    # 明天
    future = {
        "下雨": "rain", "下雪": "snow", "有风": "be windy", "天晴": "be sunny",
        "很热": "be hot", "很冷": "be cold", "阴天": "be cloudy",
    }
    for cz, ce in future.items():
        out.append((f"明天会{cz}", f"It will {ce} tomorrow."))
    return out


def _demo_pairs():
    """这/那 + 是什么 / 是名词 →  What is this? / This is a ..."""
    out = [
        ("这是什么", "What is this?"),
        ("那是什么", "What is that?"),
    ]
    nouns = {
        "一本书": "a book", "一支笔": "a pen", "一张桌子": "a table",
        "一把椅子": "a chair", "一个杯子": "a cup", "一个苹果": "an apple",
        "一部手机": "a phone", "一台电脑": "a computer", "一把钥匙": "a key",
    }
    for dz, de in [("这", "This"), ("那", "That")]:
        for nz, ne in nouns.items():
            out.append((f"{dz}是{nz}", f"{de} is {ne}."))
    return out


def _have_pairs():
    """我有 + 量词名词 →  I have a(n) ..."""
    out = []
    items = [
        ("一个哥哥", "an elder brother"), ("一个弟弟", "a younger brother"),
        ("一个姐姐", "an elder sister"), ("一个妹妹", "a younger sister"),
        ("一个朋友", "a friend"), ("一只猫", "a cat"), ("一只狗", "a dog"),
        ("一只鸟", "a bird"), ("一辆车", "a car"), ("一辆自行车", "a bicycle"),
        ("一台电脑", "a computer"), ("一部手机", "a phone"),
        ("一个房间", "a room"), ("一个花园", "a garden"),
    ]
    for nz, ne in items:
        out.append((f"我有{nz}", f"I have {ne}."))
    return out


def _liken_pairs():
    """我喜欢 + 名词 →  I like ..."""
    out = []
    nouns = {
        "音乐": "music", "足球": "football", "篮球": "basketball", "画画": "painting",
        "摄影": "photography", "旅行": "traveling", "数学": "math",
        "英语": "English", "中文": "Chinese", "游泳": "swimming",
        "爬山": "hiking", "读书": "reading", "小狗": "dogs", "小猫": "cats",
    }
    for nz, ne in nouns.items():
        out.append((f"我喜欢{nz}", f"I like {ne}."))
    return out


def _go_pairs():
    """我要去 + 地点 →  I will go to ..."""
    out = []
    places = {
        "北京": "Beijing", "上海": "Shanghai", "广州": "Guangzhou",
        "学校": "school", "超市": "the supermarket", "商店": "the store",
        "公园": "the park", "图书馆": "the library", "健身房": "the gym",
        "机场": "the airport", "火车站": "the railway station",
        "医院": "the hospital", "银行": "the bank", "山上": "the mountain",
    }
    for pz, pe in places.items():
        out.append((f"我要去{pz}", f"I will go to {pe}."))
    return out


def _floor_pairs():
    out = []
    for fz, fe in [("一楼", "the first floor"), ("二楼", "the second floor"),
                   ("三楼", "the third floor")]:
        out.append((f"我的房间在{fz}", f"My room is on {fe}."))
    return out


def _dow_pairs():
    """今天是星期X →  Today is Monday. 等"""
    out = []
    days = [("星期一", "Monday"), ("星期二", "Tuesday"), ("星期三", "Wednesday"),
            ("星期四", "Thursday"), ("星期五", "Friday"), ("星期六", "Saturday"),
            ("星期日", "Sunday")]
    for dz, de in days:
        out.append((f"今天是{dz}", f"Today is {de}."))
    return out


def _with_friend():
    """我和朋友 + 动词 →  I ... with my friend"""
    out = []
    acts = {
        "散步": "take a walk", "爬山": "hike", "跑步": "run",
        "购物": "go shopping", "看足球赛": "watch a football match",
        "看电影": "watch a movie",
    }
    for uz, ue in acts.items():
        out.append((f"我和朋友{uz}", f"I {ue} with my friend."))
    return out


def _daytime_pairs():
    """现在是 + 时段 →  It is ... now"""
    out = []
    for tz, te in [("早上", "morning"), ("上午", "morning"), ("下午", "afternoon"),
                   ("傍晚", "evening"), ("深夜", "late at night")]:
        out.append((f"现在是{tz}", f"It is {te} now."))
    return out


# 汇总所有扩展
def build_extended_pairs():
    extra = []
    for fn in [_want_pairs, _like_pairs, _progressive_pairs, _time_pairs,
               _everyday_pairs, _weather_pairs, _demo_pairs, _have_pairs,
               _liken_pairs, _go_pairs, _floor_pairs, _dow_pairs,
               _with_friend, _daytime_pairs]:
        extra.extend(fn())

    # 手写句对优先，模板扩展去重（保留先出现的）
    seen = set()
    allpairs = []
    for p in PAIRS + extra:
        if p not in seen:
            seen.add(p)
            allpairs.append(p)
    return allpairs


def split_pairs(pairs, seed=42, tr=0.8, va=0.1):
    rng = random.Random(seed)
    pool = list(pairs)
    rng.shuffle(pool)
    n = len(pool)
    n_tr = int(n * tr)
    n_va = int(n * va)
    return pool[:n_tr], pool[n_tr:n_tr + n_va], pool[n_tr + n_va:]


def to_rows(ps):
    rows = []
    for zh, en in ps:
        rows.append({"conversations": [
            {"role": "user", "content": "translate following chinese into english -- " + zh},
            {"role": "assistant", "content": en},
        ]})
    return rows


def main():
    pairs = build_extended_pairs()
    tr, va, te = split_pairs(pairs)
    ds = DatasetDict({
        "train": Dataset.from_list(to_rows(tr)),
        "validation": Dataset.from_list(to_rows(va)),
        "test": Dataset.from_list(to_rows(te)),
    })
    out = "data/dataset_clean"
    ds.save_to_disk(out)
    print(f"生成完成: 共 {len(pairs)} 对 → 写入 {out}")
    print(f"train={len(tr)}  validation={len(va)}  test={len(te)}")


if __name__ == "__main__":
    main()