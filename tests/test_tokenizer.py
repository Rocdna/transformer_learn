from data.tokenizer import CharTokenizer


# ==========================================
# 准备一些测试文本
# ==========================================

texts = [
    "你好",
    "我喜欢人工智能",
    "Hello",
    "I love AI"
]


# ==========================================
# 创建 Tokenizer
# ==========================================

tokenizer = CharTokenizer(texts)


# ==========================================
# 查看词表
# ==========================================

print("词表大小：")
print(tokenizer.vocab_size)

print()


print("词表：")
print(tokenizer.vocab)

print()


# ==========================================
# 测试 encode
# ==========================================

text = "你好"

ids = tokenizer.encode(
    text,
    add_bos=True,
    add_eos=True
)

print("原始文本：")
print(text)

print()

print("Token IDs：")
print(ids)

print()


# ==========================================
# 测试 decode
# ==========================================

decoded = tokenizer.decode(ids)

print("解码结果：")
print(decoded)