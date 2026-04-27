from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. 连接
connections.connect(host="localhost", port="19530")
print("连接成功！")

# 2. 建表
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]

collection = Collection("rag_collection", CollectionSchema(fields))
print("表创建成功！")  

# 3. 插入
collection.insert([
    [[0.1]*768],
    ["测试文本"]
])
print("数据插入成功！")

#创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print("索引创建成功！")

# 4. 查询
collection.load()

results = collection.search(
    data=[[0.1]*768],
    anns_field="embedding",
    param={"nprobe": 10},
    limit=1,
    output_fields=["text"]
)

print(results[0][0].entity.get("text"))
print("查询成功！")