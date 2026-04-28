import json
with open('data/jds.json','r',encoding='utf-8') as f:
    jds=json.load(f)
for j in jds[:10]:
    ss=j.get('structured_summary',{})
    print(f"{j['id']:2d} | {j['company']:8s} | {j['position']:12s} | 年限={ss.get('min_years')}-{ss.get('max_years')} | 学历={ss.get('min_education')} | 领域={ss.get('domain')} | 向量={j.get('vector_indexed')} | chunks={len(j.get('meta',{}).get('chunk_ids',[]))}")
