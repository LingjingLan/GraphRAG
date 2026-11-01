# Multi-method GraphRAG experiments

Included methods and baselines: HippoRAG, LightRAG, HyperGraphRAG, dual-channel retrieval, and triple-based QA. The scripts build/extract graph representations, run question answering over the benchmark, and calculate retrieval/generation metrics.

This module intentionally excludes report PDFs, full corpora, vector indexes, model caches, generated answers, and experiment outputs. Provide those locally before running an experiment.

## 图校正

不要按公司重抽索引，也不要整库重跑 GraphRAG。已经改过的公司先改正文（`Datasets/Corpus/esg_ocr.json` 和对应的 `ESG_KG/output/ocr_md/*.context.md`），再只重连对不上校正后语料的三元组和边。

整个语料库校正完成后，再对所有方法一次性处理：只把连错的三元组或边改成和校正后的语料一致。不整家重跑，不整库重跑。

HippoRAG（gpt-4o-mini、gpt-5.1、gpt-5.4）和 HippoRAG1 的 OpenIE 已经按当前 `esg_ocr.json` 补上语料里有、索引里没有的句子，并改了 `graph.pickle` 里的边。例如 ALS 第 69 页不再把 Brisbane 和 Canberra 连在一起，Brisbane 连到 219 kWp。各方法的 `by_report` 三元组文件写入了来源为 `corpus_reconnect` 的新三元组。没有重算 embedding，也没有重跑 LightRAG、HyperGraphRAG 的内部图。

下面是目前待改的错连。

### MGL-2024

Statements 已按 PDF 对齐金额列（Notes 不再当成数值列）。错连是金额列错位。抽查时四列都要是 CONSOLIDATED 2024 / 2023、COMPANY 2024 / 2023：

- 第 1 页 Effective interest rate method：15,751 / 9,751 / 1,650 / 1,217。不能再出现 CONSOLIDATED = 2。
- 第 1 页 Employment expenses 的 COMPANY 2024 是 (4)。
- 第 6 页 Employment expenses（现金流量表）的 CONSOLIDATED 2023 是 (6,532)。
- 第 6 页购入库存股的 COMPANY 2024 是 (1,028)。
- 第 6 页经营活动现金净额的 COMPANY 2024 是 6,051。

### ALS-2024

第 69 页已安装太阳能容量表已按原表改好。错连是容量和地点被配错；国家汇总表（Australia 15 套、合计 1,569.7 kWp 等）不用改。抽查：

- Brisbane, Australia 的已安装容量包含 219 kWp，另有 99.9 kWp。不能再留下 “Brisbane = Canberra”。
- Singapore 的已安装容量是 167.4 kWp，不是 87.8 kWp。
- Rayong, Thailand 是 189 kWp，不是 99.9 kWp。
- Canberra, Australia 自己是 99.9 kWp。

### Accenture-2024

对照 `ESG_report/Services/Accenture_2024.pdf`。错连是包容与多元化表的列：2023 被标成目标公布年，Column 2 / Column 4 / Column 5 把目标、当年和基线混在一起，员工、高管、董事没有分开。`esg_ocr.json` 第 7、8、19 页已按原报告改好；`ESG_KG/output/ocr_md/Accenture-2024.context.md` 已与语料对齐。`questions_2034V2.json` 里按 yiheng 备注改过的题包括 L1_0122–0125、L3_0051–0054、L3_0056–0058、L3_0063–0067、L4_0019、L4_0020、L4_0024。

抽查：

- 员工总体男性 2024 年是 52%。67% 是高管，70% 是董事。
- 美国员工 Hispanic American and Latinx：2024 年 10.4%，目标公布时 9.5%，2025 年目标 13.0%。5.0% 是美国董事 2024 年，不是员工总体。
- 南非员工 Coloured：2024 年 10.5%，2020 年基线 6.0%，2025 年目标 10.0%。
- 第 7 页员工女性：2025 年目标 50%，2024 和 2023 都是 48%。不能再出现 Column 2 = 50%。

### DHL-2024

第 16 页双重重要性表已改回 Topic / Perspective / Metrics。原先把 Topic 写进 Perspective，再把 Perspective 和 Metrics 顺次错一列。

### SHL-2024

第 18 页重要性议题表已把对照议题和“最受影响的利益相关方”分开。Waste management、Employee health and safety 等是对照议题，Employees 等是利益相关方，不能再互相当成同一列。

### Deloitte-2024

第 20 页样本口径：N=300 是主样本，N=250 是行业加样。不能再把 N=250 写成 N=300 那一列的数值。

### dell-2024

第 83 页第三方来源和第 94 页面授或网络培训课时已按 FY22 / FY23 / FY24 对齐。供应链审计表里的 P 是 Priority Findings 的行标签，不是发现数量。

### DBS-2024

员工表：20,727 是 2024 年男性全职人数，40,593 是 2024 年全职合计。Table 6 的职级表头不再写入 Table 3。

### bmw-2024

环境指标等表里，把报告页码或汇总标记当成数值的句子已改。核能占比那一格的 PDF 文本层串进了总用水量，对应题目已从 V2 删除，语料不写入这个百分比。总用水量 5,813,743 m³ 仍是水表上的数。

### jpmc-2024

碳强度、融资排放和批发信贷表里写成 Column 的句子已改成报告中的列名。

### APA-2024

第 82 页联合国可持续发展目标索引：Location in Report 列已改回主题和页码，不再把目标原文当成位置。

### BASF-2024

金额表里 Column 2 被写成单位表头 € million 的句子已改回数值列。

### bp-2024

INREM 项目原文是 aims to enable 60 个用水安全社区，2024 年实际完成的是 30 个村庄的处理系统。目录页码 37 不再当成章节属性值。

### Microsoft-2024

封面写的是 FY23。目录和导航里的页码不再当成 Progress through partnership 的指标值。

### Ramsay-2024

环境记分卡已按主体（Australia、UK-Acute、France）和指标分开：麻醉气体变化、地氟烷变化、能源、废弃物、市政用水各自的 FY22 / FY23 / FY24 和强度列。不能再把总吨数写成强度，或把法国分项写成整个 Ramsay Santé。

### Snam-2024

生物甲烷 2024 年预算是 20 Mscm。Scope 1 和 2 的 2024 年实际降幅是 -29%，2035 年目标是 -65%。生物多样性标题下的 Column 2 不再拿走这个减排目标。重要性矩阵里 Own workforce 的财务重要性，以及 Affected communities 只有影响重要性，已按原表改回。

### Starbucks-2024

咖啡供应链不合规整改：FY24 列是 84%，FY23 列是 34%。取水数据属于 Water Withdrawals，不再放在 Dairy Methane 下面。

`questions_2034V2.json` 现为 2009 道。原先按备注删掉的 219 道里，194 道已改成报告里能够唯一回答的问法并放回；仍删除 25 道。这 25 道在报告里没有可改成的事实：P&G 雷达图读数、Qantas 页眉被当成风险表字段、Ramsay 没有指标名的 100 和 31%、BMW 核能占比的文本层串行、JPMorgan 排放范围没有单一口径、DBS 报告中不存在的奖项、Snam 页眉导航链、bp 目录页码，以及 Lundin 无法唯一确定的实质性议题。Apple、QANTAS、NextEra Energy、Coles、Hearing-Australia、P&G、enel、lundin-mining、Woolworths Group、McDonalds 只改了题目或删了题，语料没有对上的错句，这些公司不需要为这一轮重连。

错连所在的产物（届时只改这些文件里对不上语料的连接，路径里的公司名换成上面列出的公司，含 MGL-2024、ALS-2024、Accenture-2024）：

| 方法 | 索引 | 三元组 |
|---|---|---|
| HippoRAG2 | `ESG_KG/output/indexes/hipporag/{gpt-4o-mini,gpt-5.1,gpt-5.4}/<company>/` | `ESG_KG/output/triples/hipporag/{model}/by_report/<company>_triples.json` |
| HippoRAG1 | `ESG_KG/output/indexes/hipporag1/gpt-5.1/<company>/` | 该方法对应的 by_report 文件 |
| LightRAG | `ESG_KG/output/triples/LightRAG/gpt-5.1/indexes/<company>/` | `.../by_report/<company>_triples.json` |
| HyperGraphRAG | `ESG_KG/output/indexes/hypergraphrag/gpt-5.1/<company>/` | `ESG_KG/output/triples/HyperGraphRAG/gpt-5.1/` 下该公司条目 |
| dual-channel | `ESG_KG/output/indexes/dualchannel/openai/<company>/` 和 `.../openai_mdblocks/<company>/` | 无单独三元组；只改与错连事实对应的缓存块 |
