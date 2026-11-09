"""Build plain-language interpretation from completed, validated outputs only."""
import json,html,hashlib
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
checks=json.loads((P/'results/completion_checks.json').read_text())
assert checks['all_checked_shifts_match'] and checks['noisy_pairs']==253440
s=pd.read_csv(P/'results/summary.csv');n=pd.read_csv(P/'results/noise_only_summary.csv')
g=s[(s['mode']=='both')&(s.delay_samples>=0)&s.snr_db.isin([20,0,-10,-20])]
table=g.pivot(index=['noise','delay_samples'],columns='snr_db',values='mean_abs_ms').reset_index()
table['真实延迟（ms）']=table.pop('delay_samples')/32
table=table.rename(columns={'noise':'噪声类型',20.:'SNR 20 dB',0.:'SNR 0 dB',-10.:'SNR −10 dB',-20.:'SNR −20 dB'})
table=table[['噪声类型','真实延迟（ms）','SNR 20 dB','SNR 0 dB','SNR −10 dB','SNR −20 dB']]
table['噪声类型']=table['噪声类型'].map({'white':'白噪声','ar1':'有色噪声'})
table.to_csv(P/'results/中文关键结果.csv',index=False)
paragraphs=[
 '本轮要回答的是：同一段固定波形、同一个真实延迟下，增加独立噪声会怎样改变算法估计的延迟大小。没有改变真实延迟，也没有改变窗口、搜索范围或E。',
 '以下表格比较双通道独立加噪条件。所有数字是先取每个窗口延迟的绝对值，再在完整记录内平均，最后对36模板×20噪声实现取平均。它们不是单次估计，也不是海马真实延迟。'
]
for kind,label in [('white','白噪声'),('ar1','有色噪声')]:
    a=s[(s.noise==kind)&(s['mode']=='both')&(s.snr_db==-20)&(s.delay_samples>=0)]
    z=a[a.delay_samples==0].iloc[0];large=a[a.delay_samples==400].iloc[0]
    no=n[n.noise==kind].iloc[0]
    paragraphs.append(f'{label}：SNR降到−20 dB时，真实延迟为0 ms的平均绝对估计值为{z.mean_abs_ms:.3f} ms；真实延迟为12.5 ms时为{large.mean_abs_ms:.3f} ms。对应RMSE分别为{z.mean_rmse_ms:.3f}、{large.mean_rmse_ms:.3f} ms。')
    paragraphs.append(f'在这一强噪声条件下，0–12.5 ms各真实延迟条件的平均绝对估计值落在{a.mean_abs_ms.min():.3f}–{a.mean_abs_ms.max():.3f} ms；两路独立{label}、没有共同模板的对照均值为{no.mean_abs_ms:.3f} ms。这些数值只适用于当前估计器、窗口和±15.625 ms搜索范围。')
paragraphs += [
 '如何解释：低SNR会让估计越来越受噪声与有限搜索范围影响；它不等于把原来的真实延迟统一往上推。应联合查看曲线、逐次配对变化及RMSE。真实延迟较小时可能出现幅度高估，较大时也可能出现幅度低估。',
 '图中有符号均值与绝对值均值要分开理解：一批正、负的错误估计可以在有符号平均时抵消，但取绝对值后会留下较大的数值。纯噪声对照没有可定义的共同事件延迟，因此其非零绝对估计值不应被解释为真实传输时间。',
 '用于审稿回复：结果提供了对“较低SNR必然系统性产生更大延迟”这一普遍化表述的条件限定；同时也明确展示了噪声造成表观延迟的可能性。因此，它不能用来声称海马高延迟已排除SNR混杂。',
 '模板限制：固定模板包含原始记录的背景噪声，这部分也随波形一起移位，成为两路信号的共有成分；只有新增噪声是独立的。本实验验证构造条件下的敏感性，不等同于两路真实神经记录具有相同的事件波形或已知的生理SNR。',
 '汇总的Monte Carlo标准误只描述独立新增噪声带来的数值不确定性。36模板来自同一人9个试次，不能作为36名独立受试者进行生物学推广。'
]
answer='SNR与估计延迟大小：本轮结果解读\n\n'+'\n\n'.join(paragraphs[:2])+'\n\n'+table.to_string(index=False,float_format=lambda x:f'{x:.3f}')+'\n\n'+'\n\n'.join(paragraphs[2:])
(P/'结论解读.txt').write_text(answer)
page=P/'扩展实验报告.html';content=page.read_text()
section='<section id="interpretation"><h2>先看结果：低SNR不等于高延迟</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[:2])+table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[2:])+'</section>'
if '<section id="interpretation">' not in content:
    content=content.replace('<h2>一、关系曲线</h2>',section+'<h2>一、关系曲线</h2>')
page.write_text(content)
files={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.name!='output_manifest.json'}
(P/'output_manifest.json').write_text(json.dumps(files,ensure_ascii=False,indent=2))
print(answer)
