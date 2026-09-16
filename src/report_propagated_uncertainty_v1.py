"""Report all frozen uncertainty candidates and the failed primary gate."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/propagated_uncertainty_v1_2026-09-14'
REPORT = ROOT/'reports/PROPAGATED_UNCERTAINTY_V1_2026-09-14.md'
LABELS = dict(structure_texture_far='far', nostructure_texture_near_withloop='planar', sitting_xyz='sitting')
MODES = ['iid', 'isotropic_marginal', 'block_marginal', 'propagated_full']
GEOMETRIES = ['lg_baseline', 'unit_refit', 'depth_weighted', 'huber_refit', 'depth_huber_refit']


def load(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(lines, titles): lines.extend(['', '| '+' | '.join(titles)+' |', '|'+'|'.join(['---']*len(titles))+'|'])
def row(lines, values): lines.append('| '+' | '.join(str(v) for v in values)+' |')
def fmt(v, scale=1000): return f'{v*scale:+.3f}'


def main():
    groups, decision, runtime = load('summary.json'), load('decision.json'), load('runtime.json')
    predictions, evaluated = load('predictions.json'), load('evaluation.json')
    shared = load('shared_frame_cache_audit.json')
    lookup = {(g['budget'], g['geometry'], g['mode'], g['sequence']): g for g in groups}
    plook = {(r['sequence'], r['offset'], r['budget'], r['geometry'], r['mode']): r for r in predictions}
    matrix = np.array([[lookup[b, 'depth_weighted', m, s]['path_gain_vs_same_geometry_iid_m']*1000 for m in MODES[1:]] for b in [16, 32] for s in LABELS])
    labels = [f'B{b} / {LABELS[s]}' for b in [16, 32] for s in LABELS]
    fig, ax = plt.subplots(figsize=(8.7, 5.3), layout='constrained')
    maximum = float(np.max(np.abs(matrix))); norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    plot = ax.imshow(matrix, cmap='RdBu', norm=norm, aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f'{value:+.2f}', ha='center', va='center', color='white' if abs(value) > .6*maximum else '#111111', fontsize=12)
    ax.set_yticks(range(6), labels); ax.set_xticks(range(3), ['Isotropic marginals', 'Full 3-D marginals', 'Propagated full'])
    ax.axhline(2.5, color='white', lw=3)
    ax.set_title('Fixed depth-weighted geometry: path error gains (mm)\nEqual mean noise variance; positive means better than original iid', fontsize=12)
    fig.colorbar(plot, ax=ax, label='Original absolute error - candidate absolute error (mm)', shrink=.88)
    fig.supxlabel('Seen fr3 development. Each cell is the mean of five fixed windows. No candidate promoted.', fontsize=9)
    fig.savefig(OUT/'uncertainty_controls.png', dpi=180); fig.savefig(OUT/'uncertainty_controls.pdf'); plt.close(fig)

    lines = ['# 传播不确定度候选：完成比较，主门槛未通过', '',
        '**600 个固定条件结果完成；主候选失败，保留现有链，不启动大规模训练。** 本轮固定 5 条既有几何链、所有选帧及对应缓存，只改变路程估计器的观测误差结构。全部 600 项无数值失败或回退；150 个 iid 数值对照的 q 与旧结果完全一致，路程最大复算差 1.17×10⁻¹² m。', '',
        '主候选预先指定为 **depth_weighted + propagated_full，B16**。虽然相对原 LG 主对照有 19.59% 路程收益，但相对已有 depth、Huber、depth×Huber 的改善不够稳定。不能把既有深度加权的收益算成新传播模型的独立贡献，也不能在结果后换用表中表现更好的次要组合。', '',
        '## 冻结的比较是什么', '',
        '几何链包括 LG、单位重拟合、深度加权、Huber、深度×Huber，均使用旧配准矩阵和同一原始 RANSAC 共识。每条链配 4 种观测协方差结构，两个预算、3 个录像、每录像 5 个窗口，共 5×4×2×3×5=600 个结果。', '',
        '原状态先验保持不变：Wiener 速度过程、初始位置/速度各轴方差 100、原 13 点 q 网格、每区间 8 点积分；q 只由当前候选的观测似然选择。新的稠密 Gaussian conditioning 处理跨时间、跨坐标的协方差，iid 模式复现旧 Kalman/RTS。', '',
        '局部协方差代理采用逆配准 A（下一相机坐标 → 当前相机坐标）的左 SE(3) 小扰动。对预测源点 y=A·target，J=[I,−skew(y)]；H=ΣwJᵀJ，C=(Σw‖residual‖²/(3N−6))H⁻¹。w 是该冻结几何链的原有最终权重，归一化均值为 1；条件数超 10¹² 或非法输入明确失败。', '',
        '随后假设各边小扰动独立，以 Jⱼᵢ=[Rᵢ,−skew(pⱼ−pᵢ)Rᵢ]（j>i）传播到所有后续位置。由同一条边影响多个位置产生时间相关；这尚未包括相邻边共享观测带来的交叉协方差。', '',
        '统一尺度规则：非锚点传播块整体归一化，取 95% 传播结构 + 5% iid 底噪，使平均坐标边际方差仍为 0.003² m²；锚点沿用原 0.003²I。所有模式的总边际方差相同。该操作主动放弃绝对协方差标定，只检验相对结构。没有用 GT 拟合尺度或扫参数。']
    table(lines, ['模式', '保留的结构', '作用'])
    row(lines, ['iid', '每点 0.003²I，点间独立', '原方法数值对照'])
    row(lines, ['isotropic_marginal', '每点使用完整候选块的 trace/3，方向等方差，点间独立', '只改变沿时间的边际权重'])
    row(lines, ['block_marginal', '保留每点完整 3×3 块，去掉跨时间块', '在前者上加入方向性'])
    row(lines, ['propagated_full', '保留完整矩阵', '进一步加入累积产生的时间相关；主候选'])
    lines += ['', '这些都是传统估计工具及本轮明确限定的代理假设，不是新协方差理论。Censi 的工作指出配准协方差需要考虑观测和对应的依赖，不能把当前目标函数形状直接等同于实际估计误差分布；本轮没有实现其完整观测扰动模型。[作者原论文](https://censi.science/pub/research/2007-icra-icpcov.pdf)。', '',
        'SE(3) 位姿不确定度及复合传播已有系统研究，本轮使用一阶小扰动框架，并用数值扰动验证坐标约定。[Barfoot 与 Furgale，2014，ETH 官方记录](https://www.research-collection.ethz.ch/entities/publication/ba3a20a6-c07e-4535-a435-10b2dbccb582)。固定先验滤波与 Gaussian conditioning 的基础见 [Särkkä 与 Svensson 教材](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)。', '',
        '## 主候选与四条强对照', '',
        '预先门槛要求同时超过四条对照：平均路程绝对误差改善至少 5 mm 且 5%，每录像路程均改善、原始位置 RMSE 不变差、15 窗口完整且无回退。下表是序列等权结果；单位 mm。']
    table(lines, ['对照', '对照路程 MAE', '主候选路程 MAE', '改善', '相对改善', '每录像路径均改善', '门槛'])
    for c in decision['comparisons']:
        base_error = mean(p['baseline_absolute_path_error_m'] for p in c['pairs'])
        candidate = mean(p['candidate_absolute_path_error_m'] for p in c['pairs'])
        row(lines, [c['baseline'], f'{base_error*1000:.3f}', f'{candidate*1000:.3f}', fmt(c['sequence_equal_gain_m']), f'{c["relative_gain"]*100:.2f}%', c['checks']['each_sequence_path_positive'], '通过' if c['passed'] else '未通过'])
    lines += ['', '主候选相对已有深度加权：far −1.627 mm、planar −0.447 mm、sitting +8.254 mm。相对深度×Huber：far −4.635 mm、planar −0.891 mm、sitting +6.966 mm，且原深度几何的位置 RMSE 在这三个录像均值上都逊于该对照。即使只看路程，主候选也已失败。', '',
        '## 相同几何下的机制对照与预算反例', '',
        '下表只看 depth_weighted 几何，避免混入前端或重拟合收益。正收益表示优于该几何原来的 iid 路程链。']
    table(lines, ['预算', '模式', '总体路程 MAE mm', '相对原链改善 mm', 'far 改善 mm', 'planar 改善 mm', 'sitting 改善 mm'])
    for budget in [16, 32]:
        for mode in MODES:
            gg = [lookup[budget, 'depth_weighted', mode, s] for s in LABELS]
            row(lines, [budget, mode, f'{mean(g["path_mae_m"] for g in gg)*1000:.3f}', fmt(mean(g['path_gain_vs_same_geometry_iid_m'] for g in gg)), *[fmt(g['path_gain_vs_same_geometry_iid_m']) for g in gg]])
    lines += ['', '![同几何的误差结构对照](E:/research/VLLM/paper3/experiments/propagated_uncertainty_v1_2026-09-14/uncertainty_controls.png)', '',
        '**B32 是主要反证：** 主结构把已有深度链路程 MAE 从 48.611 增至 71.169 mm，恶化 22.557 mm。far 和 planar 的各 5 个窗口全部变差，sitting 的各 5 个窗口全部改善；不能取三组中一组作成功结论。去掉时间相关也未消除 B32 问题，两个边际模式仍整体变差。', '',
        'B16 的等方差边际模式在三个录像均值均略有收益，但整体只有 1.796 mm；B32 转为整体 −3.540 mm。本轮不会据此把它换成主方法。', '',
        '## 全部几何链均值，防止只展示主候选', '',
        '每行仍按三个录像等权，列为路程 MAE，单位 mm；单位重拟合用于数值控制。']
    table(lines, ['预算', '冻结几何', 'iid', 'isotropic_marginal', 'block_marginal', 'propagated_full'])
    for budget in [16, 32]:
        for geometry in GEOMETRIES:
            values = [mean(lookup[budget, geometry, mode, s]['path_mae_m'] for s in LABELS)*1000 for mode in MODES]
            row(lines, [budget, geometry, *[f'{v:.3f}' for v in values]])
    lines += ['', '## q 变化不足以单独解释失败', '',
        '以下读取已冻结的预测，不新增或选择 q。比较 full-depth 与同几何 iid：']
    table(lines, ['预算', '录像', 'q 改变窗口数', '预测路程变长窗口数', '同几何 full 路程变差窗口数'])
    for budget in [16, 32]:
        for sequence in LABELS:
            part = [r for r in predictions if (r['budget'], r['geometry'], r['mode'], r['sequence']) == (budget, 'depth_weighted', 'propagated_full', sequence)]
            qchanges = sum(r['q'] != plook[sequence, r['offset'], budget, 'depth_weighted', 'iid']['q'] for r in part)
            longer = sum(r['path_m'] > plook[sequence, r['offset'], budget, 'depth_weighted', 'iid']['path_m'] for r in part)
            wins = lookup[budget, 'depth_weighted', 'propagated_full', sequence]['path_win_count_vs_same_geometry_iid']
            row(lines, [budget, LABELS[sequence], qchanges, longer, 5-wins])
    lines += ['', 'B32 far / 13 秒和 17 秒中，q 都仍为原 0.177827941，路径却分别变长且误差恶化 68.367 和 62.673 mm；所以至少这两个反例来自改换观测误差结构后的估计变化，不能只归咎于 q 网格跳变。其他窗口 q 同时改变，未做固定 q 的因果归因。', '',
        '该结构大多使路径变长：B32 的 15/15，B16 的 14/15。因此它有利于原先低估明显的 sitting，同时容易加剧其他窗口的高估。这是本轮固定输出上的观察，尚未证明是哪一种真实噪声机制造成。', '',
        '## 位置指标保持可比', '',
        '同一几何链的原始位置轨迹完全没有改变；论文已有 raw rigid position RMSE 仍用旧值。另存共同时间下的平滑位置 RMSE，只与同先验 iid 平滑结果比较，绝不偷换已有位置指标。full-depth 在 B16 三录像的平滑位置均值都变差，分别约 0.596/0.451/0.279 mm；B32 far/planar 变差 0.681/1.334 mm，sitting 改善 0.189 mm。路径某些收益不等于平滑位置全面更好。', '',
        '## 下一条路径的缓存条件已核查', '',
        '当前代理假设各配准边独立。前一轮在共同坐标系发现部分录像局部误差有负相邻相关；这不与累计位置误差的正相关矛盾。共享帧观测可能贡献相邻边耦合，但现有诊断尚不能单独证明其物理来源。', '',
        f'本轮追加纯缓存身份核查：{shared["adjacent_edge_pairs"]} 对相邻边，共 {shared["shared_feature_occurrences"]:,} 次共享特征出现，中位每邻边对 {shared["shared_ids_median"]:.0f} 个共享 ID、相对较小共识集占比中位数 {shared["shared_fraction_median"]*100:.2f}%。所有共享 ID 对应的 3D 坐标差严格为 0，无重复 ID。没有读取 GT 或新图像。', '',
        '这说明可以在当前缓存上实现并验证“同一个观测在相邻配准中怎样共同传播”的线性影响模型。它只证明数据关联条件具备，没有识别真实点噪声协方差，也没有产生新候选结果。不能把全局传播矩阵有非对角元素等同于已经考虑全部共享观测相关。', '',
        '下一步先做一个明确的 CPU 试验：固定点噪声代理和权重，显式保留共享帧在相邻边中的交叉项，并与同边际、去交叉项的模型比较；先通过 PSD、坐标变换和数值扰动验证，再冻结评分。偏差、RANSAC 条件化、空间相关、绝对尺度标定仍须列为未解决项。', '',
        '## 运行、复算与研究边界', '',
        f'- 预测阶段 {runtime["wall_s"]:.3f} 秒（不含输入哈希检查），3,450 次局部代理计算，600 个预测，0 失败/回退。新增 GPU 调用/图像解码为 0。',
        '- 7 项新合成测试包括旧 RTS/NLL 复现、独立稠密 Gaussian 公式、逆配准协方差旋转一致性、位姿数值扰动、同总边际方差、整链坐标不变性及非法输入。',
        '- 候选源代码、测试、参数与旧输入先冻结 3,465 个文件；全部预测和协方差随后封存，才重新读取已见 fr3 GT。',
        '- 独立复核覆盖 600 个预测、7,800 个 q 似然值、150 组协方差、3,450 条边、全部门槛和共享缓存。稳定 QR 补核的 NLL 最大差 1.20×10⁻⁸、路径最大差 7.32×10⁻¹² m，所有 q 一致。全套 291 项测试通过，无跳过或失败。',
        '- 一个未触发的回退文案问题已记录：配置写“原始位姿”，但预测 JSON 的 positions 字段在回退时实际返回 iid 平滑位置；公开 raw 指标始终引用原几何。本轮 0 回退，已评分结果不受影响。以后重用应以新版本明确字段，不能据现有文字误读。',
        '- 全部 120 组汇总在 summary.json，600 个逐窗口评价在 evaluation.json，主门槛的全部逐窗口比较在 decision.json；B16/B32 的反例未删改。',
        '- 这是已见 TUM RGB-D 相机自运动代理数据的开发实验，不是未见验证，也不是原始固定相机运动对象任务的最终证据。', '',
        'ALE 01 的预算受控读取、传统检测与共同时间评分冒烟已完成：B16/B8 分别解码 16/8 帧，预测全部封存后，对全部 8 个稀疏参考时点评分，OSPA 型均值 14.853/15.036 μm。它只证明工具链可运行，不能代替领域强基线比较，02 不因本轮结果开放。详见 [01 冒烟记录](E:/research/VLLM/paper3/reports/HUH7_01_BUDGET_SMOKE_V1_2026-09-14.md)。root 另用独立指派公式复算其全部 16 个评分、预算日志与插值输出，分数/中心最大差均为 0。当前证据不支持开始大规模训练或追加 A100 长时实验。', '',
        '[主独立复核](E:/research/VLLM/paper3/experiments/propagated_uncertainty_v1_2026-09-14/independent_review.json) · [稳定 QR 补核](E:/research/VLLM/paper3/experiments/propagated_uncertainty_v1_2026-09-14/independent_stable_gp_review.json) · [全套测试与阶段封存](E:/research/VLLM/paper3/experiments/propagated_uncertainty_v1_2026-09-14/completion.json)。', '']
    with REPORT.open('x', encoding='utf-8') as f: f.write('\n'.join(lines))
    print(REPORT)


if __name__ == '__main__': main()
