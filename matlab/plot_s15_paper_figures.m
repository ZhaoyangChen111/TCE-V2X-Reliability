function plot_s15_paper_figures(root_results, run_nc, run_c)
% Paper-style figure package for S15 A-line + corrected MDP-lite
% Usage:
%   plot_s15_paper_figures('../workspace/results/runs', ...
%       'S15_NoCong_RUN_ID', 'S15_Cong_RUN_ID')

clc; close all;

%% ===================== CONFIG =====================
if nargin ~= 3
    error(['Expected root_results, run_nc, and run_c. ', ...
        'See matlab/README.md for a repository-relative example.']);
end
cfg.root_results = char(root_results);
cfg.run_nc = char(run_nc);
cfg.run_c  = char(run_c);

cfg.main_scenario = 'UrbMask';
cfg.main_retrans  = 2;
cfg.deadline_ms   = 20;
cfg.grace_ms      = 50;

cfg.out_dir = fullfile(cfg.root_results, sprintf('MATLAB_FIGS_S15_AlineMDP_PAPER_%s_%s', cfg.main_scenario, datestr(now,'yyyymmdd_HHMMSS')));
if ~exist(cfg.out_dir,'dir'), mkdir(cfg.out_dir); end

cfg.font_name   = 'Times New Roman';
cfg.font_axis   = 10.5;
cfg.font_title  = 12.2;
cfg.font_legend = 9.5;
cfg.font_note   = 8.8;
cfg.line_width  = 1.9;
cfg.marker_size = 7.2;
cfg.export_res  = 360;

cfg.policy_order = {'noretx','classic','udrc','mdplite','nomikos'};

cfg.policy_label = containers.Map();
cfg.policy_label('noretx')  = 'NoRet';
cfg.policy_label('classic') = 'Classic';
cfg.policy_label('udrc')    = 'UDRC';
cfg.policy_label('mdplite') = 'MDP-lite';
cfg.policy_label('nomikos') = 'Nomikos';

cfg.policy_color = containers.Map();
cfg.policy_color('noretx')  = [0.42 0.42 0.42];
cfg.policy_color('classic') = [0.00 0.45 0.74];
cfg.policy_color('udrc')    = [0.12 0.62 0.33];
cfg.policy_color('mdplite') = [0.56 0.34 0.82];
cfg.policy_color('nomikos') = [0.85 0.37 0.12];

cfg.load_color = containers.Map({'NoCong','Cong'}, {[0.22 0.45 0.88],[0.88 0.28 0.23]});
cfg.state_color = containers.Map({'timely','late','lost'}, {[0.16 0.62 0.34],[0.93 0.68 0.18],[0.76 0.79 0.84]});
cfg.metric_color = containers.Map({'phy','tce'}, {[0.45 0.45 0.45],[0.00 0.43 0.74]});

%% ===================== LOAD =====================
cmpNC = load_compare_bundle(cfg.root_results, cfg.run_nc, cfg.main_scenario, cfg.main_retrans);
cmpC  = load_compare_bundle(cfg.root_results, cfg.run_c , cfg.main_scenario, cfg.main_retrans);
distNC = load_distance_bundle(cfg.root_results, cfg.run_nc, cfg.main_scenario, cfg.main_retrans, cfg.deadline_ms, cfg.grace_ms);
distC  = load_distance_bundle(cfg.root_results, cfg.run_c , cfg.main_scenario, cfg.main_retrans, cfg.deadline_ms, cfg.grace_ms);
heatNC = load_heatmap_bundle(cfg.root_results, cfg.run_nc, cfg.main_retrans);
heatC  = load_heatmap_bundle(cfg.root_results, cfg.run_c , cfg.main_retrans);

make_fig1_tce_vs_distance(cfg, distNC, distC);
make_fig2_state_decomposition(cfg, cmpNC, cmpC);
make_fig3_congestion_performance(cfg, cmpNC, cmpC);
make_fig4_avg_retransmissions(cfg, cmpNC, cmpC);
make_fig5_gain_over_cost(cfg, cmpNC, cmpC);
make_fig6_pdr_vs_tce(cfg, cmpNC, cmpC);
make_fig7_multiscenario_heatmap(cfg, heatNC, heatC);

fprintf('\nAll figures exported to:\n%s\n', cfg.out_dir);
end

%% ===================== LOADERS =====================
function T = load_compare_bundle(root_results, run_id, scenario, retrans_main)
run_dir = fullfile(root_results, run_id, 'tables');
T0 = readtable(fullfile(run_dir, sprintf('policy_compare__%s__ret0.csv', scenario)), 'TextType','string');
Tm = readtable(fullfile(run_dir, sprintf('policy_compare__%s__ret%d.csv', scenario, retrans_main)), 'TextType','string');
T = [T0; Tm];
T.policy_key = normalize_policy_key(T);
T = T(ismember(T.policy_key, string({'noretx','classic','udrc','mdplite','nomikos'})), :);
end

function S = load_distance_bundle(root_results, run_id, scenario, retrans_main, deadline_ms, grace_ms)
run_dir = fullfile(root_results, run_id, 'tables');
S = struct();

f = find_first_csv(run_dir, sprintf('tce_by_distance__%s__ret0__custom__D%d__G%d__*noretx*', scenario, deadline_ms, grace_ms));
if ~isempty(f)
    T = readtable(f, 'TextType','string'); T.policy_key = repmat("noretx", height(T), 1); S.noretx = T;
end

patterns = { ...
    'classic', sprintf('tce_by_distance__%s__ret%d__custom__D%d__G%d__*classic*', scenario, retrans_main, deadline_ms, grace_ms); ...
    'udrc',    sprintf('tce_by_distance__%s__ret%d__custom__D%d__G%d__*udrc*', scenario, retrans_main, deadline_ms, grace_ms); ...
    'mdplite', sprintf('tce_by_distance__%s__ret%d__custom__D%d__G%d__*mdplite*', scenario, retrans_main, deadline_ms, grace_ms); ...
    'nomikos', sprintf('tce_by_distance__%s__ret%d__custom__D%d__G%d__*nomikos*', scenario, retrans_main, deadline_ms, grace_ms)};
for i = 1:size(patterns,1)
    key = patterns{i,1}; pat = patterns{i,2};
    f = find_first_csv(run_dir, pat);
    if ~isempty(f)
        T = readtable(f, 'TextType','string'); T.policy_key = repmat(string(key), height(T), 1); S.(key) = T;
    end
end
end

function H = load_heatmap_bundle(root_results, run_id, retrans_main)
for sc = ["Ref","UrbMask","Tunnel"]
    H.(sc) = load_compare_bundle(root_results, run_id, sc, retrans_main);
end
end

%% ===================== FIGURE 1 =====================
function make_fig1_tce_vs_distance(cfg, distNC, distC)
fig = figure('Color','w','Position',[70 70 1180 325]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(tl); hold(ax1,'on'); box(ax1,'on');
plot_distance_panel(ax1, cfg, distNC);
style_axes(ax1, cfg, 'Distance (m)', 'TCE', '(a) No congestion');
ax2 = nexttile(tl); hold(ax2,'on'); box(ax2,'on');
plot_distance_panel(ax2, cfg, distC);
style_axes(ax2, cfg, 'Distance (m)', 'TCE', '(b) Congestion');
set([ax1 ax2], 'XLim',[18 200], 'YLim',[0 1.0], 'XTick',20:20:200, 'YTick',0:0.2:1);
for ax = [ax1 ax2]
    ax.TickDir = 'out';
end
lgd = legend(ax1, 'Location','northoutside','Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
lgd.Layout.Tile = 'north';
sgtitle(fig, sprintf('TCE versus distance in %s (ret = %d, D20/G50)', cfg.main_scenario, cfg.main_retrans), ...
    'FontName',cfg.font_name,'FontSize',cfg.font_title,'FontWeight','bold');
export_figure(fig, cfg, 'Figure1_TCE_vs_distance_S15_AlineMDP_paper');
end

function plot_distance_panel(ax, cfg, S)
for i = 1:numel(cfg.policy_order)
    k = cfg.policy_order{i};
    if isfield(S, k)
        T = S.(k);
        plot(ax, T.dist_bin_center, T.tce, '-', 'LineWidth', cfg.line_width, 'Color', cfg.policy_color(k), 'DisplayName', cfg.policy_label(k));
    end
end
light_grid(ax);
end

%% ===================== FIGURE 2 =====================
function make_fig2_state_decomposition(cfg, cmpNC, cmpC)
fig = figure('Color','w','Position',[80 80 1180 390]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(tl); hold(ax1,'on'); box(ax1,'on'); plot_state_stack(ax1,cfg,cmpNC); style_axes(ax1,cfg,'Proportion','','(a) No congestion');
ax2 = nexttile(tl); hold(ax2,'on'); box(ax2,'on'); plot_state_stack(ax2,cfg,cmpC);  style_axes(ax2,cfg,'Proportion','','(b) Congestion');
set([ax1 ax2], 'XLim',[0 1], 'XTick',0:0.2:1, 'XTickLabel',compose('%.0f%%',100*(0:0.2:1)));
sgtitle(fig, sprintf('State decomposition in %s (ret = %d)', cfg.main_scenario, cfg.main_retrans), ...
    'FontName',cfg.font_name,'FontSize',cfg.font_title,'FontWeight','bold');
lgd = legend(ax1, {'Timely','Late','Lost'}, 'Location','southoutside','Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
lgd.Layout.Tile = 'south';
export_figure(fig, cfg, 'Figure2_State_decomposition_S15_AlineMDP_paper');
end

function plot_state_stack(ax,cfg,T)
A = align_policies(cfg,T);
timely = A.mean_timely_rate;
late = A.mean_late_ratio_total;
lost = max(0, 1 - timely - late);
Y = [timely, late, lost];
b = barh(ax, Y, 'stacked', 'BarWidth', 0.62);
set(b(1), 'FaceColor', cfg.state_color('timely'), 'EdgeColor','none');
set(b(2), 'FaceColor', cfg.state_color('late'),   'EdgeColor','none');
set(b(3), 'FaceColor', cfg.state_color('lost'),   'EdgeColor','none');
ax.YTick = 1:height(A); ax.YTickLabel = cellstr(policy_labels(cfg, A.policy_key)); ax.YDir = 'reverse';
light_grid(ax);
for i = 1:size(Y,1)
    cum = 0;
    vals = Y(i,:);
    for j=1:3
        v = vals(j);
        xc = cum + v/2;
        if v >= 0.045
            txtc = 'w'; if j==3, txtc = 'k'; end
            text(ax, xc, i, sprintf('%.0f%%',100*v), 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontName',cfg.font_name,'FontSize',9.5,'Color',txtc,'FontWeight','bold');
        elseif v >= 0.012
            text(ax, min(cum+v+0.012,0.98), i-0.23, sprintf('%.0f%%',100*v), 'HorizontalAlignment','left','VerticalAlignment','middle', ...
                'FontName',cfg.font_name,'FontSize',8.8,'Color',[0.20 0.20 0.20],'FontWeight','bold');
        end
        cum = cum + v;
    end
end
end

%% ===================== FIGURE 3 =====================
function make_fig3_congestion_performance(cfg, cmpNC, cmpC)
fig = figure('Color','w','Position',[90 90 900 335]);
ax = axes(fig); hold(ax,'on'); box(ax,'on');
plot_load_pair(ax, cfg, cmpNC, cmpC, 'mean_tce', 'ci95_tce');
ymin = min([cmpNC.mean_tce; cmpC.mean_tce]) - 0.008;
ymax = max([cmpNC.mean_tce; cmpC.mean_tce]) + 0.004;
set(ax,'YLim',[ymin ymax]);
style_axes(ax, cfg, '', 'Mean TCE', sprintf('Load sensitivity in %s (ret = %d)', cfg.main_scenario, cfg.main_retrans));
legend(ax, {'No congestion','Congestion'}, 'Location','northwest', 'Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
export_figure(fig, cfg, 'Figure3_Performance_under_congestion_S15_AlineMDP_paper');
end

%% ===================== FIGURE 4 =====================
function make_fig4_avg_retransmissions(cfg, cmpNC, cmpC)
fig = figure('Color','w','Position',[100 100 900 335]);
ax = axes(fig); hold(ax,'on'); box(ax,'on');
Tnc = cmpNC; Tc = cmpC;
Tnc.mean_retx = Tnc.mean_avg_attempts - 1; Tnc.ci95_retx = Tnc.ci95_avg_attempts;
Tc.mean_retx  = Tc.mean_avg_attempts - 1;  Tc.ci95_retx  = Tc.ci95_avg_attempts;
plot_load_pair(ax, cfg, Tnc, Tc, 'mean_retx', 'ci95_retx');
set(ax,'YLim',[-0.03 1.60]);
style_axes(ax, cfg, '', 'Average retransmissions per packet', sprintf('Retransmission burden in %s (ret = %d)', cfg.main_scenario, cfg.main_retrans));
legend(ax, {'No congestion','Congestion'}, 'Location','northwest', 'Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
export_figure(fig, cfg, 'Figure4_Avg_retransmissions_S15_AlineMDP_paper');
end

%% ===================== FIGURE 5 =====================
function make_fig5_gain_over_cost(cfg, cmpNC, cmpC)
keys = ["mdplite","udrc"];
Tnc = cmpNC(ismember(cmpNC.policy_key, keys), :);
Tc  = cmpC(ismember(cmpC.policy_key, keys), :);
allKeys = align_order_subset(keys, unique([Tnc.policy_key; Tc.policy_key], 'stable'));
fig = figure('Color','w','Position',[110 110 860 320]);
ax = axes(fig); hold(ax,'on'); box(ax,'on');
if isempty(allKeys)
    text(ax,0.5,0.5,'No UDRC/MDP-lite gain-cost diagnostics found.', 'HorizontalAlignment','center', 'FontName',cfg.font_name,'FontSize',cfg.font_axis); axis(ax,'off');
else
    x = 1:numel(allKeys);
    for i = 1:numel(allKeys)
        k = allKeys(i);
        r1 = Tnc(Tnc.policy_key==k,:); r2 = Tc(Tc.policy_key==k,:);
        v1 = getnum(r1,'mean_avg_gain_over_cost'); e1 = getnum(r1,'ci95_avg_gain_over_cost');
        v2 = getnum(r2,'mean_avg_gain_over_cost'); e2 = getnum(r2,'ci95_avg_gain_over_cost');
        plot(ax, [x(i)-0.12 x(i)+0.12], [v1 v2], '-', 'Color',[0.72 0.72 0.72], 'LineWidth',1.2, 'HandleVisibility','off');
        errorbar(ax, x(i)-0.12, v1, e1, 'o', 'Color', cfg.load_color('NoCong'), 'MarkerFaceColor', cfg.load_color('NoCong'), 'MarkerSize',cfg.marker_size,'LineWidth',1.0,'CapSize',8, 'DisplayName', ternary(i==1,'No congestion',''));
        errorbar(ax, x(i)+0.12, v2, e2, 'o', 'Color', cfg.load_color('Cong'),   'MarkerFaceColor', cfg.load_color('Cong'),   'MarkerSize',cfg.marker_size,'LineWidth',1.0,'CapSize',8, 'DisplayName', ternary(i==1,'Congestion',''));
        add_point_label(ax, x(i)-0.12, v1, '%.2f', 0.004); add_point_label(ax, x(i)+0.12, v2, '%.2f', 0.004);
    end
    ax.XTick = x; ax.XTickLabel = cellstr(policy_labels(cfg, allKeys));
    set(ax,'YLim',[0.18 0.27], 'YTick',0.18:0.02:0.26);
    text(ax, x(end)+0.35, 0.265, 'Reference threshold = 1 (outside displayed range)', ...
        'HorizontalAlignment','right', 'FontName',cfg.font_name,'FontSize',cfg.font_note, 'Color',[0.35 0.35 0.35]);
    light_grid(ax);
    legend(ax, {'No congestion','Congestion'}, 'Location','northwest', 'Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
    style_axes(ax, cfg, '', 'Average \Deltau / C_i', sprintf('Average utility-gain ratio in %s (ret = %d)', cfg.main_scenario, cfg.main_retrans));
end
export_figure(fig, cfg, 'Figure5_Gain_over_cost_S15_AlineMDP_paper');
end

%% ===================== FIGURE 6 =====================
function make_fig6_pdr_vs_tce(cfg, cmpNC, cmpC)
fig = figure('Color','w','Position',[95 95 1160 330]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(tl); hold(ax1,'on'); box(ax1,'on'); plot_metric_dumbbell(ax1,cfg,cmpNC); style_axes(ax1,cfg,'','Rate','(a) No congestion');
ax2 = nexttile(tl); hold(ax2,'on'); box(ax2,'on'); plot_metric_dumbbell(ax2,cfg,cmpC);  style_axes(ax2,cfg,'','Rate','(b) Congestion');
ymin = min([cmpNC.mean_tce; cmpNC.mean_phy_rate; cmpC.mean_tce; cmpC.mean_phy_rate]) - 0.015;
ymax = max([cmpNC.mean_tce; cmpNC.mean_phy_rate; cmpC.mean_tce; cmpC.mean_phy_rate]) + 0.015;
set([ax1 ax2], 'YLim',[ymin ymax]);
sgtitle(fig, sprintf('PHY success versus TCE in %s (ret = %d)', cfg.main_scenario, cfg.main_retrans), ...
    'FontName',cfg.font_name,'FontSize',cfg.font_title,'FontWeight','bold');
lgd = legend(ax1, {'PHY success','TCE'}, 'Location','northoutside', 'Orientation','horizontal', 'FontName',cfg.font_name,'FontSize',cfg.font_legend,'Box','off');
lgd.Layout.Tile = 'north';
export_figure(fig, cfg, 'Figure6_PDR_vs_TCE_S15_AlineMDP_paper');
end

function plot_metric_dumbbell(ax,cfg,T)
A = align_policies(cfg,T);
x = 1:height(A);
for i = 1:numel(x)
    v1 = A.mean_phy_rate(i); e1 = A.ci95_phy_rate(i);
    v2 = A.mean_tce(i);      e2 = A.ci95_tce(i);
    plot(ax, [x(i) x(i)], [v2 v1], '-', 'Color',[0.72 0.72 0.72], 'LineWidth',1.2, 'HandleVisibility','off');
    errorbar(ax, x(i)-0.08, v1, e1, 'o', 'Color',cfg.metric_color('phy'), 'MarkerFaceColor',cfg.metric_color('phy'), 'MarkerSize',cfg.marker_size*0.85, 'LineWidth',0.9, 'CapSize',6, 'DisplayName', ternary(i==1,'PHY success',''));
    errorbar(ax, x(i)+0.08, v2, e2, 's', 'Color',cfg.metric_color('tce'), 'MarkerFaceColor',cfg.metric_color('tce'), 'MarkerSize',cfg.marker_size*0.72, 'LineWidth',0.9, 'CapSize',6, 'DisplayName', ternary(i==1,'TCE',''));
    gap = v1 - v2;
    text(ax, x(i), max(v1,v2)+0.0045, sprintf('gap %.3f', gap), 'HorizontalAlignment','center', 'FontName',cfg.font_name,'FontSize',8.8, 'Color',[0.35 0.35 0.35]);
end
ax.XTick = x; ax.XTickLabel = cellstr(policy_labels(cfg,A.policy_key));
light_grid(ax);
end

%% ===================== FIGURE 7 =====================
function make_fig7_multiscenario_heatmap(cfg, heatNC, heatC)
pols = ["classic","udrc","mdplite","nomikos"];
scs = ["Ref","UrbMask","Tunnel"];
M1 = nan(numel(scs), numel(pols));
M2 = nan(numel(scs), numel(pols));
for i = 1:numel(scs)
    Tnc = align_policies(cfg, heatNC.(scs(i)));
    Tc  = align_policies(cfg, heatC.(scs(i)));
    base_nc = get_metric(Tnc,'noretx','mean_tce');
    base_c  = get_metric(Tc, 'noretx','mean_tce');
    for j = 1:numel(pols)
        M1(i,j) = get_metric(Tnc, pols(j), 'mean_tce') - base_nc;
        M2(i,j) = get_metric(Tc,  pols(j), 'mean_tce') - base_c;
    end
end
cmin = min([M1(:); M2(:)]);
cmax = max([M1(:); M2(:)]);
fig = figure('Color','w','Position',[100 100 1080 390]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
ax1 = nexttile(tl); plot_heatmap_panel(ax1, M1, scs, pols, cfg, '(a) No congestion', cmin, cmax, false);
ax2 = nexttile(tl); plot_heatmap_panel(ax2, M2, scs, pols, cfg, '(b) Congestion', cmin, cmax, true);
sgtitle(fig, sprintf('Scenario summary: \DeltaTCE relative to NoRet (ret = %d)', cfg.main_retrans), ...
    'FontName',cfg.font_name,'FontSize',cfg.font_title,'FontWeight','bold');
export_figure(fig, cfg, 'Figure7_Multiscenario_heatmap_S15_AlineMDP_paper');
end

function plot_heatmap_panel(ax, M, scs, pols, cfg, ttl, cmin, cmax, withColorbar)
imagesc(ax, M);
colormap(ax, turbo(256));
caxis(ax, [cmin cmax]);
if withColorbar
    cb = colorbar(ax); cb.FontName = cfg.font_name; cb.FontSize = cfg.font_axis; cb.Label.String = '\DeltaTCE'; cb.Label.FontName = cfg.font_name;
end
ax.XTick = 1:numel(pols); ax.XTickLabel = cellstr(policy_labels(cfg, pols));
ax.YTick = 1:numel(scs); ax.YTickLabel = cellstr(scs);
ax.TickLabelInterpreter = 'none';
set(ax, 'FontName',cfg.font_name, 'FontSize',cfg.font_axis, 'LineWidth',0.9, 'TickDir','out', 'YDir','normal');
title(ax, ttl, 'FontName',cfg.font_name, 'FontSize',cfg.font_title, 'FontWeight','bold');
for i = 1:size(M,1)
    for j = 1:size(M,2)
        val = M(i,j);
        txtc = 'w'; if val < cmin + 0.55*(cmax-cmin), txtc = 'k'; end
        text(ax, j, i, sprintf('%.3f', val), 'HorizontalAlignment','center', 'FontName',cfg.font_name, 'FontSize',10, 'FontWeight','bold', 'Color',txtc);
    end
end
end

%% ===================== HELPERS =====================
function plot_load_pair(ax, cfg, Tnc, Tc, ycol, ecol)
A = align_policies(cfg, Tnc); B = align_policies(cfg, Tc);
x = 1:height(A);
for i = 1:numel(x)
    y1 = A.(ycol)(i); y2 = B.(ycol)(i);
    e1 = A.(ecol)(i); e2 = B.(ecol)(i);
    plot(ax, [x(i)-0.10 x(i)+0.10], [y1 y2], '-', 'Color',[0.72 0.72 0.72], 'LineWidth',1.2, 'HandleVisibility','off');
    errorbar(ax, x(i)-0.10, y1, e1, 'o', 'Color',cfg.load_color('NoCong'), 'MarkerFaceColor',cfg.load_color('NoCong'), 'MarkerSize',cfg.marker_size, 'LineWidth',1.0, 'CapSize',8, 'DisplayName', ternary(i==1,'No congestion',''));
    errorbar(ax, x(i)+0.10, y2, e2, 'o', 'Color',cfg.load_color('Cong'),   'MarkerFaceColor',cfg.load_color('Cong'),   'MarkerSize',cfg.marker_size, 'LineWidth',1.0, 'CapSize',8, 'DisplayName', ternary(i==1,'Congestion',''));
    add_point_label(ax, x(i)-0.10, y1, '%.3f', 0.0018);
    add_point_label(ax, x(i)+0.10, y2, '%.3f', 0.0018);
end
ax.XTick = x; ax.XTickLabel = cellstr(policy_labels(cfg, A.policy_key));
light_grid(ax);
end

function A = align_policies(cfg, T)
keys = string(cfg.policy_order(:));
A = table(keys, 'VariableNames', {'policy_key'});
for c = 1:numel(T.Properties.VariableNames)
    vn = T.Properties.VariableNames{c};
    if strcmp(vn,'policy_key'), continue; end
    if isnumeric(T.(vn))
        A.(vn) = nan(height(A),1);
    else
        A.(vn) = strings(height(A),1);
    end
end
for i = 1:height(A)
    idx = find(T.policy_key == A.policy_key(i), 1, 'first');
    if ~isempty(idx)
        for c = 1:numel(T.Properties.VariableNames)
            vn = T.Properties.VariableNames{c};
            if strcmp(vn,'policy_key'), continue; end
            A.(vn)(i) = T.(vn)(idx);
        end
    end
end
end

function keys = normalize_policy_key(T)
keys = strings(height(T),1);
for i = 1:height(T)
    if ismember('retx_policy', T.Properties.VariableNames)
        rp = lower(string(T.retx_policy(i)));
    else
        rp = "";
    end
    if ismember('policy_tag', T.Properties.VariableNames)
        pt = lower(string(T.policy_tag(i)));
    else
        pt = "";
    end
    if contains(rp,'noretx') || contains(pt,'noretx')
        keys(i) = "noretx";
    elseif contains(rp,'classical') || contains(pt,'classic')
        keys(i) = "classic";
    elseif contains(rp,'udrc') || contains(pt,'udrc')
        keys(i) = "udrc";
    elseif contains(rp,'mdp') || contains(pt,'mdp')
        keys(i) = "mdplite";
    elseif contains(rp,'nomikos') || contains(pt,'nomikos')
        keys(i) = "nomikos";
    else
        keys(i) = "other";
    end
end
end

function labels = policy_labels(cfg, keys)
labels = strings(numel(keys),1);
for i = 1:numel(keys)
    if isKey(cfg.policy_label, char(keys(i)))
        labels(i) = string(cfg.policy_label(char(keys(i))));
    else
        labels(i) = string(keys(i));
    end
end
end

function ordered = align_order_subset(order_keys, seen_keys)
ordered = strings(0,1);
for i = 1:numel(order_keys)
    if any(seen_keys == order_keys(i))
        ordered(end+1,1) = string(order_keys(i)); %#ok<AGROW>
    end
end
end

function f = find_first_csv(run_dir, pattern)
if endsWith(pattern, '.csv')
    query = pattern;
else
    query = [pattern '.csv'];
end
D = dir(fullfile(run_dir, query));
if isempty(D)
    f = '';
else
    [~,idx] = max([D.datenum]);
    f = fullfile(D(idx).folder, D(idx).name);
end
end

function v = get_metric(T, key, col)
idx = find(T.policy_key == key, 1, 'first');
if isempty(idx), v = nan; else, v = T.(col)(idx); end
end

function v = getnum(T, col)
if isempty(T) || ~ismember(col, T.Properties.VariableNames), v = nan; else, v = T.(col)(1); end
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function add_point_label(ax, x, y, fmt, dy)
if nargin < 5, dy = 0.012*range_nonzero(ax.YLim); end
if ~isnan(y)
    text(ax, x, y + dy, sprintf(fmt, y), 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontName','Times New Roman', 'FontSize',8.8, 'Color',[0.35 0.35 0.35]);
end
end

function r = range_nonzero(lim)
r = lim(2)-lim(1); if r<=0, r=1; end
end

function light_grid(ax)
grid(ax,'on'); ax.GridAlpha = 0.13; ax.MinorGridAlpha = 0.07; ax.XMinorGrid='off'; ax.YMinorGrid='off';
end

function style_axes(ax, cfg, xlab, ylab, ttl)
set(ax,'FontName',cfg.font_name,'FontSize',cfg.font_axis,'LineWidth',0.95,'TickDir','out');
if ~isempty(xlab), xlabel(ax, xlab, 'FontName',cfg.font_name,'FontSize',cfg.font_axis); end
if ~isempty(ylab), ylabel(ax, ylab, 'FontName',cfg.font_name,'FontSize',cfg.font_axis); end
if ~isempty(ttl), title(ax, ttl, 'FontName',cfg.font_name,'FontSize',cfg.font_title, 'FontWeight','bold'); end
end

function export_figure(fig, cfg, stem)
set(fig,'InvertHardcopy','off');
exportgraphics(fig, fullfile(cfg.out_dir, [stem '.png']), 'Resolution', cfg.export_res);
exportgraphics(fig, fullfile(cfg.out_dir, [stem '.pdf']), 'ContentType','vector');
end
