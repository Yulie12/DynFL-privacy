function plot_round6_j_seed_profiles()
% Independent preview of round-6 candidate J for each original seed.
% Read-only inputs. The existing paper figure and main.tex are not replaced.

previewDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(previewDir);
repoDir = fileparts(fileparts(figuresDir));
campaignDir = fullfile(repoDir, 'out', 'fmnist_joint_jt_100r_3seed_20261022');
seeds = [42, 43, 44];
names = {'P0', 'P1', 'P2', 'P3', 'P4'};
labels = {'P_0', 'P_1', 'P_2', 'P_3', 'P_4'};
j = zeros(5, 3);
finalLoss = zeros(5, 3);
savedJ = readtable(fullfile(figuresDir, 'three_seed_aggregate', ...
    'round6_candidate_per_seed.csv'), 'TextType', 'string');
for s = 1:numel(seeds)
    seedDir = fullfile(campaignDir, sprintf('seed_%d', seeds(s)));
    records = jsondecode(fileread(fullfile(seedDir, ...
        'greedy_lambda075', 'candidates_round_005.json')));
    assert(records.applied_round == 6 && records.decision_after_round == 5);
    assert(strcmp(records.previous_candidate, 'P0'));
    assert(numel(records.candidates) == 5);
    for k = 1:5
        c = records.candidates(strcmp({records.candidates.name}, names{k}));
        assert(isscalar(c) && c.feasible && strcmp(c.status, 'ready'));
        j(k, s) = c.J;
        checkRow = savedJ(savedJ.seed == seeds(s) ...
            & savedJ.candidate == names{k}, :);
        assert(height(checkRow) == 1 && abs(checkRow.J - c.J) < 1e-9);
        metrics = readtable(fullfile(seedDir, ['fixed_', names{k}], ...
            'round_metrics.csv'), 'VariableNamingRule', 'preserve');
        assert(height(metrics) == 100 && isequal(metrics.round, (1:100)'));
        finalLoss(k, s) = metrics.test_loss(end);
    end
end
assert(all(isfinite(j(:))) && all(isfinite(finalLoss(:))));

colors = [0, 114, 178; 213, 94, 0; 0, 158, 115] / 255;
styles = {'-', '--', ':'};
markers = {'o', '^', 's'};
sizes = [6, 4.6, 3.2];
fig = figure('Visible', 'off', 'Color', 'white', 'Units', 'centimeters', ...
    'Position', [2, 2, 8.9, 6.7], 'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);
ax = axes(fig, 'Position', [0.18, 0.18, 0.78, 0.68]);
hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.014, 0.014], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
handles = gobjects(1, 3);
for s = 1:numel(seeds)
    handles(s) = plot(ax, 1:5, j(:, s), 'Color', colors(s, :), ...
        'LineStyle', styles{s}, 'LineWidth', 0.9, ...
        'Marker', markers{s}, 'MarkerSize', sizes(s), ...
        'MarkerFaceColor', 'white', 'MarkerEdgeColor', colors(s, :));
end
xlim(ax, [0.72, 5.28]);
ylim(ax, [2.5, 4.9]);
xticks(ax, 1:5);
xticklabels(ax, labels);
yticks(ax, 2.5:0.5:4.5);
ytickformat(ax, '%.1f');
grid(ax, 'on');
ax.XGrid = 'off';
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.25;
xlabel(ax, 'Candidate', 'FontName', 'Times New Roman', 'FontSize', 8.5);
ylabel(ax, 'Historical deviation contribution, {\it J}', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');
legendLabels = arrayfun(@(seed) sprintf('Seed %d', seed), seeds, ...
    'UniformOutput', false);
lgd = legend(ax, handles, legendLabels, 'NumColumns', 3, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontName', 'Times New Roman', ...
    'FontSize', 8, 'Interpreter', 'none');
lgd.Units = 'normalized';
lgd.Position = [0.17, 0.925, 0.80, 0.05];
lgd.ItemTokenSize = [16, 8];
drawnow;
outputStem = fullfile(previewDir, 'experiment_round6_J_by_seed');
exportgraphics(fig, [outputStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [8.9, 6.7], ...
    'PaperPosition', [0, 0, 8.9, 6.7], 'PaperPositionMode', 'manual');
print(fig, [outputStem, '.png'], '-dpng', '-r600');
savefig(fig, [outputStem, '.fig']);

fprintf('FIGURE_PDF=%s.pdf\n', outputStem);
fprintf('FIGURE_PNG=%s.png\n', outputStem);
fprintf('CHECK=raw JSON J agrees with saved per-seed values\n');
fprintf('CHECK=all fifteen fixed runs contain rounds 1 through 100\n');
for s = 1:numel(seeds)
    [~, jOrder] = sort(j(:, s), 'descend');
    [~, lossOrder] = sort(finalLoss(:, s), 'ascend');
    fprintf('SEED=%d J_DESC=%s FINAL_LOSS_ASC=%s\n', ...
        seeds(s), strjoin(names(jOrder), '>'), strjoin(names(lossOrder), '<'));
end
[~, meanOrder] = sort(mean(finalLoss, 2), 'ascend');
fprintf('MEAN_FINAL_LOSS_ASC=%s\n', strjoin(names(meanOrder), '<'));
for k = 1:5
    fprintf('%s J=[%.12f %.12f %.12f] FINAL_LOSS=[%.12f %.12f %.12f]\n', ...
        names{k}, j(k, :), finalLoss(k, :));
end
end
