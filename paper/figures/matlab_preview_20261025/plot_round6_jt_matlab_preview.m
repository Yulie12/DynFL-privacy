function plot_round6_jt_matlab_preview(publishToPaper)
% Replot round-6 J/T scores from the existing experiment JSON records.
% false: create an independent preview. true: also replace the paper PDF/PNG.
% No training, source CSV, experiment record, or LaTeX file is modified.

if nargin == 0
    publishToPaper = false;
end
assert(islogical(publishToPaper) && isscalar(publishToPaper));
previewDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(previewDir);
repoDir = fileparts(fileparts(figuresDir));
runDir = fullfile(repoDir, 'out', 'fmnist_joint_jt_100r_3seed_20261022');
names = {'P0', 'P1', 'P2', 'P3', 'P4'};
labels = {'P_0', 'P_1', 'P_2', 'P_3', 'P_4'};
seeds = [42, 43, 44];
colors = [153, 153, 153; 213, 94, 0; 0, 114, 178; ...
    0, 158, 115; 204, 121, 167] / 255;
markers = {'s', '^', 'd', 'o', 'v'};
t = zeros(5, 3);
j = zeros(5, 3);

for s = 1:numel(seeds)
    inputPath = fullfile(runDir, sprintf('seed_%d', seeds(s)), ...
        'greedy_lambda075', 'candidates_round_005.json');
    payload = jsondecode(fileread(inputPath));
    assert(payload.applied_round == 6 && payload.decision_after_round == 5, ...
        'The source must describe the first online decision at round 6.');
    assert(strcmp(payload.previous_candidate, 'P0'));
    assert(numel(payload.candidates) == 5);
    for k = 1:5
        c = payload.candidates(strcmp({payload.candidates.name}, names{k}));
        assert(isscalar(c) && c.feasible && strcmp(c.status, 'ready'), ...
            'Each candidate must be present and feasible in all three seeds.');
        t(k, s) = c.T;
        j(k, s) = c.J;
    end
end
assert(all(isfinite(t(:))) && all(isfinite(j(:))));
tMean = mean(t, 2);
jMean = mean(j, 2);
tSD = std(t, 0, 2); % sample SD, denominator n-1
jSD = std(j, 0, 2);

% Independently check the existing statistics without rewriting them.
summary = readtable(fullfile(figuresDir, 'three_seed_aggregate', ...
    'round6_candidate_mean_sd.csv'), 'TextType', 'string');
perSeed = readtable(fullfile(figuresDir, 'three_seed_aggregate', ...
    'round6_candidate_per_seed.csv'), 'TextType', 'string');
for k = 1:5
    row = summary(summary.candidate == names{k}, :);
    assert(height(row) == 1 && row.group_size == 3 ...
        && row.T_n == 3 && row.J_n == 3);
    assert(max(abs([tMean(k), tSD(k), jMean(k), jSD(k)] ...
        - [row.T_mean, row.T_std, row.J_mean, row.J_std])) < 1e-9, ...
        'Recomputed mean/sample SD differs from the saved statistics.');
    for s = 1:numel(seeds)
        observation = perSeed(perSeed.seed == seeds(s) ...
            & perSeed.candidate == names{k}, :);
        assert(height(observation) == 1);
        assert(max(abs([t(k, s), j(k, s)] ...
            - [observation.T, observation.J])) < 1e-9);
    end
end
[~, bestJ] = max(j, [], 1);
[~, bestT] = min(t, [], 1);
assert(all(bestJ == 3), 'The maximum-J callout must match all raw seeds.');

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'centimeters', 'Position', [2, 2, 8.9, 8.4], ...
    'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
% A white canvas boundary preserves outer whitespace in exportgraphics.
% It has no data content and stays outside all axes and annotations.
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);
ax = axes(fig, 'Position', [0.18, 0.22, 0.79, 0.65]);
hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.014, 0.014], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
xlim(ax, [1.75, 2.02]);
ylim(ax, [2.55, 4.95]);
xticks(ax, 1.75:0.05:2.00);
yticks(ax, [2.75, 3.00, 3.50, 4.00, 4.50]);
xtickformat(ax, '%.2f');
ytickformat(ax, '%.2f');
grid(ax, 'on');
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.25;
ax.GridLineStyle = '-';

% Both dimensions use the full sample SD, with no rescaling or truncation.
for k = 1:5
    drawErrorCross(ax, tMean(k), jMean(k), tSD(k), jSD(k), colors(k, :));
end
h = gobjects(1, 5);
for k = 1:5
    h(k) = plot(ax, tMean(k), jMean(k), 'LineStyle', 'none', ...
        'Marker', markers{k}, 'MarkerSize', 6, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', [0.20, 0.20, 0.20], ...
        'LineWidth', 0.65, 'Color', colors(k, :));
end
xlabel(ax, 'Predicted round latency, {\it T} (s)', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');
ylabel(ax, 'Historical deviation contribution, {\it J}', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');

% Candidate labels stay outside the data area and cannot cover error bars.
lgd = legend(ax, h, labels, 'NumColumns', 5, 'Orientation', 'horizontal', ...
    'Box', 'off', 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'Interpreter', 'tex');
lgd.Units = 'normalized';
lgd.Position = [0.17, 0.92, 0.81, 0.05];
lgd.ItemTokenSize = [10, 8];
text(ax, 0.03, 0.965, 'Largest {\it J}: P_2 in 3/3 seeds', ...
    'Units', 'normalized', 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'Color', colors(3, :), 'Interpreter', 'tex', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

% Report the seed-specific latency winners instead of a boundary of means.
annotation(fig, 'textbox', [0.18, 0.068, 0.79, 0.04], ...
    'String', 'Minimum {\it T} by seed', 'Interpreter', 'tex', ...
    'FontName', 'Times New Roman', 'FontSize', 7.5, 'Color', [0.25, 0.25, 0.25], ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', 'Margin', 0);
for s = 1:numel(seeds)
    annotation(fig, 'textbox', [0.18 + (s-1)*0.79/3, 0.022, 0.79/3, 0.04], ...
        'String', sprintf('%d: %s', seeds(s), labels{bestT(s)}), ...
        'Interpreter', 'tex', 'FontName', 'Times New Roman', 'FontSize', 8, ...
        'Color', colors(bestT(s), :), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'EdgeColor', 'none', 'Margin', 0);
end

drawnow;
previewStem = fullfile(previewDir, 'experiment_round6_jt_matlab_preview');
% Export vector PDF with embedded fonts. The white boundary above preserves
% the canvas margins that would otherwise be removed by tight cropping.
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [8.9, 8.4], ...
    'PaperPosition', [0, 0, 8.9, 8.4], 'PaperPositionMode', 'manual');
exportgraphics(fig, [previewStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
print(fig, [previewStem, '.png'], '-dpng', '-r600');
savefig(fig, [previewStem, '.fig']);

if publishToPaper
    backupDir = fullfile(previewDir, 'original_figure_backup');
    if ~isfolder(backupDir)
        mkdir(backupDir);
    end
    paperStem = fullfile(figuresDir, 'experiment_round6_J_T_candidates');
    for extension = {'.pdf', '.png'}
        suffix = extension{1};
        targetPath = [paperStem, suffix];
        [~, backupName, backupSuffix] = fileparts(targetPath);
        backupPath = fullfile(backupDir, [backupName, backupSuffix]);
        if isfile(targetPath) && ~isfile(backupPath)
            [ok, message] = copyfile(targetPath, backupPath);
            assert(ok, 'Original figure backup failed: %s', message);
        end
        [ok, message] = copyfile([previewStem, suffix], targetPath, 'f');
        assert(ok, 'Paper figure replacement failed: %s', message);
        fprintf('PAPER_FIGURE=%s\n', targetPath);
    end
    fprintf('ORIGINAL_FIGURE_BACKUP=%s\n', backupDir);
end
fprintf('MATLAB_RELEASE=%s\n', version('-release'));
fprintf('SOURCE=raw candidates_round_005.json for seeds 42,43,44\n');
fprintf('CHECK=raw records and existing CSV statistics agree\n');
fprintf('FIGURE_PDF=%s.pdf\n', previewStem);
fprintf('FIGURE_PNG=%s.png\n', previewStem);
for k = 1:5
    fprintf('%s T=%.12f SD=%.12f J=%.12f SD=%.12f\n', ...
        names{k}, tMean(k), tSD(k), jMean(k), jSD(k));
end
for s = 1:numel(seeds)
    fprintf('SEED=%d MAX_J=%s MIN_T=%s\n', seeds(s), names{bestJ(s)}, names{bestT(s)});
end
end

function drawErrorCross(ax, x, y, sx, sy, color)
common = {'Color', color, 'LineWidth', 0.85, 'HandleVisibility', 'off'};
plot(ax, [x-sx, x+sx], [y, y], common{:});
plot(ax, [x, x], [y-sy, y+sy], common{:});
for endpoint = [x-sx, x+sx]
    plot(ax, [endpoint, endpoint], [y-0.025, y+0.025], common{:});
end
for endpoint = [y-sy, y+sy]
    plot(ax, [x-0.0016, x+0.0016], [endpoint, endpoint], common{:});
end
end
