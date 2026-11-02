function plot_fixed_loss_matlab_preview(publishToPaper)
% Plot the Fixed comparison from existing three-seed statistics.
% plot_fixed_loss_matlab_preview(false) updates only the independent preview.
% plot_fixed_loss_matlab_preview(true) also replaces the paper PDF and PNG,
% retaining the original files in original_figure_backup before replacement.
% No experiment data or LaTeX source is modified. MATLAB R2022b.

if nargin == 0
    publishToPaper = false;
end
assert(islogical(publishToPaper) && isscalar(publishToPaper), ...
    'publishToPaper must be a scalar logical value.');

previewDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(previewDir);
inputPath = fullfile(figuresDir, 'three_seed_aggregate', ...
    'round_metrics_mean_sd.csv');
stats = readtable(inputPath, 'TextType', 'string');
strategies = "fixed_P" + string(0:4);
colors = [77, 77, 77; 213, 94, 0; 0, 114, 178; ...
    0, 158, 115; 204, 121, 167] / 255;
styles = {'-', '--', '-.', ':', '-'};
labels = {'Fixed P_0', 'Fixed P_1', 'Fixed P_2', ...
    'Fixed P_3', 'Fixed P_4'};
threshold = 1.40;
series = cell(1, 5);
firstPoints = zeros(5, 3);

for j = 1:5
    rows = sortrows(stats(stats.strategy == strategies(j), :), 'round');
    assert(height(rows) == 100 && isequal(rows.round, (1:100)'), ...
        'Each Fixed strategy must contain the original 100 rounds.');
    assert(all(rows.group_size == 3) && all(rows.test_loss_n == 3) ...
        && all(rows.system_time_n == 3), ...
        'Every plotted point must contain all three seeds.');
    assert(all(isfinite(rows.system_time_mean)) ...
        && all(isfinite(rows.test_loss_mean)) ...
        && all(isfinite(rows.test_loss_std)) ...
        && all(rows.test_loss_std >= 0) ...
        && all(diff(rows.system_time_mean) > 0), ...
        'Invalid mean, sample SD, or cumulative latency.');
    series{j} = rows;
    k = find(rows.test_loss_mean <= threshold, 1, 'first');
    assert(~isempty(k), 'Each Fixed mean curve must reach loss 1.40.');
    firstPoints(j, :) = [rows.round(k), ...
        rows.system_time_mean(k), rows.test_loss_mean(k)];
end

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'centimeters', 'Position', [2, 2, 8.9, 8.0], ...
    'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
ax = axes(fig, 'Position', [0.15, 0.16, 0.81, 0.70]);
hold(ax, 'on');
styleAxes(ax, 8);
xlim(ax, [0, 200]);
ylim(ax, [1.15, 2.35]);
xticks(ax, 0:40:200);
yticks(ax, 1.2:0.2:2.2);
ax.YGrid = 'on';
ax.GridColor = [0.7, 0.7, 0.7];
ax.GridAlpha = 0.28;
ax.GridLineStyle = '-';

% All bands are drawn before the curves, so no band hides a mean line.
for j = 1:5
    rows = series{j};
    drawBand(ax, rows, colors(j, :));
end
yline(ax, threshold, ':', 'Color', [0.42, 0.42, 0.42], ...
    'LineWidth', 0.75, 'HandleVisibility', 'off');
text(ax, 5, 1.425, 'Loss = 1.40', 'FontName', 'Times New Roman', ...
    'FontSize', 7, 'Color', [0.3, 0.3, 0.3], ...
    'VerticalAlignment', 'bottom');

lines = gobjects(1, 5);
for j = 1:5
    rows = series{j};
    lines(j) = plot(ax, rows.system_time_mean, rows.test_loss_mean, ...
        'Color', colors(j, :), 'LineStyle', styles{j}, 'LineWidth', 1.2);
    if j == 5
        % A sparse diamond distinguishes the fifth style in MATLAB.
        set(lines(j), 'Marker', 'd', 'MarkerIndices', 10:10:100, ...
            'MarkerSize', 2.8, 'MarkerFaceColor', 'white');
    end
    plot(ax, firstPoints(j, 2), firstPoints(j, 3), 'o', ...
        'Color', colors(j, :), 'MarkerFaceColor', colors(j, :), ...
        'MarkerSize', 3.5, 'LineWidth', 0.6, 'HandleVisibility', 'off');
end
xlabel(ax, 'Cumulative system latency (s)', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5);
ylabel(ax, 'Test loss', 'FontName', 'Times New Roman', 'FontSize', 8.5);
lgd = legend(ax, lines, labels, 'NumColumns', 3, ...
    'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', 'Times New Roman', 'FontSize', 7.5, 'Interpreter', 'tex');
lgd.Units = 'normalized';
lgd.Position = [0.16, 0.90, 0.79, 0.075];
lgd.ItemTokenSize = [17, 8];

% The inset reuses the same observations and SD; no interpolation is used.
zoom = axes(fig, 'Position', [0.50, 0.59, 0.44, 0.26]);
hold(zoom, 'on');
styleAxes(zoom, 6.5);
zoom.Box = 'on';
for j = 1:5
    drawBand(zoom, series{j}, colors(j, :));
end
yline(zoom, threshold, ':', 'Color', [0.42, 0.42, 0.42], ...
    'LineWidth', 0.65, 'HandleVisibility', 'off');
for j = 1:5
    rows = series{j};
    h = plot(zoom, rows.system_time_mean, rows.test_loss_mean, ...
        'Color', colors(j, :), 'LineStyle', styles{j}, 'LineWidth', 1.05);
    if j == 5
        set(h, 'Marker', 'd', 'MarkerIndices', 10:10:100, ...
            'MarkerSize', 2.2, 'MarkerFaceColor', 'white');
    end
    plot(zoom, firstPoints(j, 2), firstPoints(j, 3), 'o', ...
        'Color', colors(j, :), 'MarkerFaceColor', colors(j, :), ...
        'MarkerSize', 3, 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% Candidate identities are shown by the shared legend to avoid inset occlusion.
xlim(zoom, [100, 155]);
ylim(zoom, [1.36, 1.44]);
xticks(zoom, [105, 120, 135, 150]);
yticks(zoom, [1.37, 1.40, 1.43]);
ytickformat(zoom, '%.2f');
title(zoom, 'Near loss = 1.40', 'FontName', 'Times New Roman', ...
    'FontSize', 7, 'FontWeight', 'normal');

% Link the source rectangle to the enlarged region using actual axis limits.
rectangle(ax, 'Position', [100, 1.36, 55, 0.08], ...
    'EdgeColor', [0.55, 0.55, 0.55], 'LineWidth', 0.55, 'LineStyle', '--');
ax.Units = 'normalized';
p = ax.Position;
sourceLeft = p(1) + (100 / 200) * p(3);
sourceRight = p(1) + (155 / 200) * p(3);
sourceTop = p(2) + ((1.44 - 1.15) / (2.35 - 1.15)) * p(4);
q = zoom.Position;
annotation(fig, 'line', [sourceLeft, q(1)], [sourceTop, q(2)], ...
    'Color', [0.65, 0.65, 0.65], 'LineWidth', 0.5, 'LineStyle', ':');
annotation(fig, 'line', [sourceRight, q(1) + q(3)], [sourceTop, q(2)], ...
    'Color', [0.65, 0.65, 0.65], 'LineWidth', 0.5, 'LineStyle', ':');

drawnow;
stem = fullfile(previewDir, 'experiment_fixed_loss_matlab_inset_preview');
exportgraphics(fig, [stem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
exportgraphics(fig, [stem, '.png'], 'Resolution', 600, ...
    'BackgroundColor', 'white');
savefig(fig, [stem, '.fig']);

if publishToPaper
    backupDir = fullfile(previewDir, 'original_figure_backup');
    if ~isfolder(backupDir)
        mkdir(backupDir);
    end
    paperStem = fullfile(figuresDir, 'experiment_fixed_loss_vs_cumulative_T');
    for extension = {'.pdf', '.png'}
        suffix = extension{1};
        paperPath = [paperStem, suffix];
        [~, paperName, paperExtension] = fileparts(paperPath);
        backupPath = fullfile(backupDir, [paperName, paperExtension]);
        if isfile(paperPath) && ~isfile(backupPath)
            [ok, message] = copyfile(paperPath, backupPath);
            assert(ok, 'Original figure backup failed: %s', message);
        end
        [ok, message] = copyfile([stem, suffix], paperPath, 'f');
        assert(ok, 'Paper figure replacement failed: %s', message);
        fprintf('PAPER_FIGURE=%s\n', paperPath);
    end
    fprintf('ORIGINAL_FIGURE_BACKUP=%s\n', backupDir);
end

fprintf('MATLAB_RELEASE=%s\n', version('-release'));
fprintf('INPUT=%s\n', inputPath);
fprintf('FIGURE_PDF=%s.pdf\n', stem);
fprintf('FIGURE_PNG=%s.png\n', stem);
fprintf('FIGURE_EDITABLE=%s.fig\n', stem);
fprintf('First recorded points on mean curves with loss <= 1.40:\n');
for j = 1:5
    fprintf('%s round=%d T=%.9f loss=%.9f\n', strategies(j), ...
        firstPoints(j, 1), firstPoints(j, 2), firstPoints(j, 3));
end
end

function drawBand(ax, rows, color)
t = rows.system_time_mean;
loss = rows.test_loss_mean;
sd = rows.test_loss_std;
fill(ax, [t; flipud(t)], [loss - sd; flipud(loss + sd)], color, ...
    'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function styleAxes(ax, fontSize)
set(ax, 'FontName', 'Times New Roman', 'FontSize', fontSize, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.018, 0.018], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
end
