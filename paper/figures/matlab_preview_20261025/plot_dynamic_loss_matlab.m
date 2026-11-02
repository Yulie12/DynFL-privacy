function plot_dynamic_loss_matlab(publishToPaper, layout)
% Restyle the dynamic Loss-T comparison using Figure 4's MATLAB format.
% Read-only inputs: existing aggregate CSV and 21 original run CSVs.
% Every mean and sample SD is checked against the original three seeds.
% No smoothing, interpolation, training, or LaTeX modification is performed.
% MATLAB R2022b. The default 'combined' layout retains the two-panel figure.
% plot_dynamic_loss_matlab(true, 'split') publishes two independent figures,
% without overwriting the existing combined paper PDF/PNG.

if nargin == 0
    publishToPaper = false;
end
assert(islogical(publishToPaper) && isscalar(publishToPaper), ...
    'publishToPaper must be a scalar logical value.');
if nargin < 2
    layout = 'combined';
end
assert(any(strcmp(layout, {'combined', 'split'})), ...
    'layout must be combined or split.');
isSplit = strcmp(layout, 'split');

previewDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(previewDir);
projectDir = fileparts(fileparts(figuresDir));
campaignDir = fullfile(projectDir, 'out', ...
    'fmnist_joint_jt_100r_3seed_20261022');
inputPath = fullfile(figuresDir, 'three_seed_aggregate', ...
    'round_metrics_mean_sd.csv');
stats = readtable(inputPath, 'TextType', 'string');
strategies = ["greedy_lambda025", "greedy_lambda050", ...
    "greedy_lambda075", "max_J", "min_T", ...
    "hysteresis_lambda050", "periodic_lambda050"];
labels = {'Greedy \lambda=0.25', 'Greedy \lambda=0.50', ...
    'Greedy \lambda=0.75', 'Maximum J', 'Minimum T', ...
    'Hysteresis', 'Periodic'};

% Same palette, fonts, line widths and band opacity as the Fixed figure.
palette = [77, 77, 77; 213, 94, 0; 0, 114, 178; ...
    0, 158, 115; 204, 121, 167] / 255;
colors = palette([1, 2, 3, 3, 4, 5, 1], :);
styles = {'-', '--', '-.', '-.', ':', '-', '--'};
markers = {'none', 'none', 'none', 'none', 'none', 'd', 's'};
series = cell(1, numel(strategies));

for j = 1:numel(strategies)
    rows = sortrows(stats(stats.strategy == strategies(j), :), 'round');
    assert(height(rows) == 100 && isequal(rows.round, (1:100)'), ...
        'Each dynamic strategy must contain the original 100 rounds.');
    assert(all(rows.group_size == 3) && all(rows.test_loss_n == 3) ...
        && all(rows.system_time_n == 3), ...
        'Every plotted point must contain all three seeds.');
    assert(all(isfinite(rows.system_time_mean)) ...
        && all(isfinite(rows.test_loss_mean)) ...
        && all(isfinite(rows.test_loss_std)) ...
        && all(rows.test_loss_std >= 0) ...
        && all(diff(rows.system_time_mean) > 0), ...
        'Invalid mean, sample SD, or cumulative latency.');

    % Independently check the plotted values against unmodified raw CSVs.
    rawLoss = zeros(100, 3);
    rawTime = zeros(100, 3);
    for k = 1:3
        rawPath = fullfile(campaignDir, sprintf('seed_%d', 41 + k), ...
            strategies(j), 'round_metrics.csv');
        raw = sortrows(readtable(rawPath, 'TextType', 'string'), 'round');
        assert(height(raw) == 100 && isequal(raw.round, (1:100)'), ...
            'Raw run must contain all 100 original rounds: %s', rawPath);
        rawLoss(:, k) = raw.test_loss;
        rawTime(:, k) = raw.system_time;
    end
    assert(max(abs(rows.test_loss_mean - mean(rawLoss, 2))) < 1e-9 ...
        && max(abs(rows.test_loss_std - std(rawLoss, 0, 2))) < 1e-9 ...
        && max(abs(rows.system_time_mean - mean(rawTime, 2))) < 1e-8 ...
        && max(abs(rows.system_time_std - std(rawTime, 0, 2))) < 1e-8, ...
        'Aggregate means/sample SDs differ from the raw runs: %s', ...
        strategies(j));
    series{j} = rows;
end

if ~isSplit
    fig = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'centimeters', 'Position', [2, 2, 17.8, 8.0], ...
        'InvertHardcopy', 'off');
    cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
end
panels = {[1, 2, 3], [2, 4, 5, 6, 7]};
panelLabels = {'(a)', '(b)'};
splitNames = {'experiment_greedy_weights_loss_vs_cumulative_T', ...
    'experiment_objectives_switching_loss_vs_cumulative_T'};
threshold = 1.30;
zoomX = [130, 190];
zoomY = [1.25, 1.35];

for panel = 1:2
    if isSplit
        fig = figure('Visible', 'off', 'Color', 'white', ...
            'Units', 'centimeters', 'Position', [2, 2, 8.9, 8.0], ...
            'InvertHardcopy', 'off');
        cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
        panelWidth = 1;
        offset = 0;
    else
        panelWidth = 0.5;
        offset = (panel - 1) * panelWidth;
    end
    ax = axes(fig, 'Position', ...
        [0.15 * panelWidth + offset, 0.16, 0.81 * panelWidth, 0.70]);
    hold(ax, 'on');
    styleAxes(ax, 8);
    xlim(ax, [0, 200]);
    ylim(ax, [1.15, 2.35]);
    xticks(ax, 0:40:200);
    yticks(ax, 1.2:0.2:2.2);
    ax.YGrid = 'on';
    ax.GridColor = [0.7, 0.7, 0.7];
    ax.GridAlpha = 0.28;
    members = panels{panel};

    % Draw all bands first; no band's face can obscure a mean curve.
    for j = members
        drawBand(ax, series{j}, colors(j, :));
    end
    yline(ax, threshold, ':', 'Color', [0.42, 0.42, 0.42], ...
        'LineWidth', 0.75, 'HandleVisibility', 'off');
    text(ax, 5, 1.325, 'Loss = 1.30', 'FontName', 'Times New Roman', ...
        'FontSize', 7, 'Color', [0.3, 0.3, 0.3], ...
        'VerticalAlignment', 'bottom');
    lines = gobjects(1, numel(members));
    for k = 1:numel(members)
        j = members(k);
        lines(k) = drawCurve(ax, series{j}, colors(j, :), ...
            styles{j}, markers{j}, 1.2, 2.8);
        assert(isequal(lines(k).XData(:), series{j}.system_time_mean) ...
            && isequal(lines(k).YData(:), series{j}.test_loss_mean), ...
            'A plotted mean curve does not match the source data.');
    end
    xlabel(ax, 'Cumulative system latency (s)', ...
        'FontName', 'Times New Roman', 'FontSize', 8.5);
    ylabel(ax, 'Test loss', 'FontName', 'Times New Roman', 'FontSize', 8.5);
    lgd = legend(ax, lines, labels(members), 'NumColumns', 3, ...
        'Orientation', 'horizontal', 'Box', 'off', ...
        'FontName', 'Times New Roman', 'FontSize', 7.5, 'Interpreter', 'tex');
    lgd.Units = 'normalized';
    lgd.Position = ...
        [0.154 * panelWidth + offset, 0.90, 0.804 * panelWidth, 0.075];
    lgd.ItemTokenSize = [17, 8];
    if ~isSplit
        annotation(fig, 'textbox', [0.017 + offset, 0.905, 0.04, 0.06], ...
            'String', panelLabels{panel}, 'FontName', 'Times New Roman', ...
            'FontSize', 8.5, 'LineStyle', 'none', 'Margin', 0, ...
            'VerticalAlignment', 'middle');
    end

    % Reuse the same round-aligned observations and sample SD in the inset.
    % No crossing circles: mean-curve crossings are not the mean of each
    % seed's first crossing times reported in the surrounding manuscript.
    zoom = axes(fig, 'Position', ...
        [0.50 * panelWidth + offset, 0.59, 0.44 * panelWidth, 0.26]);
    hold(zoom, 'on');
    styleAxes(zoom, 6.5);
    zoom.Box = 'on';
    xlim(zoom, zoomX);
    ylim(zoom, zoomY);
    xticks(zoom, [140, 160, 180]);
    yticks(zoom, [1.26, 1.30, 1.34]);
    ytickformat(zoom, '%.2f');
    for j = members
        drawBand(zoom, series{j}, colors(j, :));
    end
    yline(zoom, threshold, ':', 'Color', [0.42, 0.42, 0.42], ...
        'LineWidth', 0.65, 'HandleVisibility', 'off');
    for j = members
        drawCurve(zoom, series{j}, colors(j, :), ...
            styles{j}, markers{j}, 1.05, 2.2);
    end
    title(zoom, 'Near loss = 1.30', 'FontName', 'Times New Roman', ...
        'FontSize', 7, 'FontWeight', 'normal');

    rectangle(ax, 'Position', [zoomX(1), zoomY(1), ...
        diff(zoomX), diff(zoomY)], 'EdgeColor', [0.55, 0.55, 0.55], ...
        'LineWidth', 0.55, 'LineStyle', '--');
    p = ax.Position;
    q = zoom.Position;
    sourceX = p(1) + zoomX / 200 * p(3);
    sourceTop = p(2) + (zoomY(2) - 1.15) / (2.35 - 1.15) * p(4);
    annotation(fig, 'line', [sourceX(1), q(1)], [sourceTop, q(2)], ...
        'Color', [0.65, 0.65, 0.65], 'LineWidth', 0.5, 'LineStyle', ':');
    annotation(fig, 'line', [sourceX(2), q(1) + q(3)], [sourceTop, q(2)], ...
        'Color', [0.65, 0.65, 0.65], 'LineWidth', 0.5, 'LineStyle', ':');

    if isSplit
        drawnow;
        splitStem = fullfile(previewDir, splitNames{panel});
        exportgraphics(fig, [splitStem, '.pdf'], 'ContentType', 'vector', ...
            'BackgroundColor', 'white');
        exportgraphics(fig, [splitStem, '.png'], 'Resolution', 600, ...
            'BackgroundColor', 'white');
        savefig(fig, [splitStem, '.fig']);
        if publishToPaper
            for extension = {'.pdf', '.png'}
                suffix = extension{1};
                paperPath = fullfile(figuresDir, [splitNames{panel}, suffix]);
                [ok, message] = copyfile([splitStem, suffix], paperPath, 'f');
                assert(ok, 'Split paper figure export failed: %s', message);
                fprintf('PAPER_FIGURE=%s\n', paperPath);
            end
        end
        fprintf('FIGURE_EDITABLE=%s.fig\n', splitStem);
        clear cleanupFigure
    end
end

if ~isSplit
drawnow;
stem = fullfile(previewDir, 'experiment_dynamic_loss_matlab');
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
    paperStem = fullfile(figuresDir, 'experiment_dynamic_loss_vs_cumulative_T');
    for extension = {'.pdf', '.png'}
        suffix = extension{1};
        paperPath = [paperStem, suffix];
        backupPath = fullfile(backupDir, ...
            ['experiment_dynamic_loss_before_matlab_20261025', suffix]);
        if isfile(paperPath) && ~isfile(backupPath)
            [ok, message] = copyfile(paperPath, backupPath);
            assert(ok, 'Original dynamic figure backup failed: %s', message);
        end
        [ok, message] = copyfile([stem, suffix], paperPath, 'f');
        assert(ok, 'Dynamic paper figure replacement failed: %s', message);
        fprintf('PAPER_FIGURE=%s\n', paperPath);
    end
    fprintf('ORIGINAL_FIGURE_BACKUP=%s\n', backupDir);
end
fprintf('VALIDATED=21 raw runs; 700 round-aligned aggregate points\n');
fprintf('STATISTICS=three-seed means and sample SD (n-1 denominator)\n');
fprintf('MATLAB_RELEASE=%s\n', version('-release'));
fprintf('INPUT=%s\n', inputPath);
fprintf('FIGURE_PDF=%s.pdf\n', stem);
fprintf('FIGURE_PNG=%s.png\n', stem);
fprintf('FIGURE_EDITABLE=%s.fig\n', stem);
else
    fprintf('VALIDATED=21 raw runs; 700 round-aligned aggregate points\n');
    fprintf('STATISTICS=three-seed means and sample SD (n-1 denominator)\n');
    fprintf('SPLIT_FIGURES=2 single-column figures; combined figure unchanged\n');
    fprintf('MATLAB_RELEASE=%s\n', version('-release'));
    fprintf('INPUT=%s\n', inputPath);
end
end

function drawBand(ax, rows, color)
t = rows.system_time_mean;
loss = rows.test_loss_mean;
sd = rows.test_loss_std;
fill(ax, [t; flipud(t)], [loss - sd; flipud(loss + sd)], color, ...
    'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end

function h = drawCurve(ax, rows, color, lineStyle, marker, width, markerSize)
h = plot(ax, rows.system_time_mean, rows.test_loss_mean, ...
    'Color', color, 'LineStyle', lineStyle, 'LineWidth', width);
if ~strcmp(marker, 'none')
    set(h, 'Marker', marker, 'MarkerIndices', 10:10:100, ...
        'MarkerSize', markerSize, 'MarkerFaceColor', 'white');
end
end

function styleAxes(ax, fontSize)
set(ax, 'FontName', 'Times New Roman', 'FontSize', fontSize, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.018, 0.018], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
end
