function plot_selection_switching_matlab(publishToPaper)
% Accessible single-column selection strips and switching statistics.
% MATLAB R2022b. Color is redundant with five vector fill patterns.
% Inputs are existing CSVs and raw metrics; no training or input writes.
% Candidate changes count adjacent-round differences, including round 5 to 6.
% The two summary panels show means and sample SD across all three seeds.

if nargin == 0
    publishToPaper = false;
end
assert(islogical(publishToPaper) && isscalar(publishToPaper), ...
    'publishToPaper must be a scalar logical value.');
previewDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(previewDir);
projectDir = fileparts(fileparts(figuresDir));
aggregateDir = fullfile(figuresDir, 'three_seed_aggregate');
campaignDir = fullfile(projectDir, 'out', ...
    'fmnist_joint_jt_100r_3seed_20261022');
sequence = readtable(fullfile(aggregateDir, ...
    'selection_sequences_by_seed.csv'), 'TextType', 'string');
perSeed = readtable(fullfile(aggregateDir, 'switching_per_seed.csv'), ...
    'TextType', 'string');
stats = readtable(fullfile(aggregateDir, 'switching_mean_sd.csv'), ...
    'TextType', 'string');
strategies = ["greedy_lambda025", "greedy_lambda050", ...
    "greedy_lambda075", "hysteresis_lambda050", ...
    "periodic_lambda050", "max_J", "min_T"];
labels = {'Greedy \lambda=0.25', 'Greedy \lambda=0.50', ...
    'Greedy \lambda=0.75', 'Hysteresis', ...
    'Periodic', 'Maximum J', 'Minimum T'};
seeds = [42, 43, 44];
candidateNames = "P" + string(0:4);
% Keep the original hues as pale backgrounds; patterns carry identity too.
candidateColors = [0.78, 0.78, 0.78; 0.97, 0.87, 0.72; ...
    0.76, 0.87, 0.95; 0.80, 0.93, 0.86; 0.95, 0.85, 0.91];
codes = zeros(7, 3, 100);
means = zeros(7, 2);
spreads = zeros(7, 2);
assert(height(sequence) == 2100 && height(perSeed) == 21 ...
    && height(stats) == 7, 'Unexpected input coverage.');

for j = 1:7
    counts = zeros(3, 2);
    for k = 1:3
        rows = sortrows(sequence(sequence.strategy == strategies(j) ...
            & sequence.seed == seeds(k), :), 'round');
        assert(height(rows) == 100 && isequal(rows.round, (1:100)'), ...
            'Every strategy/seed must contain all 100 rounds.');
        [valid, candidateCode] = ismember(rows.candidate, candidateNames);
        assert(all(valid) && all(candidateCode(1:5) == 1), ...
            'Invalid candidate or common P0 prefix.');
        codes(j, k, :) = candidateCode;
        counts(k, 1) = nnz(diff(candidateCode));
        rawPath = fullfile(campaignDir, sprintf('seed_%d', seeds(k)), ...
            strategies(j), 'round_metrics.csv');
        raw = sortrows(readtable(rawPath, 'TextType', 'string'), 'round');
        assert(height(raw) == 100 && isequal(raw.round, (1:100)'), ...
            'Raw metrics must contain the original 100 rounds.');
        assert(all(isfinite(raw.mode_changes)) ...
            && all(raw.mode_changes >= 0) ...
            && all(raw.mode_changes == fix(raw.mode_changes)), ...
            'Client mode changes must be observed nonnegative counts.');
        counts(k, 2) = sum(raw.mode_changes);
        saved = perSeed(perSeed.strategy == strategies(j) ...
            & perSeed.seed == seeds(k), :);
        assert(height(saved) == 1 ...
            && saved.candidate_switches == counts(k, 1) ...
            && saved.client_mode_reassignments == counts(k, 2), ...
            'Per-seed switching counts do not reconcile.');
    end
    means(j, :) = mean(counts, 1);
    spreads(j, :) = std(counts, 0, 1);
    saved = stats(stats.strategy == strategies(j), :);
    assert(height(saved) == 1 && saved.group_size == 3 ...
        && saved.candidate_switches_n == 3 ...
        && saved.client_mode_reassignments_n == 3, ...
        'Each summary must contain all three seeds.');
    savedValues = [saved.candidate_switches_mean, ...
        saved.client_mode_reassignments_mean; ...
        saved.candidate_switches_std, saved.client_mode_reassignments_std];
    assert(max(abs(savedValues - [means(j, :); spreads(j, :)]), [], 'all') ...
        < 1e-7, 'Aggregate switching statistics do not reconcile.');
end

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'centimeters', 'Position', [2, 2, 8.9, 14.2], ...
    'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
centers = 2:4:26;
yLimits = [0.3, 27.7];
heat = axes(fig, 'Position', [0.35, 0.48, 0.61, 0.43]);
hold(heat, 'on');
styleAxes(heat);
set(heat, 'YDir', 'reverse', 'YColor', 'none', 'YTick', [], ...
    'Clipping', 'off');
xlim(heat, [0.5, 100.5]);
ylim(heat, yLimits);
xticks(heat, [1, 20, 40, 60, 80, 100]);
xlabel(heat, 'Round', 'FontName', 'Times New Roman', 'FontSize', 8.5);
title(heat, '(a) Candidate selection', 'FontName', 'Times New Roman', ...
    'FontSize', 8.5, 'FontWeight', 'normal', 'HorizontalAlignment', 'left');
heat.Title.Units = 'normalized';
heat.Title.Position = [0, 1.025, 0];

% Draw contiguous observed runs as vector rectangles, not a raster heatmap.
for j = 1:7
    groupStart = (j - 1) * 4;
    for k = 1:3
        observed = reshape(codes(j, k, :), 1, 100);
        boundaries = [1, find(diff(observed) ~= 0) + 1, 101];
        y = groupStart + k;
        for run = 1:(numel(boundaries) - 1)
            firstRound = boundaries(run);
            runLength = boundaries(run + 1) - firstRound;
            drawCandidatePattern(heat, [firstRound - 0.5, ...
                y - 0.42, runLength, 0.84], observed(firstRound), ...
                candidateColors(observed(firstRound), :));
        end
        text(heat, -2.3, y, string(seeds(k)), ...
            'FontName', 'Times New Roman', 'FontSize', 6.5, ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
            'Color', [0.32, 0.32, 0.32], 'Clipping', 'off');
    end
    rectangle(heat, 'Position', [0.5, groupStart + 0.55, 100, 2.9], ...
        'EdgeColor', [0.85, 0.85, 0.85], 'LineWidth', 0.35);
    text(heat, -10, centers(j), labels{j}, 'Interpreter', 'tex', ...
        'FontName', 'Times New Roman', 'FontSize', 7.5, ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
        'Color', [0.12, 0.12, 0.12], 'Clipping', 'off');
end
xline(heat, 5.5, '--', 'Color', [0.25, 0.25, 0.25], ...
    'LineWidth', 0.65, 'HandleVisibility', 'off');
text(heat, -2.3, -0.15, 'Seed', 'FontName', 'Times New Roman', ...
    'FontSize', 6.5, 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Color', [0.32, 0.32, 0.32], ...
    'Clipping', 'off');
% Custom vector legend repeats exactly the patterns in the selection strips.
key = axes(fig, 'Position', [0.33, 0.95, 0.64, 0.035]);
hold(key, 'on');
xlim(key, [0, 5]);
ylim(key, [0, 1]);
set(key, 'Visible', 'off', 'YDir', 'reverse');
for j = 1:5
    drawCandidatePattern(key, [j - 0.97, 0.18, 0.42, 0.60], ...
        j, candidateColors(j, :));
    text(key, j - 0.45, 0.48, sprintf('P_%d', j - 1), ...
        'FontName', 'Times New Roman', 'FontSize', 7.5, ...
        'Interpreter', 'tex', 'VerticalAlignment', 'middle');
end

summaryPositions = {[0.33, 0.06, 0.24, 0.30], ...
    [0.68, 0.06, 0.28, 0.30]};
% A small unticked left gutter keeps near-zero markers fully visible.
% It does not add or alter any count; tick labels still start at zero.
summaryLimits = {[-4, 80], [-100, 2000]};
summaryTicks = {[0, 40, 80], [0, 1000, 2000]};
summaryTitles = {{'(b) Candidate', 'changes'}, ...
    {'(c) Client mode', 'changes'}};
for metric = 1:2
    ax = axes(fig, 'Position', summaryPositions{metric});
    hold(ax, 'on');
    styleAxes(ax);
    set(ax, 'YDir', 'reverse', 'YColor', 'none', 'YTick', []);
    ylim(ax, yLimits);
    xlim(ax, summaryLimits{metric});
    xticks(ax, summaryTicks{metric});
    ax.XGrid = 'on';
    ax.GridColor = [0.75, 0.75, 0.75];
    ax.GridAlpha = 0.3;
    xlabel(ax, 'Count', 'FontName', 'Times New Roman', 'FontSize', 8.5);
    title(ax, summaryTitles{metric}, 'FontName', 'Times New Roman', ...
        'FontSize', 8, 'FontWeight', 'normal');
    for j = 1:7
        value = means(j, metric);
        sd = spreads(j, metric);
        y = centers(j) + 0.48;
        color = [0.25, 0.25, 0.25];
        plot(ax, [value - sd, value + sd], [y, y], '-', ...
            'Color', color, 'LineWidth', 0.9);
        plot(ax, [value - sd, value - sd], y + [-0.16, 0.16], '-', ...
            'Color', color, 'LineWidth', 0.75);
        plot(ax, [value + sd, value + sd], y + [-0.16, 0.16], '-', ...
            'Color', color, 'LineWidth', 0.75);
        plot(ax, value, y, 'o', 'MarkerSize', 3.5, ...
            'MarkerFaceColor', color, 'MarkerEdgeColor', 'white', ...
            'LineWidth', 0.45);
        text(ax, mean(summaryLimits{metric}), centers(j) - 0.70, ...
            sprintf('%.1f \\pm %.1f', value, sd), 'Interpreter', 'tex', ...
            'FontName', 'Times New Roman', 'FontSize', 7.2, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', [0.18, 0.18, 0.18]);
        if metric == 1
            text(ax, -16, centers(j), labels{j}, 'Interpreter', 'tex', ...
                'FontName', 'Times New Roman', 'FontSize', 7.2, ...
                'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'middle', 'Clipping', 'off');
        end
    end
end
% Keep a small outer gutter when exportgraphics crops to visible objects.
annotation(fig, 'rectangle', [0.005, 0.005, 0.99, 0.99], ...
    'Color', 'white', 'LineWidth', 0.1);

drawnow;
stem = fullfile(previewDir, 'experiment_selection_switching_matlab');
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
    for extension = {'.pdf', '.png'}
        suffix = extension{1};
        paperPath = fullfile(figuresDir, ['experiment_selection_switching', suffix]);
        backupPath = fullfile(backupDir, ...
            ['experiment_selection_switching_before_patterns_20261026', suffix]);
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
fprintf('VALIDATED=2100 categorical selections; 21 per-seed switching counts\n');
fprintf('STATISTICS=three-seed mean and sample SD; no candidate-ID averaging\n');
fprintf('LAYOUT=single column; color plus plain/slash/horizontal/vertical/dot patterns\n');
fprintf('FIGURE_PDF=%s.pdf\n', stem);
fprintf('FIGURE_PNG=%s.png\n', stem);
fprintf('FIGURE_EDITABLE=%s.fig\n', stem);
end

function drawCandidatePattern(ax, position, candidate, faceColor)
% Draw in physical units so the legend and plot have the same hatch pitch.
% Every run is centered on a motif, including a one-round-wide run.
rectangle(ax, 'Position', position, 'FaceColor', faceColor, ...
    'EdgeColor', [0.68, 0.68, 0.68], 'LineWidth', 0.15);
if candidate == 1
    return
end
fig = ancestor(ax, 'figure');
figPosition = fig.Position;
scaleX = ax.Position(3) * figPosition(3) / diff(ax.XLim);
scaleY = ax.Position(4) * figPosition(4) / diff(ax.YLim);
width = position(3) * scaleX;
height = position(4) * scaleY;
ink = [0.12, 0.12, 0.12];
lineWidth = 0.28;
pitch = 0.065; % cm; about 0.65 mm in the full-size single-column PDF
switch candidate
    case 2
        center = (height - width) / 2;
        extent = ceil((height + width) / pitch) + 1;
        for intercept = center + (-extent:extent) * pitch
            left = max(0, -intercept);
            right = min(width, height - intercept);
            if right > left
                x = position(1) + [left, right] / scaleX;
                y = position(2) + ([left, right] + intercept) / scaleY;
                plot(ax, x, y, '-', 'Color', ink, 'LineWidth', lineWidth, ...
                    'HandleVisibility', 'off');
            end
        end
    case 3
        for y = centeredMotifs(height, pitch, 0.008)
            plot(ax, position(1) + [0, position(3)], ...
                position(2) + [y, y] / scaleY, '-', ...
                'Color', ink, 'LineWidth', lineWidth, ...
                'HandleVisibility', 'off');
        end
    case 4
        for x = centeredMotifs(width, pitch, 0.008)
            plot(ax, position(1) + [x, x] / scaleX, ...
                position(2) + [0, position(4)], '-', ...
                'Color', ink, 'LineWidth', lineWidth, ...
                'HandleVisibility', 'off');
        end
    case 5
        diameter = 0.024;
        for x = centeredMotifs(width, pitch, diameter / 2)
            for y = centeredMotifs(height, pitch, diameter / 2)
                rectangle(ax, 'Position', [ ...
                    position(1) + (x - diameter / 2) / scaleX, ...
                    position(2) + (y - diameter / 2) / scaleY, ...
                    diameter / scaleX, diameter / scaleY], ...
                    'Curvature', [1, 1], 'FaceColor', ink, 'EdgeColor', 'none');
            end
        end
end
end

function points = centeredMotifs(span, pitch, margin)
count = ceil(span / pitch) + 1;
points = span / 2 + (-count:count) * pitch;
points = points(points >= margin & points <= span - margin);
end

function styleAxes(ax)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 7.5, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.018, 0.018], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
end
