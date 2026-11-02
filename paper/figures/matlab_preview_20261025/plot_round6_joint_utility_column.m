function plot_round6_joint_utility_column()
% Stacked J/T contributions to first-decision utility, from raw JSON.
% Reads only the nine completed formal candidate records. No training,
% simulator, configuration, raw data, or manuscript files are modified.
% Output: the existing publication PDF/PNG and one editable MATLAB figure.

scriptDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(scriptDir);
repoDir = fileparts(fileparts(figuresDir));
campaignDir = fullfile(repoDir, 'out', 'fmnist_joint_jt_100r_3seed_20261022');
seeds = [42, 43, 44];
strategies = {'greedy_lambda025', 'greedy_lambda050', 'greedy_lambda075'};
lambdas = [0.25, 0.50, 0.75];
names = {'P0', 'P1', 'P2', 'P3', 'P4'};
labels = {'P_0', 'P_1', 'P_2', 'P_3', 'P_4'};
termColors = [0, 114, 178; 190, 190, 190] / 255;

utilities = zeros(3, 5, 3);
jContributions = zeros(3, 5, 3);
tContributions = zeros(3, 5, 3);
selectedIndices = zeros(3, 3);
scores = zeros(3, 5, 4);
for s = 1:numel(seeds)
    referenceHistory = '';
    for w = 1:numel(strategies)
        source = fullfile(campaignDir, sprintf('seed_%d', seeds(s)), ...
            strategies{w}, 'candidates_round_005.json');
        record = jsondecode(fileread(source));
        assert(record.decision_after_round == 5 && record.applied_round == 6);
        assert(abs(record.lambda - lambdas(w)) < 1e-12);
        assert(numel(record.candidates) == 5 ...
            && record.evaluated_profiles == 5 ...
            && record.feasible_combinations == 5);
        assert(strcmp(record.tie_rule, 'P0<P1<P2<P3<P4'));
        if w == 1
            referenceHistory = record.history_id;
        else
            assert(strcmp(record.history_id, referenceHistory), ...
                'The three weights must share the initial history within each seed.');
        end
        currentScores = zeros(5, 4);
        for k = 1:numel(names)
            candidate = record.candidates(strcmp({record.candidates.name}, names{k}));
            assert(isscalar(candidate) && candidate.feasible ...
                && strcmp(candidate.status, 'ready'));
            currentScores(k, :) = [candidate.J, candidate.T, ...
                candidate.J_norm, candidate.T_norm];
            utilities(s, k, w) = candidate.utility;
            jContributions(s, k, w) = lambdas(w) * candidate.J_norm;
            tContributions(s, k, w) = (1 - lambdas(w)) * candidate.T_norm;
            expectedUtility = jContributions(s, k, w) + tContributions(s, k, w);
            assert(abs(candidate.utility - expectedUtility) < 1e-12);
            assert(candidate.selected == strcmp(names{k}, record.selected_candidate));
        end
        if w == 1
            scores(s, :, :) = reshape(currentScores, [1, 5, 4]);
        else
            originalScores = reshape(scores(s, :, :), [5, 4]);
            assert(max(abs(currentScores(:) - originalScores(:))) < 1e-12);
        end
        rawJ = currentScores(:, 1);
        rawT = currentScores(:, 2);
        expectedJ = (rawJ - min(rawJ)) / (max(rawJ) - min(rawJ));
        expectedT = (max(rawT) - rawT) / (max(rawT) - min(rawT));
        assert(max(abs(expectedJ - currentScores(:, 3))) < 1e-12);
        assert(max(abs(expectedT - currentScores(:, 4))) < 1e-12);
        selected = find(strcmp(names, record.selected_candidate));
        assert(isscalar(selected));
        selectedIndices(s, w) = selected;
        candidateUtility = reshape(utilities(s, :, w), [5, 1]);
        [~, utilityOrder] = sortrows([-candidateUtility, (1:5)'], [1, 2]);
        assert(selected == utilityOrder(1), ...
            'The recorded choice must maximize utility with the declared index rule.');
    end
end
assert(all(isfinite(utilities(:))) ...
    && all(utilities(:) >= 0) && all(utilities(:) <= 1));
assert(isequal(selectedIndices, [4, 3, 3; 2, 3, 3; 1, 3, 3]));
assert(utilities(1, 3, 2) == utilities(1, 4, 2), ...
    'The saved seed-42, lambda-0.50 tie must remain a tie.');

% Compute the sample SD only after the per-seed normalized utilities exist.
utilityMean = reshape(mean(utilities, 1), [5, 3]);
utilitySD = reshape(std(utilities, 0, 1), [5, 3]);
jContributionMean = reshape(mean(jContributions, 1), [5, 3]);
tContributionMean = reshape(mean(tContributions, 1), [5, 3]);
decompositionError = jContributionMean + tContributionMean - utilityMean;
assert(max(abs(decompositionError(:))) < 1e-12, ...
    'The two mean weighted contributions must sum to mean total U.');
selectionCounts = zeros(5, 3);
for k = 1:numel(names)
    selectionCounts(k, :) = sum(selectedIndices == k, 1);
end
assert(all(sum(selectionCounts, 1) == 3));
assert(all(utilitySD(3, :) < min(utilitySD([1, 2, 4, 5], :), [], 1)), ...
    'P2 must have the smallest observed sample SD at each tested weight.');
assert(all(reshape(scores(:, 3, 3), [3, 1]) == 1));
p2LatencySD = std(reshape(scores(:, 3, 4), [3, 1]), 0);
assert(max(abs(utilitySD(3, :) - (1 - lambdas) * p2LatencySD)) < 1e-12, ...
    'For P2 the normalized-J term is constant across these seeds.');
meanChanges = diff(utilityMean, 1, 2);
sdChanges = diff(utilitySD, 1, 2);
assert(all(all(meanChanges([3, 5], :) > 0)) ...
    && all(all(meanChanges([1, 2, 4], :) < 0)));
assert(all(sdChanges(:) < 0));

% Lambda is the main x axis, with five candidate bars at each tested weight.
% The two positive stacked segments are means of
% the per-seed weighted normalized J and T terms, not raw J or time in seconds.
% Total-U SD is computed directly from the three saved total utilities; it is
% not the sum of component SDs. Show the full symmetric intervals even where
% mean minus SD is negative. No cumulative sum across weights is performed.
figureSize = [8.5, 5.8];
fig = figure('Visible', 'off', 'Color', 'white', 'Units', 'centimeters', ...
    'Position', [2, 2, figureSize], 'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);
ax = axes(fig, 'Position', [0.15, 0.23, 0.83, 0.59]);
hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8.5, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.010, 0.010], ...
    'Box', 'off', 'Layer', 'bottom', 'Color', 'white', ...
    'XGrid', 'off', 'YGrid', 'on', ...
    'GridColor', [0.72, 0.72, 0.72], 'GridAlpha', 0.25);
xPositions = [1:5, 7:11, 13:17];
xlim(ax, [0.35, 17.65]);
ylim(ax, [-0.22, 1.05]);
assert(min(utilityMean(:) - utilitySD(:)) > ax.YLim(1) ...
    && max(utilityMean(:) + utilitySD(:)) < 0.90);
yticks(ax, -0.2:0.2:0.8);
ytickformat(ax, '%.1f');
xticks(ax, [3, 9, 15]);
xticklabels(ax, {'0.25', '0.50', '0.75'});
ax.TickLabelInterpreter = 'tex';
xtickangle(ax, 0);
barValues = [jContributionMean(:), tContributionMean(:)];
barHandles = bar(ax, xPositions, barValues, 'stacked', 'BarWidth', 0.72);
for term = 1:2
    set(barHandles(term), 'FaceColor', termColors(term, :), ...
        'EdgeColor', 'white', 'LineWidth', 0.3);
    assert(isequal(barHandles(term).YData(:), barValues(:, term)));
end
errorbar(ax, xPositions, utilityMean(:)', utilitySD(:)', ...
    'LineStyle', 'none', 'Marker', 'none', ...
    'Color', [0.12, 0.12, 0.12], 'LineWidth', 0.65, 'CapSize', 2.8, ...
    'HandleVisibility', 'off');
yline(ax, 0, '-', 'Color', [0.30, 0.30, 0.30], 'LineWidth', 0.65, ...
    'HandleVisibility', 'off');
for w = 1:numel(lambdas)
    for k = 1:numel(names)
        index = k + 5 * (w - 1);
        fontWeight = 'normal';
        if k == 3
            fontWeight = 'bold';
        end
        text(ax, xPositions(index), 0.95, labels{k}, ...
            'FontName', 'Times New Roman', 'FontSize', 8, ...
            'FontWeight', fontWeight, 'Interpreter', 'tex', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end
xlabel(ax, '\lambda', 'FontName', 'Times New Roman', ...
    'FontSize', 9.5, 'Interpreter', 'tex');
ylabel(ax, 'Mean joint utility, {\it U}', 'FontName', 'Times New Roman', ...
    'FontSize', 9, 'Interpreter', 'tex');
% Independent square swatches avoid stacked-bar legend clipping in PDF export.
legendHandles = gobjects(1, 2);
for term = 1:2
    legendHandles(term) = plot(ax, NaN, NaN, 'LineStyle', 'none', ...
        'Marker', 's', 'MarkerSize', 6, 'Color', termColors(term, :), ...
        'MarkerFaceColor', termColors(term, :), ...
        'MarkerEdgeColor', termColors(term, :));
end
leg = legend(ax, legendHandles, {'{\it J} contribution', '{\it T} contribution'}, ...
    'NumColumns', 2, ...
    'Orientation', 'horizontal', 'Box', 'off', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');
leg.ItemTokenSize = [10, 9];
leg.Units = 'normalized';
leg.Position = [0.16, 0.90, 0.82, 0.07];
drawnow;
outputStem = fullfile(figuresDir, 'experiment_round6_joint_utility');
exportgraphics(fig, [outputStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', figureSize, ...
    'PaperPosition', [0, 0, figureSize], 'PaperPositionMode', 'manual');
print(fig, [outputStem, '.png'], '-dpng', '-r600');
savefig(fig, fullfile(scriptDir, 'experiment_round6_joint_utility_column.fig'));

fprintf('PDF=%s.pdf\nPNG=%s.png\n', outputStem, outputStem);
fprintf('CHECK=9 completed raw candidate JSON records; 3 seeds x 3 weights\n');
fprintf('CHECK=per-seed history and J/T scores agree across weights\n');
fprintf('CHECK=all recorded utilities, choices and the seed-42 tie verified\n');
fprintf('CHECK=stacked means are lambda*J_norm and (1-lambda)*T_norm\n');
fprintf('CHECK=stack heights equal mean U for all 15 bars\n');
fprintf('CHECK=lambda is the main x axis; each group contains separate P0-P4 bars\n');
fprintf('CHECK=error bars use sample SD of total U, not summed component SDs\n');
fprintf('CHECK=every full mean-plus/minus-SD interval is visible\n');
fprintf('CHECK=P2 has the smallest SD at all three tested weights\n');
fprintf('CHECK=SD(U_P2)=(1-lambda)*SD(T_norm_P2); not a training-stability measure\n');
fprintf('CHECK=raw inputs and main.tex are read-only\n');
fprintf('FIGURE_SIZE_CM=%.1f,%.1f\n', figureSize);
fprintf('P2_T_NORM_SAMPLE_SD=%.12f\n', p2LatencySD);
for w = 1:numel(lambdas)
    for k = 1:numel(names)
        fprintf('LAMBDA=%.2f CANDIDATE=%s J_TERM=%.12f T_TERM=%.12f MEAN=%.12f SD=%.12f SELECTED=%d/3\n', ...
            lambdas(w), names{k}, jContributionMean(k, w), tContributionMean(k, w), ...
            utilityMean(k, w), utilitySD(k, w), selectionCounts(k, w));
    end
end
end
