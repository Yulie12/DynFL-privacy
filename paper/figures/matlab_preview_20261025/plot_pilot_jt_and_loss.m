function plot_pilot_jt_and_loss()
% Paired candidate J/T and observed loss, from the same completed pilot.
% Read-only experiment inputs. Does not edit main.tex or any existing figure.
% The two generated publication files belong to this function and can be
% regenerated. No interpolation, smoothing, averaging, or error bars are used.

scriptDir = fileparts(mfilename('fullpath'));
figuresDir = fileparts(scriptDir);
repoDir = fileparts(fileparts(figuresDir));
pilotDir = fullfile(repoDir, 'out', ...
    'fmnist_mode_switch_j_accuracy_pilot_10r_20261022');
names = {'P0', 'P1', 'P2', 'P3', 'P4'};
labels = {'P_0', 'P_1', 'P_2', 'P_3', 'P_4'};
branches = {'P0_fixed', 'P1_switch_r5', 'P2_switch_r5', ...
    'P3_switch_r5', 'P4_switch_r5'};
colors = [153, 153, 153; 213, 94, 0; 0, 114, 178; ...
    0, 158, 115; 204, 121, 167] / 255;
markers = {'s', '^', 'd', 'o', 'v'};
styles = {'-', '--', '-.', ':', '-'};
stepsPerRound = [320, 300, 400, 300, 360];

registration = jsondecode(fileread(fullfile(pilotDir, 'preregistration.json')));
assert(registration.seed == 42 && registration.config.rounds == 10 ...
    && registration.prefix_rounds == 5);
audit = jsondecode(fileread(fullfile(pilotDir, 'audit.json')));
assert(strcmp(audit.status, 'passed') && all([audit.checks.passed]), ...
    'The saved pilot audit must pass before its results can be plotted.');
candidateTable = readtable(fullfile(pilotDir, 'candidate_evaluations.csv'), ...
    'TextType', 'string');
exported = readtable(fullfile(pilotDir, 'round_metrics.csv'), 'TextType', 'string');
prefix = readtable(fullfile(pilotDir, 'shared_prefix', 'round_metrics.csv'), ...
    'VariableNamingRule', 'preserve');
integrity = readtable(fullfile(pilotDir, 'checkpoint_integrity.csv'), ...
    'TextType', 'string');
assert(height(prefix) == 5 && isequal(prefix.round, (1:5)'));
assert(height(exported) == 50 && height(candidateTable) == 5);
referenceDecision = jsondecode(fileread(fullfile(pilotDir, 'seed_42', ...
    branches{1}, 'candidates_round_005.json')));
assert(referenceDecision.decision_after_round == 5 ...
    && referenceDecision.applied_round == 6);
assert(numel(referenceDecision.candidates) == 5);

j = zeros(1, 5);
t = zeros(1, 5);
loss = zeros(10, 5);
accuracy = zeros(10, 5);
systemTime = zeros(10, 5);
for k = 1:5
    rawCandidate = referenceDecision.candidates( ...
        strcmp({referenceDecision.candidates.name}, names{k}));
    assert(isscalar(rawCandidate) && rawCandidate.feasible ...
        && strcmp(rawCandidate.status, 'ready'));
    j(k) = rawCandidate.J;
    t(k) = rawCandidate.T;
    savedCandidate = candidateTable(candidateTable.candidate == names{k}, :);
    assert(height(savedCandidate) == 1 ...
        && savedCandidate.decision_after_round == 5 ...
        && savedCandidate.applied_round == 6);
    assert(max(abs([j(k), t(k)] ...
        - [savedCandidate.j_pred, savedCandidate.system_latency_T])) < 1e-12);

    branchDir = fullfile(pilotDir, 'seed_42', branches{k});
    metrics = readtable(fullfile(branchDir, 'round_metrics.csv'), ...
        'VariableNamingRule', 'preserve');
    assert(height(metrics) == 10 && isequal(metrics.round, (1:10)'));
    assert(all(metrics.executed_local_steps(6:10) == stepsPerRound(k)));
    assert(all(metrics.max_epsilon <= registration.config.epsilon + 1e-8));
    for field = {'test_loss', 'test_accuracy', 'system_time', ...
            'round_duration', 'executed_local_steps'}
        column = field{1};
        assert(max(abs(metrics.(column)(1:5) - prefix.(column))) < 1e-12, ...
            'A branch does not share the registered five-round prefix.');
    end
    savedMetrics = sortrows(exported(exported.strategy == branches{k}, :), 'round');
    assert(height(savedMetrics) == 10 && isequal(savedMetrics.round, (1:10)'));
    for field = {'test_loss', 'test_accuracy', 'system_time', 'round_duration'}
        column = field{1};
        assert(max(abs(savedMetrics.(column) - metrics.(column))) < 1e-12);
    end
    decision = jsondecode(fileread(fullfile(branchDir, 'candidates_round_005.json')));
    assert(strcmp(decision.history_id, referenceDecision.history_id));
    assert(strcmp(decision.selected_candidate, names{k}));
    for candidateIndex = 1:5
        c = decision.candidates(strcmp({decision.candidates.name}, names{candidateIndex}));
        ref = referenceDecision.candidates( ...
            strcmp({referenceDecision.candidates.name}, names{candidateIndex}));
        assert(c.feasible && max(abs([c.J, c.T] - [ref.J, ref.T])) < 1e-12);
    end
    checkpoint = integrity(integrity.strategy == branches{k} & integrity.round == 5, :);
    assert(height(checkpoint) == 1 ...
        && checkpoint.sha256 == string(audit.shared_round5_checkpoint_sha256));
    assert(abs(metrics.predicted_j(6) - j(k)) < 1e-12);
    loss(:, k) = metrics.test_loss;
    accuracy(:, k) = metrics.test_accuracy;
    systemTime(:, k) = metrics.system_time;
end
assert(all(isfinite(j)) && all(isfinite(t)) && all(isfinite(loss(:))));
[~, jOrder] = sort(j, 'descend');
for r = 6:10
    [~, lossOrder] = sort(loss(r, :), 'ascend');
    assert(isequal(jOrder, lossOrder), 'The stated loss ranking must match every branch.');
end

fig = figure('Visible', 'off', 'Color', 'white', 'Units', 'centimeters', ...
    'Position', [2, 2, 17.8, 7.4], 'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
% Preserve whitespace and font embedding with exportgraphics in R2022b.
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);
leftPosition = [0.085, 0.20, 0.375, 0.60];
rightPosition = [0.575, 0.20, 0.395, 0.60];
axLeft = axes(fig, 'Position', leftPosition);
hold(axLeft, 'on');
styleAxes(axLeft);
for k = 1:5
    plot(axLeft, t(k), j(k), 'LineStyle', 'none', 'Color', colors(k, :), ...
        'Marker', markers{k}, 'MarkerSize', 6, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
        'LineWidth', 0.65);
end
xlim(axLeft, [1.75, 1.87]);
ylim(axLeft, [2.55, 4.85]);
xticks(axLeft, [1.76, 1.80, 1.84]);
yticks(axLeft, [2.75, 3.00, 3.50, 4.00, 4.50]);
xtickformat(axLeft, '%.2f');
ytickformat(axLeft, '%.2f');
xlabel(axLeft, 'System latency, {\it T} (s)', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');
ylabel(axLeft, 'Historical deviation contribution, {\it J}', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, 'Interpreter', 'tex');

axRight = axes(fig, 'Position', rightPosition);
hold(axRight, 'on');
styleAxes(axRight);
xlim(axRight, [0.85, 10.2]);
ylim(axRight, [2.15, 2.305]);
xticks(axRight, 1:10);
yticks(axRight, 2.15:0.03:2.30);
ytickformat(axRight, '%.2f');
% Shared measurements are drawn once. The gray region is the common
% P0 prefix, not an uncertainty band.
patch(axRight, [0.85, 5, 5, 0.85], [2.15, 2.15, 2.305, 2.305], ...
    [0.95, 0.95, 0.95], 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(axRight, 1:5, prefix.test_loss, '-', 'Color', [0.4, 0.4, 0.4], ...
    'LineWidth', 1.05, 'Marker', '.', 'MarkerSize', 5, ...
    'HandleVisibility', 'off');
curveHandles = gobjects(1, 5);
for k = 1:5
    curveHandles(k) = plot(axRight, 5:10, loss(5:10, k), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', 1.05, ...
        'Marker', markers{k}, 'MarkerSize', 3.8, 'MarkerIndices', 2:6, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', colors(k, :));
end
xline(axRight, 5, '--', 'Color', [0.45, 0.45, 0.45], ...
    'LineWidth', 0.8, 'HandleVisibility', 'off');
text(axRight, 2.9, 2.298, 'Common P_0', 'FontName', 'Times New Roman', ...
    'FontSize', 7.5, 'Color', [0.3, 0.3, 0.3], 'Interpreter', 'tex', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
xlabel(axRight, 'Round', 'FontName', 'Times New Roman', 'FontSize', 8.5);
ylabel(axRight, 'Test loss', 'FontName', 'Times New Roman', 'FontSize', 8.5);

lgd = legend(axRight, curveHandles, labels, 'NumColumns', 5, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontName', 'Times New Roman', ...
    'FontSize', 8.5, 'Interpreter', 'tex');
lgd.Units = 'normalized';
lgd.Position = [0.24, 0.925, 0.55, 0.05];
lgd.ItemTokenSize = [16, 8];
axLeft.Position = leftPosition;
axRight.Position = rightPosition;
annotation(fig, 'textbox', [0.07, 0.835, 0.405, 0.045], ...
    'String', '(a) Candidate scores at round 6', 'EdgeColor', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 9, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Margin', 0);
annotation(fig, 'textbox', [0.565, 0.835, 0.415, 0.045], ...
    'String', '(b) Loss before and after switching', 'EdgeColor', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 9, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Margin', 0);

drawnow;
outputStem = fullfile(figuresDir, 'experiment_pilot_JT_and_loss');
exportgraphics(fig, [outputStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [17.8, 7.4], ...
    'PaperPosition', [0, 0, 17.8, 7.4], 'PaperPositionMode', 'manual');
print(fig, [outputStem, '.png'], '-dpng', '-r600');
savefig(fig, fullfile(scriptDir, 'experiment_pilot_JT_and_loss.fig'));

fprintf('PDF=%s.pdf\nPNG=%s.png\n', outputStem, outputStem);
fprintf('SOURCE=single seed 42, same completed ten-round pilot\n');
fprintf('CHECK=all 434 saved audit checks passed\n');
fprintf('CHECK=all branches share round-5 checkpoint and candidate history\n');
fprintf('CHECK=raw branch metrics equal saved pilot export\n');
fprintf('CHECK=all existing experiment inputs are read-only\n');
fprintf('J_DESC=%s\n', strjoin(names(jOrder), '>'));
fprintf('ROUND5_LOSS=%.12f\n', prefix.test_loss(5));
for k = 1:5
    fprintf('%s J=%.12f T=%.12f LOSS10=%.12f LOSS_REDUCTION=%.12f ACC10=%.4f CUM_T10=%.12f STEPS10=%d\n', ...
        names{k}, j(k), t(k), loss(10, k), loss(5, k)-loss(10, k), ...
        accuracy(10, k), systemTime(10, k), 1600+5*stepsPerRound(k));
end
end

function styleAxes(ax)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'LineWidth', 0.7, 'TickDir', 'out', 'TickLength', [0.014, 0.014], ...
    'Box', 'off', 'Layer', 'top', 'Color', 'white');
grid(ax, 'on');
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.25;
end
