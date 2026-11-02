function plot_pilot_loss_jt_inset()
% Main loss trajectory and linked pre-round-6 J/T inset from one pilot.
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
    'Position', [2, 2, 17.8, 8.5], 'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);
mainPosition = [0.10, 0.16, 0.86, 0.74];
axMain = axes(fig, 'Position', mainPosition);
hold(axMain, 'on');
styleAxes(axMain);
axMain.FontSize = 9;
xlim(axMain, [0.85, 10.25]);
ylim(axMain, [2.15, 2.305]);
xticks(axMain, 1:10);
axMain.TickLabelInterpreter = 'tex';
xticklabels(axMain, {'1','2','3','4','5','\bf6','7','8','9','10'});
yticks(axMain, 2.15:0.03:2.30);
ytickformat(axMain, '%.2f');

% The gray area represents only the common P0 prefix, not uncertainty.
patch(axMain, [0.85, 5, 5, 0.85], [2.15, 2.15, 2.305, 2.305], ...
    [0.96, 0.96, 0.96], 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(axMain, 1:5, prefix.test_loss, '-', 'Color', [0.4, 0.4, 0.4], ...
    'LineWidth', 1.1, 'Marker', '.', 'MarkerSize', 6, ...
    'HandleVisibility', 'off');
curveHandles = gobjects(1, 5);
for k = 1:5
    curveHandles(k) = plot(axMain, 5:10, loss(5:10, k), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', 1.15, ...
        'Marker', markers{k}, 'MarkerSize', 4.5, 'MarkerIndices', 2:6, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', colors(k, :));
end
text(axMain, 2.6, 2.299, 'Common P_0', 'FontName', 'Times New Roman', ...
    'FontSize', 8.5, 'Color', [0.3, 0.3, 0.3], 'Interpreter', 'tex', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');
xlabel(axMain, 'Round', 'FontName', 'Times New Roman', 'FontSize', 9.5);
ylabel(axMain, 'Test loss', 'FontName', 'Times New Roman', 'FontSize', 9.5);

% Highlight the actual round-6 observations without moving their positions.
% The inset title explicitly locates scoring before execution of this round.
nodeBottom = min(loss(6, :)) - 0.0017;
nodeTop = max(loss(6, :)) + 0.0017;
guideColor = [0.36, 0.36, 0.36];
plot(axMain, [6, 6], [2.15, nodeBottom], '--', ...
    'Color', guideColor, 'LineWidth', 0.75, 'HandleVisibility', 'off');
rectangle(axMain, 'Position', [5.85, nodeBottom, 0.30, nodeTop-nodeBottom], ...
    'Curvature', [1, 1], 'EdgeColor', guideColor, 'LineWidth', 0.75, ...
    'LineStyle', '-', 'HandleVisibility', 'off');

lgd = legend(axMain, curveHandles, labels, 'NumColumns', 5, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontName', 'Times New Roman', ...
    'FontSize', 9, 'Interpreter', 'tex');
lgd.Units = 'normalized';
lgd.Position = [0.25, 0.94, 0.55, 0.045];
lgd.ItemTokenSize = [17, 8];
axMain.Position = mainPosition;

% The inset lies in the empty upper-right area, above all loss trajectories.
insetPosition = [0.64, 0.64, 0.30, 0.23];
axInset = axes(fig, 'Position', insetPosition);
hold(axInset, 'on');
styleAxes(axInset);
set(axInset, 'FontSize', 7.5, 'Box', 'on', 'LineWidth', 0.65, ...
    'TickLength', [0.018, 0.018], 'Color', 'white');
for k = 1:5
    plot(axInset, t(k), j(k), 'LineStyle', 'none', 'Color', colors(k, :), ...
        'Marker', markers{k}, 'MarkerSize', 4.6, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', [0.2, 0.2, 0.2], ...
        'LineWidth', 0.55);
end
xlim(axInset, [1.75, 1.87]);
ylim(axInset, [2.55, 4.85]);
xticks(axInset, [1.76, 1.80, 1.84]);
yticks(axInset, [3.0, 4.0, 4.5]);
xtickformat(axInset, '%.2f');
ytickformat(axInset, '%.1f');
xlabel(axInset, '{\it T} (s)', 'FontName', 'Times New Roman', ...
    'FontSize', 8, 'Interpreter', 'tex');
ylabel(axInset, '{\it J}', 'FontName', 'Times New Roman', ...
    'FontSize', 8, 'Interpreter', 'tex');
title(axInset, 'J and T before round 6', 'FontName', 'Times New Roman', ...
    'FontSize', 8.5, 'FontWeight', 'normal', 'Interpreter', 'none');

% This is a visual link to round 6, not a statement that scores use its loss.
nodeFigureX = mainPosition(1) + (6-0.85)/(10.25-0.85)*mainPosition(3);
nodeFigureY = mainPosition(2) + (nodeTop-2.15)/(2.305-2.15)*mainPosition(4);
annotation(fig, 'arrow', [nodeFigureX, insetPosition(1)+0.01], ...
    [nodeFigureY, insetPosition(2)-0.008], 'Color', guideColor, ...
    'LineWidth', 0.8, 'HeadLength', 5, 'HeadWidth', 5);

drawnow;
outputStem = fullfile(figuresDir, 'experiment_pilot_loss_JT_inset');
exportgraphics(fig, [outputStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', [17.8, 8.5], ...
    'PaperPosition', [0, 0, 17.8, 8.5], 'PaperPositionMode', 'manual');
print(fig, [outputStem, '.png'], '-dpng', '-r600');
savefig(fig, fullfile(scriptDir, 'experiment_pilot_loss_JT_inset.fig'));

fprintf('PDF=%s.pdf\nPNG=%s.png\n', outputStem, outputStem);
fprintf('SOURCE=same completed ten-round pilot for both the main plot and inset\n');
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


