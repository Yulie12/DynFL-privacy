function plot_pilot_loss_jt_folded()
% Folded initialization and J/T inset linked to the common round-5 state.
% Read-only experiment inputs. Does not edit main.tex or other figures.
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
% Main plot, legend and inset use exactly the same marker properties.
candidateMarkerStyle = cell(1, 5);
for k = 1:5
    candidateMarkerStyle{k} = {'LineStyle', 'none', ...
        'Color', colors(k, :), 'Marker', markers{k}, 'MarkerSize', 4.2, ...
        'MarkerFaceColor', colors(k, :), 'MarkerEdgeColor', colors(k, :), ...
        'LineWidth', 0.55};
end
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
% A candidate is dominated only if another is no worse in both objectives
% and strictly better in at least one: minimize T and maximize J.
nondominated = true(1, 5);
for k = 1:5
    dominatesK = t <= t(k) & j >= j(k) & (t < t(k) | j > j(k));
    nondominated(k) = ~any(dominatesK);
end
frontierIndices = find(nondominated);
[~, frontierOrder] = sort(t(frontierIndices), 'ascend');
frontierIndices = frontierIndices(frontierOrder);
assert(isequal(frontierIndices, [4, 1, 3]), ...
    'The saved candidate scores must give the P3, P0, P2 boundary.');
[~, jOrder] = sort(j, 'descend');
for r = 6:10
    [~, lossOrder] = sort(loss(r, :), 'ascend');
    assert(isequal(jOrder, lossOrder), 'The stated loss ranking must match every branch.');
end

% Design at the final single-column physical size so typography is not
% halved when included with width=0.96\columnwidth.
figureSize = [8.5, 7.8];
fig = figure('Visible', 'off', 'Color', 'white', 'Units', 'centimeters', ...
    'Position', [2, 2, figureSize], 'InvertHardcopy', 'off');
cleanupFigure = onCleanup(@() close(fig)); %#ok<NASGU>
annotation(fig, 'rectangle', [0.001, 0.001, 0.998, 0.998], ...
    'Color', 'white', 'FaceColor', 'none', 'LineWidth', 0.1);

% Retain the actual round-5 endpoint. Earlier rounds are explicitly omitted,
% not joined into a made-up compressed slope. Rounds 5 to 10 remain linear.
mainPosition = [0.17, 0.18, 0.79, 0.66];
mainX = [4.75, 10.20];
mainY = [2.15, 2.28];
axMain = axes(fig, 'Position', mainPosition);
hold(axMain, 'on');
styleAxes(axMain);
axMain.FontSize = 9;
xlim(axMain, mainX);
ylim(axMain, mainY);
xticks(axMain, 5:10);
axMain.TickLabelInterpreter = 'tex';
xticklabels(axMain, {'\bf5','6','7','8','9','10'});
yticks(axMain, [2.15, 2.17, 2.19, 2.21, 2.23, 2.25, 2.27]);
ytickformat(axMain, '%.2f');

curveHandles = gobjects(1, 5);
mainMarkerHandles = gobjects(1, 5);
for k = 1:5
    curveHandles(k) = plot(axMain, 5:10, loss(5:10, k), ...
        'Color', colors(k, :), 'LineStyle', styles{k}, 'LineWidth', 1.1, ...
        'Marker', 'none', 'HandleVisibility', 'off');
end
for k = 1:5
    mainMarkerHandles(k) = plot(axMain, 6:10, loss(6:10, k), ...
        candidateMarkerStyle{k}{:});
end
commonLoss = prefix.test_loss(5);
plot(axMain, 5, commonLoss, 'o', 'Color', [0.25, 0.25, 0.25], ...
    'MarkerFaceColor', [0.45, 0.45, 0.45], 'MarkerSize', 4.5, ...
    'LineWidth', 0.7, 'HandleVisibility', 'off');
text(axMain, 5.10, 2.276, 'P_0 ends', ...
    'FontName', 'Times New Roman', 'FontSize', 8.5, ...
    'Color', [0.3, 0.3, 0.3], 'Interpreter', 'tex', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
xlabel(axMain, 'Round', 'FontName', 'Times New Roman', 'FontSize', 9.5);
ylabel(axMain, 'Test loss', 'FontName', 'Times New Roman', 'FontSize', 9.5);

guideColor = [0.36, 0.36, 0.36];
plot(axMain, [5, 5], [mainY(1), commonLoss-0.0038], '--', ...
    'Color', guideColor, 'LineWidth', 0.75, 'HandleVisibility', 'off');
lgd = legend(axMain, mainMarkerHandles, labels, 'NumColumns', 5, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontName', 'Times New Roman', ...
    'FontSize', 9, 'Interpreter', 'tex');
lgd.Units = 'normalized';
lgd.Position = [0.20, 0.945, 0.75, 0.04];
lgd.ItemTokenSize = [10, 6];
axMain.Position = mainPosition;

% A physical circle is centered precisely on the actual common round-5 point.
nodeFigureX = mainPosition(1) + (5-mainX(1))/diff(mainX)*mainPosition(3);
nodeFigureY = mainPosition(2) + (commonLoss-mainY(1))/diff(mainY)*mainPosition(4);
circleWidth = 0.047;
circleHeight = circleWidth * figureSize(1) / figureSize(2);
annotation(fig, 'ellipse', [nodeFigureX-circleWidth/2, ...
    nodeFigureY-circleHeight/2, circleWidth, circleHeight], ...
    'Color', guideColor, 'LineWidth', 0.85);

% The break is before the first displayed round, never between rounds 5 and 6.
breakX = mainPosition(1) + 0.010;
breakY = mainPosition(2);
annotation(fig, 'rectangle', [breakX-0.003, breakY-0.0045, 0.020, 0.009], ...
    'FaceColor', 'white', 'EdgeColor', 'none');
for shift = [0, 0.009]
    annotation(fig, 'line', [breakX+shift, breakX+shift+0.006], ...
        [breakY-0.007, breakY+0.007], 'Color', [0.15, 0.15, 0.15], ...
        'LineWidth', 0.8);
end
annotation(fig, 'textbox', [0.17, 0.025, 0.72, 0.04], ...
    'String', 'Rounds 1 to 4 collapsed', 'EdgeColor', 'none', ...
    'FontName', 'Times New Roman', 'FontSize', 8, 'Color', [0.4, 0.4, 0.4], ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Margin', 0);

insetPosition = [0.50, 0.623, 0.45, 0.27];
axInset = axes(fig, 'Position', insetPosition);
hold(axInset, 'on');
styleAxes(axInset);
set(axInset, 'FontSize', 8, 'Box', 'on', 'LineWidth', 0.7, ...
    'TickLength', [0.018, 0.018], 'Color', 'white');
% Descriptive boundary of the observed nondominated points only. The line
% does not add feasible mixtures or represent an online optimization step.
plot(axInset, t(frontierIndices), j(frontierIndices), '--', ...
    'Color', [0.40, 0.40, 0.40], 'LineWidth', 0.85, ...
    'HandleVisibility', 'off');
insetMarkerHandles = gobjects(1, 5);
for k = 1:5
    insetMarkerHandles(k) = plot(axInset, t(k), j(k), ...
        candidateMarkerStyle{k}{:});
    for property = {'Marker', 'MarkerSize', 'MarkerFaceColor', ...
            'MarkerEdgeColor', 'Color', 'LineWidth', 'LineStyle'}
        assert(isequal(get(mainMarkerHandles(k), property{1}), ...
            get(insetMarkerHandles(k), property{1})), ...
            'The main plot, legend and inset must share exact marker styles.');
    end
end
xlim(axInset, [1.75, 1.87]);
ylim(axInset, [2.55, 4.85]);
xticks(axInset, [1.76, 1.80, 1.84]);
yticks(axInset, [3.0, 4.0, 4.5]);
xtickformat(axInset, '%.2f');
ytickformat(axInset, '%.1f');
xLabelInset = xlabel(axInset, '{\it T} (s)', ...
    'FontName', 'Times New Roman', 'FontSize', 9, 'Interpreter', 'tex');
set(xLabelInset, 'Units', 'normalized', 'Position', [1, -0.14, 0], ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');
yLabelInset = ylabel(axInset, '{\it J}', 'Rotation', 0, ...
    'FontName', 'Times New Roman', 'FontSize', 9, 'Interpreter', 'tex');
set(yLabelInset, 'Units', 'normalized', 'Position', [-0.14, 0.5, 0], ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
title(axInset, 'J and T after round 5', 'FontName', 'Times New Roman', ...
    'FontSize', 8.5, 'FontWeight', 'normal', 'Interpreter', 'none');
% Two light-gray arrows link the common state to the left inset corners.
% Match the restrained connector color of the fixed-loss reference figure.
% Circle attachment points are on its right upper/lower quadrants; arrow
% heads stop at the frame and do not extend into the candidate data.
connectorColor = [0.65, 0.65, 0.65];
for side = [1, -1]
    sourceX = nodeFigureX + circleWidth/2*cosd(30);
    sourceY = nodeFigureY + side*circleHeight/2*sind(30);
    targetX = insetPosition(1);
    targetY = insetPosition(2) + (side+1)/2*insetPosition(4);
    annotation(fig, 'arrow', [sourceX, targetX], [sourceY, targetY], ...
        'Color', connectorColor, 'LineWidth', 0.8, ...
        'HeadLength', 4.5, 'HeadWidth', 4.5);
end

drawnow;
outputStem = fullfile(figuresDir, 'experiment_pilot_loss_JT_folded');
exportgraphics(fig, [outputStem, '.pdf'], 'ContentType', 'vector', ...
    'BackgroundColor', 'white');
set(fig, 'PaperUnits', 'centimeters', 'PaperSize', figureSize, ...
    'PaperPosition', [0, 0, figureSize], 'PaperPositionMode', 'manual');
print(fig, [outputStem, '.png'], '-dpng', '-r600');
savefig(fig, fullfile(scriptDir, 'experiment_pilot_loss_JT_folded.fig'));

fprintf('PDF=%s.pdf\nPNG=%s.png\n', outputStem, outputStem);
fprintf('SOURCE=same completed ten-round pilot for both the main plot and inset\n');
fprintf('CHECK=all 434 saved audit checks passed\n');
fprintf('CHECK=all branches share round-5 checkpoint and candidate history\n');
fprintf('CHECK=raw branch metrics equal saved pilot export\n');
fprintf('CHECK=all existing experiment inputs are read-only\n');
fprintf('CHECK=main plot, marker-only legend and inset have identical marker properties\n');
fprintf('J_DESC=%s\n', strjoin(names(jOrder), '>'));
fprintf('NONDOMINATED_T_ASC=%s\n', strjoin(names(frontierIndices), '>'));
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
    'Box', 'off', 'Layer', 'bottom', 'Color', 'white');
grid(ax, 'on');
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.25;
end




