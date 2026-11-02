中文组会 Beamer 汇报
===================
题目：层次化联邦学习中的动态协作模式选择
作者：杨婷婷 / Tingting Yang
日期：2026年10月26日

材料
----
main.pdf                 可直接全屏展示的16:9幻灯片
main.tex                 可编辑LaTeX Beamer源码（UTF-8）
assets/                  5张矢量PDF实验图、1张联邦学习架构图和2张云更新单元分组图，自带相对路径
presenter_notes.txt      逐页讲稿提要
assets/manifest.json     图文件来源与SHA256校验记录

汇报结构与时长
--------------
主体27页，建议25—30分钟；另有4页答疑备查。
1—3页：工作概述与研究背景，约3分钟。
4—6页：七模式初衷、误差评价障碍与三模式调整，约4分钟。
7—18页：当前问题、三模式、云更新单元、历史评分及算法，约10分钟。
19—25页：实验设计、主要结果与训练工作量分析，约7分钟。
26—27页：当前结论与下一步讨论，约3分钟。
28—31页：预算公式、15种策略结果、历史EMA与统计口径，按需展示。

编译
----
编译器：XeLaTeX。不要用pdfLaTeX。
在本目录执行两次：
  xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
或将整个源码包上传Overleaf，在设置中选择XeLaTeX编译器。
使用ctex与Fandol字体，无需Windows专用字体，也无需BibTeX。
修改数据、公式或图后请重新编译并检查页面。

讲稿备注
--------
源码各frame含\note{...}。普通PDF默认隐藏备注。
如需右侧演讲者屏幕，将：
  \setbeameroption{hide notes}
改为：
  \setbeameroption{show notes on second screen=right}
再重新编译。presenter_notes.txt可直接阅读。

内容依据
--------
研究过程引入参考七模式稿及作者说明：
  E:\YTT\GROUP\DynFL-reconfiguration\tex\paper\main.tex
当前三模式建模与全部实验结果依据：
  E:\YTT\GROUP\DynFL-privacy\paper\main.tex
七模式的split、HE和联合误差模型作为初始方案介绍；后续实验来自三模式稿。
J_learn是学习误差目标，当前J是历史更新偏差信号，两者的定义和偏好方向不同。
主实验45次，短期pilot单独报告。
原论文与原图未修改，配图均为复制件。所有新文件位于独立汇报目录。
系统时间是事件仿真时间；准确率按91—100轮先在种子内平均，再跨种子汇总。
未来工作与当前完成内容已分开。

原论文：E:\YTT\GROUP\DynFL-privacy\paper\main.tex
SHA256：EF7421988ED2194163A25A4281273404A2E0487AD817F2F19EDDE8310D43A726

七模式引入稿SHA256：B90F48AD7307E4990A01BDF02BF5C62B284E6B8CA2B6F6BFDD17D516FFD8D192
