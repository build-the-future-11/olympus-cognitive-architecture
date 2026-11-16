# Prior-art boundary

The following primary works bound the contribution; none proves that Olympus v6 works.

- Jacobs, Jordan, Nowlan and Hinton (1991), Adaptive Mixtures of Local Experts.
  https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf
  Input-dependent gates and expert mixtures are established methodology.
- Wolpert (1992), Stacked Generalization.
  https://www.machine-learning.martinsewell.com/ensembles/stacking/Wolpert1992.pdf
  Meta-learning from held-out predictions predates this study. Disjoint fitting
  is a conservative implementation choice, not a new general principle.
- Sagawa et al. (2020), Distributionally Robust Neural Networks for Group Shifts.
  https://arxiv.org/abs/1911.08731
  Worst-group optimization needs generalization controls; minimizing group
  training loss alone does not establish unseen-shift robustness. Our max-batch
  group objective is explicitly distinct from their adaptive-weight algorithm.
- Ovadia et al. (2019), Can You Trust Your Model's Uncertainty?
  https://arxiv.org/abs/1906.02530
  Confidence/calibration under shift must be measured; confident intervention
  does not imply reliable intervention.
- Mozannar and Sontag (2020), Consistent Estimators for Learning to Defer to an Expert.
  https://proceedings.mlr.press/v119/mozannar20b.html
  Expert deferral provides relevant decision-objective comparisons, but this
  study uses soft mixtures, not a human-deferral loss or its consistency result.

The intended result is a falsifiable mechanism comparison within Olympus's
preserved synthetic sequence. A positive pilot would motivate an independently
frozen external-data study, not establish SOTA or a full cognitive architecture.
