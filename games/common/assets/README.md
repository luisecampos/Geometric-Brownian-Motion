# assets/

Drop a logo image here named `logo.png`, `logo.jpg`, or `logo.jpeg` and the
corner badge shows up in every game that uses `games/common/badge.py`
(currently n_back, sequence_recall, probability_quickfire - `mental_math`
has its own separate copy at `games/mental_math/assets/`, since it predates
this shared module).

No file here yet -> badge falls back to a small drawn monogram.

Logo image files are git-ignored (see the root `.gitignore`) - a real
company's logo isn't something to keep in version control. This README is
the only thing meant to be tracked here.
