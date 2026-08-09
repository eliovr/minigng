# Changelog

## [0.1.0](https://github.com/eliovr/minigng/compare/v0.0.3...v0.1.0) (2026-08-09)


### Features

* improve MiniGNG API safety, reset semantics, and export robustness ([#15](https://github.com/eliovr/minigng/issues/15)) ([d0124de](https://github.com/eliovr/minigng/commit/d0124ded181e5c8e6d0440199dc2a2ca58359305))


### Performance Improvements

* **gng:** share prototype matrix and rank on squared distance ([#11](https://github.com/eliovr/minigng/issues/11)) ([932ea69](https://github.com/eliovr/minigng/commit/932ea690215a3b612da309ac70f6fbd3c0840c90))
* **gng:** vectorize predict/fit assignment and skip per-signal edge rebuild ([#13](https://github.com/eliovr/minigng/issues/13)) ([f9be941](https://github.com/eliovr/minigng/commit/f9be94190541158de75a04073303f5d8982197b3))

## [0.0.3](https://github.com/eliovr/minigng/compare/v0.0.2...v0.0.3) (2026-06-01)


### Bug Fixes

* copy unit prototypes to avoid mutating input array X ([dcd9101](https://github.com/eliovr/minigng/commit/dcd91017f6129746e66c589efef6fd22be8f82da))
* correct unit index mapping in predict ([#5](https://github.com/eliovr/minigng/issues/5)) ([1d464b1](https://github.com/eliovr/minigng/commit/1d464b13756abf83a19eb1f97958ce69dac72fed))
* repair broken fit_predict signature ([#4](https://github.com/eliovr/minigng/issues/4)) ([dc3c5bc](https://github.com/eliovr/minigng/commit/dc3c5bc6ccfb190e597ed39a59b34c01a7d0a874))
* repair score() and sample signals without replacement ([#6](https://github.com/eliovr/minigng/issues/6)) ([f15536a](https://github.com/eliovr/minigng/commit/f15536ae6963e71db05cf14a1f4140783ecc836f))
* tidy unit seeding, predict return, RNG, types, and docs ([#7](https://github.com/eliovr/minigng/issues/7)) ([985ffac](https://github.com/eliovr/minigng/commit/985ffacd0ec762b0f0d62f9eebd6a39698d260fd))
