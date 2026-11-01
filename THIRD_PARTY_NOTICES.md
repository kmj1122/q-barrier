# Third-party notices

Original project additions are licensed under the MIT License in `LICENSE`. This repository also includes modified third-party source whose original license texts and copyright notices are retained. Those components remain governed by their respective licenses below.

## AMAGO

- Location: `amago/`
- Upstream: <https://github.com/UT-Austin-RPL/amago>
- Copyright: Jake Grigsby and other upstream contributors, as recorded in the source.
- License: MIT; see `amago/LICENSE`.
- This copy includes project-specific constrained-learning, shielding, and compatibility modifications. It is not an unmodified upstream release.

## OmniSafe Lagrange utility

- Location: `amago/amago/lagrange.py` (an exception to the AMAGO directory's MIT license).
- Copyright: 2023 OmniSafe Team; the file's original copyright and license header are retained.
- License: Apache License 2.0; the full license text is included at `vendor/safety-gymnasium/LICENSE`.

## Safety-Gymnasium

- Location: `vendor/safety-gymnasium/`
- Upstream: <https://github.com/PKU-Alignment/safety-gymnasium>
- Copyright: OmniSafe Team and other upstream contributors, as recorded in the source.
- License: Apache License 2.0; see `vendor/safety-gymnasium/LICENSE` and retained file notices.
- The bundled source includes project-specific velocity environments. Packaged assets are restricted to those needed by the supported paper benchmarks; other environments registered by the package may require omitted assets. Packaging dependency constraints may differ from the upstream distribution.

Other dependencies are installed separately and remain subject to their own licenses. Raw experiment archives obtained from external baseline authors are not included. `data/figures/` contains derived, method-attributed values used in the paper's plots; SCARED curves derive from data supplied by the SCARED authors. See that directory's provenance and uncertainty notes. Including these summaries does not relicense underlying third-party datasets.
