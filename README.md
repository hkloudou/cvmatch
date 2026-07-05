# cvmatch assets

Binary assets referenced by the main branch README, kept out of `main` to
keep the code branch lean.

- `demo/` — match results rendered by `bench/cmd/annotate`: each parent
  image with a green box drawn at the best-match location found by
  `cvmatch.Match`, plus the template (`*.tpl.png`) that was searched for.
- `samples/` — the original sample photographs from
  [OpenCV samples/data](https://github.com/opencv/opencv/tree/4.12.0/samples/data)
  (Apache-2.0), mirrored here so they are viewable in the repo.
