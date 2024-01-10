# Contributing

We are thrilled to have you join in making this project even better. Please feel free to browse through the resources and guidelines provided here, and let us know if there is anything specific you would like to contribute or discuss.

If you would like to help to develop this package you can skim through the to-do list below as well as the contribution guidelines. Just fork the project, add your code and send a pull request. We are always happy to get some help :thumbsup: !

If you encountered an issue using the `chirpdetector`, feel free to open an issue [here](https://github.com/weygoldt/chirpdetector/issues).

## Contributors guidelines

I try our best to adhere to good coding practices and catch up on writing tests
for this package. As I am currently the only one working on it, here is some
documentation of the development packages I use:

- `pre-commit` for pre-commit hooks
- `pytest` and `pytest-coverage` for unit tests
- `ruff` for linting and formatting
- `pyright` for static type checking

Before every commit, a pre-commit hook runs all these packages on the code base
and refuses a push if errors are raised. If you want to contribute, please
make sure that your code is proberly formatted and run the tests before issuing
a pull request. The formatting guidelines should be automatically picked up by your
`ruff` installaton from the `pyproject.toml` file.

## To Do

After the first release, this section will be removed an tasks will be organized
as github issues. Until them, if you fixed something, please check it off on this
list before opening a pull request.

- [ ] Give each chirp in df its own ID
- [ ] Give each spectrogram snippet its own ID
- [ ] Save spectrogram snippet ID and chirp IDs
- [ ] Write data dumping for main loop
- [ ] Write an assingment benchmarking
- [ ] Write an assignmen GUI to create a ground truth
- [ ] Find out why current assignment algo is failing at raw = raw1 - raw2
- [ ] Try a random forest classifier on PCAed envelope extractions to assign chirps
- [ ] Try a small NN, might work better than the random forest in this case
- [ ] Finish a script to analyze the Json dumps from the training loop
- [ ] Update all the docstrings after refactoring.
- [ ] Move hardcoded params from assignment algo into config.toml
- [ ] Split the messy training loop into functions.
- [ ] Remove all pyright warnings.
- [ ] Build github actions CI/CD pipeline for codecov etc.
- [x] Implement multiprocessing in main detection loop: Compute a batch of
      spectrograms parallely and pipe them all through the detector. And do this
      simulatenously from multiple cores (if the GPU can receive tensors from multiple
      cores). - Note: Multiprocessing increased the execution time due to the back and forth
      between cpu and gpu (at least this is what google said.) But I batched
      detection at least.
- [x] Check execution time for all the detect functions, got really slow after refactoring for some reason.
- [x] Fix make test, fails after ruff run
- [x] Refactor train, detect, convert. All into much smaller functions. Move accesory functions to utils
- [x] Make complete codebase pass ruff
- [x] Move the dataconverter from `gridtools` to `chirpdetector`
- [x] Extend the dataconverter to just output the spectrograms so that hand-labelling can be done in a separate step
- [x] Add a main script so that the cli is `chirpdetector <task> --<flag> <args>`
- [x] Improve simulation of chirps to include more realistic noise, undershoot and maybe even phasic-tonic evolution of the frequency of the big chirps
- [x] make the `copyconfig` script more
- [x] start writing the chirp assignment algorithm
- [x] Move all the pprinting and logging constructors to a separate module and build a unified console object so that saving logs to file is easier, also log to file as well
- [x] Add label-studio
- [x] Supply scripts to convert completely unannotated or partially annotated data to the label-studio format to make manual labeling easier
- [x] Make possible to output detections as a yolo dataset
- [x] Look up how to convert a yolo dataset to a label-studio input so we can label pre-annotated data, facilitating a full human-in-the-loop approach
- [x] Add augmentation transforms to the dataset class and add augmentations to the simulation in `gridtools`. Note to this: Unnessecary, using real data.
- [x] Change bbox to actual yolo format, not the weird one I made up (which is x1, y1, x2, y2 instead of x1, y1, w, h). This is why the label-studio export is not working.
- [x] Port cli to click, works better
- [x] Try clustering the detected chirp windows on a spectrogram, could be interesting
