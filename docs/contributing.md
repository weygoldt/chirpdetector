# Contributing

We are thrilled to have you join in making this project even better. Please
feel free to browse through the resources and guidelines provided here, and let
us know if there is anything specific you would like to contribute or discuss.

If you would like to help to develop this package you can skim through the
to-do list below as well as the contribution guidelines. Just fork the project,
add your code and send a pull request. We are always happy to get some help
:thumbsup: !

If you encountered an issue using the `chirpdetector`, feel free to open an
issue [here](https://github.com/weygoldt/chirpdetector/issues).

## Contributors guidelines

I try our best to adhere to good coding practices and catch up on writing tests
for this package. As I am currently the only one working on it, here is some
documentation of the development packages I use:

- `pre-commit` for pre-commit hooks
- `pytest` and `pytest-coverage` for unit tests
- `ruff` for linting and formatting
- `pyright` for static type checking

Before every commit, a pre-commit hook runs all these packages on the code base
and refuses a push if errors are raised. If you want to contribute, please make
sure that your code is proberly formatted and run the tests before issuing a
pull request. The formatting guidelines should be automatically picked up by
your `ruff` installaton from the `pyproject.toml` file.

## To Do

After the first release, this section will be removed an tasks will be
organized as github issues. Until them, if you fixed something, please check it
off on this list before opening a pull request.

- [ ] Rebuild the logging, verbosity and progress bar system like in the wavetracker
- [ ] Idea: Instead of simple non maximum supression for the final output
      we could try max averaging: Run non-max supression with a specific
      threshold and then group overlapping bboxes and average their
      coordinates. This could produce better bboxes.
- [ ] Improve the simulation pipeline by adding more chirps and by buiding a
      chirpGAN. And then quantify how close the model is to human performance.
      This can apply to detection and assignment performance.
- [ ] The assignment model is an issue:
  - [ ] It is trained using BCE loss and has a sigmoid activation function,
        instead, it should be trained using BCE with logits loss and no activation
        function. This is because the sigmoid function is applied to the output
        of the model in the loss function, so it is redundant to apply it to the
        output of the model as well.
  - [ ] As there are much more "not-emitter" than "emitter" samples in the dataset,
        I simply took a subset so that the classes are balanced. This is not
        good practice. Instead, I should use the weighted BCE loss, which is
        implemented in pytorch as `torch.nn.BCEWithLogitsLoss`.
  - [ ] The model is a MLP but XGBoost or a Random Forest might be better suited
        for this task. I should try these models as well.
  - [ ] Despite the fact that this is a simple task, I did not get around doing
        any hyperparamter optimization. This should be implemented for the
        model type (MLP, XGBoost, Random Forest) and the hyperparameters of the
        model and the architecture of the MLP (at least).
- [ ] Add meta-evaluation modulue (model agnostic) that can be used to evaluate
      the performance of the detection model and assignment algorithm
      but more importantly, to compare different models and algorithms,
      and most importantly, give us an idea of which data in the validation
      dataset is not well detected so that we can improve the training data
      using quantitative measures.
- [ ] Move all dataframe operations to cudf/polars, a pandas-like dataframe library
      that runs on the GPU.
- [ ] Rethink the output: Needs to be a HDF5 file that not only includes
      chirp time and ID but also the full chirp spectrograms so that
      we can later cluster them nicely.
- [ ] Clean up training data converter, remove deprecated modules
- [ ] Convert all subsets to training dataset, fix annotations, build larger simulated dataset
- [ ] Move the waveform preprocessing stuff into a nn.module as suggested by  
       the torchaudio docs here: https://pytorch.org/audio/main/transforms.html
- [ ] Write data dumping for main loop
- [ ] Try different NN architectures for assignment
- [ ] Write an assignmen GUI to create a ground truth
- [ ] Find out why current assignment algo is failing at raw = raw1 - raw2
- [ ] Try a random forest classifier on PCAed envelope extractions to assign
      chirps
- [ ] Update all the docstrings after refactoring.
- [ ] Move hardcoded params from assignment algo into config.toml
- [ ] Split the messy training loop into functions or remove it all together
      and rely on external libraries for training. Regarding this:
      Only thing this package should do is genreate good training data, then
      train wiht external libraries (e.g. ultralytics, ...).
- [ ] Remove all pyright warnings.
- [ ] Build github actions CI/CD pipeline for codecov etc.
- [x] Finish a script to analyze the Json dumps from the training loop
- [x] Implement detector class that works with the trained yolov8
- [x] Write an assingment benchmarking
- [x] Try a small NN, might work better than the random forest in this case
- [x] Give each chirp in df its own ID
- [x] Give each spectrogram snippet its own ID
- [x] Save spectrogram snippet ID and chirp IDs
- [x] Implement multiprocessing in main detection loop: Compute a batch of
      spectrograms parallely and pipe them all through the detector. And do this
      simulatenously from multiple cores (if the GPU can receive tensors from
      multiple cores). - Note: Multiprocessing increased the execution time due to
      the back and forth between cpu and gpu (at least this is what google said.)
      But I batched detection at least.
- [x] Check execution time for all the detect functions, got really slow after
      refactoring for some reason.
- [x] Fix make test, fails after ruff run
- [x] Refactor train, detect, convert. All into much smaller functions. Move
      accesory functions to utils
- [x] Make complete codebase pass ruff
- [x] Move the dataconverter from `gridtools` to `chirpdetector`
- [x] Extend the dataconverter to just output the spectrograms so that
      hand-labelling can be done in a separate step
- [x] Add a main script so that the cli is `chirpdetector <task> --<flag>
<args>`
- [x] Improve simulation of chirps to include more realistic noise, undershoot
      and maybe even phasic-tonic evolution of the frequency of the big chirps
- [x] make the `copyconfig` script more
- [x] start writing the chirp assignment algorithm
- [x] Move all the pprinting and logging constructors to a separate module and
      build a unified console object so that saving logs to file is easier, also
      log to file as well
- [x] Add label-studio
- [x] Supply scripts to convert completely unannotated or partially annotated
      data to the label-studio format to make manual labeling easier
- [x] Make possible to output detections as a yolo dataset
- [x] Look up how to convert a yolo dataset to a label-studio input so we can
      label pre-annotated data, facilitating a full human-in-the-loop approach
- [x] Add augmentation transforms to the dataset class and add augmentations to
      the simulation in `gridtools`. Note to this: Unnessecary, using real data.
- [x] Change bbox to actual yolo format, not the weird one I made up (which is
      x1, y1, x2, y2 instead of x1, y1, w, h). This is why the label-studio export
      is not working.
- [x] Port cli to click, works better
- [x] Try clustering the detected chirp windows on a spectrogram, could be
      interesting
