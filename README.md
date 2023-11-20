<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/weygoldt/chirpdetector">
    <img src="assets/logo.svg" alt="Logo" width="100" height="100">
  </a>

<!-- <br /> -->
<!-- <h1 align="center">Chirpdetector</h1> -->

  <p align="center">
    Detect brief communication signals of wave-type weakly electric fish using deep neural networks.
    <br />
    <a href="https://weygoldt.com/chirpdetector"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://weygoldt.com/chirpdetector">View Demo</a>
    |
    <a href="https://github.com/weygoldt/chirpdetector/issues">Report Bug</a>
    |
    <a href="https://github.com/weygoldt/chirpdetector/issues">Request Feature</a>
  </p>
</div>

![Codecov](https://img.shields.io/codecov/c/github/weygoldt/chirpdetector)

## The Problem

Chirps are by far the most thoroughly studied communication signal of weakly electric, if not all fish. But as soon as the electric fields of more than one fish is recorded on the same electrode, detecting chirps becomes so hard, that most of the research to date analyzes this signal in isolated individuals. This is not particularly advantageous if the focus lies on the communication aspect of this signal. 

## The Solution

To tackle this isse, this package provides an interface to detect chirps of multiple fish on spectrogram images. This enables the quantitative analysis of chirping between freely interacting fish for the first time.

This project is still work in progress and will approximately be released in spring of 2024.

## TODO

### Urgent!!!
- [ ] Refactor train, detect, convert. All into much smaller functions. Move accesory functions to utils 

- [x] Move the dataconverter from `gridtools` to `chirpdetector`
- [x] Extend the dataconverter to just output the spectrograms so that hand-labelling can be done in a separate step
- [x] Add a main script so that the cli is `chirpdetector <task> --<flag> <args>`
- [ ] Improve simulation of chirps to include more realistic noise, undershoot and maybe even phasic-tonic evolution of the frequency of the big chirps
- [x] make the `copyconfig` script more
- [ ] start writing the chirp assignment algorithm
- [x] Move all the pprinting and logging constructors to a separate module and build a unified console object so that saving logs to file is easier, also log to file as well
- [ ] Split the messy training loop into functions 
- [x] Add label-studio 
- [x] Supply scripts to convert completely unannotated or partially annotated data to the label-studio format to make manual labeling easier
- [x] Make possible to output detections as a yolo dataset 
- [x] Look up how to convert a yolo dataset to a label-studio input so we can label pre-annotated data, facilitating a full human-in-the-loop approach
- [ ] Add augmentation transforms to the dataset class and add augmentations to the simulation in `gridtools`
- [x] Change bbox to actual yolo format, not the weird one I made up (which is x1, y1, x2, y2 instead of x1, y1, w, h). This is why the label-studio export is not working.
- [x] Port cli to click, works better
