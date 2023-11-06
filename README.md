# chirpdetector 
... detects brief communication signals of weakly electric fish.

## Todo

- [x] Move the dataconverter from `gridtools` to `chirpdetector`
- [x] Extend the dataconverter to just output the spectrograms so that hand-labelling can be done in a separate step
- [x] Add a main script so that the cli is `chirpdetector <task> --<flag> <args>`
- [ ] Improve simulation of chirps to include more realistic noise, undershoot and maybe even phasic-tonic evolution of the frequency of the big chirps
- [x] make the `copyconfig` script more
- [ ] start writing the chirp assignment algorithm
- [x] Move all the pprinting and logging constructors to a separate module and build a unified console object so that saving logs to file is easier, also log to file as well
- [ ] Split the messy training loop into functions 
- [x] Add label-studio 
- [ ] Supply scripts to convert completely unannotated or partially annotated data to the label-studio format to make manual labeling easier
- [x] Make possible to output detections as a yolo dataset 
- [ ] Look up how to convert a yolo dataset to a label-studio input so we can label pre-annotated data, facilitating a full human-in-the-loop approach
- [ ] Add augmentation transforms to the dataset class and add augmentations to the simulation in `gridtools`
- [ ] Change bbox to actual yolo format, not the weird one I made up (which is x1, y1, x2, y2 instead of x1, y1, w, h). This is why the label-studio export is not working.
